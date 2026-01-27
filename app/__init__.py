from flask import Flask, request, jsonify
import os
import uuid
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import subprocess
import time
from openai import OpenAI
import google.generativeai as genai
from app.helpers.utils import cleanup_video_files
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY into os.environvvv
key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
TXT_DIR = "txt_outputs"
# At the very top of the file, after imports
os.makedirs(TXT_DIR, exist_ok=True)
FRAME_INTERVAL_SECONDS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    attn_implementation="eager",           # Very important on many cloud setups
).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)
def florence_caption(frame_path: str, task: str = "<MORE_DETAILED_CAPTION>") -> str:
    """
    Caption a single frame from disk using Florence-2.
    
    Args:
        frame_path: Path to the saved .jpg frame
        task: Florence-2 task prompt (default: detailed caption)
    
    Returns:
        str: Generated caption (or fallback raw text if post-processing fails)
    """
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    try:
        # Load image – use RGB mode (Florence-2 expects 3 channels)
        pil_image = Image.open(frame_path).convert("RGB")
        
        # Reuse your existing logic (very good & stable)
        prompt = task
        
        inputs = processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        )
        
        if "pixel_values" not in inputs or inputs["pixel_values"] is None:
            raise ValueError(f"Processor failed to extract pixel_values from {frame_path}")
        
        # Move to correct device + half-precision if on GPU
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        if DEVICE.startswith("cuda"):
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        # Generate – greedy decoding (stable, deterministic)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,          # greedy
                do_sample=False,
                use_cache=False       # often more stable in container/cloud envs
            )
        
        # Decode including special tokens (needed for correct post-processing)
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        # Try structured post-processing (usually gives clean result)
        try:
            parsed = processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(pil_image.width, pil_image.height)
            )
            # For caption tasks → usually returns a plain string under the task key
            if isinstance(parsed, dict) and task in parsed:
                return parsed[task].strip()
            # Sometimes it's already a string
            if isinstance(parsed, str):
                return parsed.strip()
            return str(parsed).strip()  # fallback stringify
            
        except Exception as post_err:
            print(f"Post-process failed for {frame_path}: {post_err}")
            # Very common fallback – strip special tokens manually
            raw_caption = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            return raw_caption if raw_caption else "No description generated"

    except Exception as e:
        print(f"Error captioning {frame_path}: {e}")
        return f"Error: {str(e)}"
def create_app():
    app = Flask(__name__)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # ---------------------
    # Health Check
    # ---------------------
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "RunPod API running"}), 200

    # ---------------------
    # Video Upload Endpoint
    # ---------------------
    @app.route("/upload-video", methods=["POST"])
    def upload_video():
        if "video" not in request.files:
            return jsonify({"error": "No video provided"}), 400

        video = request.files["video"]
        video_id = str(uuid.uuid4())

        video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{video.filename}")
        video.save(video_path)

        frames_folder = os.path.join(FRAMES_DIR, video_id)
        os.makedirs(frames_folder, exist_ok=True)

        # Step 0: Extract frames (your existing code)
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video"}), 500

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_index = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (
                fps > 0 and
                frame_index % int(fps * FRAME_INTERVAL_SECONDS) == 0 and
                frame is not None and
                frame.size > 0
            ):
                frame_path = os.path.join(frames_folder, f"frame_{saved:05d}.jpg")
                success = cv2.imwrite(frame_path, frame)
                if success:
                    saved += 1
            frame_index += 1

        cap.release()

        # Step 1: Florence inference + check success
        florence_results = []
        florence_success = True  # assume success until error

        for frame_file in sorted(os.listdir(frames_folder)):
            frame_path = os.path.join(frames_folder, frame_file)
            try:
                caption = florence_caption(frame_path)
                florence_results.append({
                    "frame": frame_file,
                    "caption": caption
                })
                if "Error" in caption:  # or your error check
                    florence_success = False
            except Exception as e:
                florence_results.append({
                    "frame": frame_file,
                    "error": str(e)
                })
                florence_success = False

        if not florence_success:
            return jsonify({
                "video_id": video_id,
                "frames_saved": saved,
                "florence_results": florence_results,
                "status": "Florence failed – skipping Whisper and GPT-5.2"
            }), 200  # or 500 if you want failure status

        # Save Florence to TXT if success
        visual_txt_path = os.path.join(TXT_DIR, f"{video_id}_visual.txt")
        with open(visual_txt_path, "w", encoding="utf-8") as f:
            for res in florence_results:
                f.write(f"{res['frame']}: {res['caption']}\n")

        # Step 2: Whisper transcription (only if Florence OK)
        audio_path = os.path.join(UPLOAD_DIR, f"{video_id}_audio.mp3")
        whisper_transcript = "No audio or transcription available"
        whisper_success = False

        try:
            # import os  # make sure it's imported

            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

            client = OpenAI(api_key=key)

            if file_size_mb <= 25:
                # Preferred: send original video directly (supported!)
                with open(video_path, "rb") as direct_file:
                    whisper_response = client.audio.translations.create(
                        model="whisper-1",
                        file=direct_file,
                        response_format="text",
                        prompt="Translate to natural English, handling Urdu-English mix code-switching naturally."
                    )
                whisper_transcript = whisper_response.strip()
                whisper_success = True
                # Optional: log for debugging
                # print(f"Used direct .mp4 upload ({file_size_mb:.1f} MB)")

            else:
                # Fallback for large videos: extract audio
                raise ValueError(f"Video > 25 MB ({file_size_mb:.1f} MB) — extracting audio")

        except Exception as direct_err:
            # Direct failed (size, format quirk, etc.) → try extraction
            try:
                # Extract audio
                subprocess_result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", video_path,
                        "-vn", "-acodec", "libmp3lame", "-q:a", "2", audio_path
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                with open(audio_path, "rb") as extracted_audio_file:
                    whisper_response = client.audio.translations.create(
                        model="whisper-1",
                        file=extracted_audio_file,
                        response_format="text",
                        prompt="Translate to natural English, handling Urdu-English mix code-switching naturally."
                    )
                whisper_transcript = whisper_response.strip()
                whisper_success = True
                # print("Used extracted .mp3 fallback")

            except subprocess.CalledProcessError as ffmpeg_err:
                whisper_transcript = f"FFmpeg failed (code {ffmpeg_err.returncode}): {ffmpeg_err.stderr.strip() or str(ffmpeg_err)}"
            except Exception as whisper_err:
                whisper_transcript = f"Whisper failed: {str(whisper_err)}"

        finally:
            # Always clean up temp file if created
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass

            # Save Whisper to TXT if success
            audio_txt_path = os.path.join(TXT_DIR, f"{video_id}_audio.txt")
            with open(audio_txt_path, "w", encoding="utf-8") as f:
                f.write(whisper_transcript)

        # Step 3: GPT-5.2 analysis (only if both OK)
        gpt_analysis = "GPT analysis skipped"
        try:
            # Read the saved transcript (you already wrote it above)
            audio_txt_path = os.path.join(TXT_DIR, f"{video_id}_audio.txt")
            
            if not os.path.exists(audio_txt_path):
                raise FileNotFoundError("Audio transcript file not found")
            
            with open(audio_txt_path, "r", encoding="utf-8") as f:
                whisper_transcript = f.read().strip()
            
            # If you re-enable Florence later, load it here too:
            # visual_txt_path = os.path.join(TXT_DIR, f"{video_id}_visual.txt")
            # with open(visual_txt_path, "r", encoding="utf-8") as f:
            #     visual_descriptions = f.read().strip()
            # Then add to prompt: f"Visual Description from Florence-2:\n{visual_descriptions}\n\n"

            combined_prompt = (
                f"Audio Transcript from Whisper:\n{whisper_transcript}\n\n"
                "You are an expert video analyst. Provide:\n"
                "1. A detailed, coherent English summary of the video content (combine audio narration/dialogue with likely visuals).\n"
                "2. Key topics, main ideas, and any important points discussed.\n"
                "3. A catchy, SEO-friendly YouTube title suggestion (max 70 characters).\n"
                "4. Optional: 3-5 relevant YouTube tags.\n"
                "Be concise yet comprehensive. If transcript is empty/short, note that and give a general inference."
            )

            # Use the shared client (or create if needed)
            # client = OpenAI(api_key=...)  # ← already have this earlier

            gpt_response = client.chat.completions.create(
                model="gpt-5.2",                    # ← correct: this is the "Thinking" version
                # model="gpt-5.2-pro",              # ← use this for even better quality (more expensive)
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional video content analyst specializing in educational, vlog, lecture, or mixed-language (Urdu-English) videos. Always produce structured, accurate, natural-English outputs."
                    },
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.4,
                max_completion_tokens=1200                   # Enough for detailed summary
            )

            gpt_analysis = gpt_response.choices[0].message.content.strip()

        except Exception as e:
            gpt_analysis = f"GPT analysis failed: {str(e)}"
            print("GPT error details:", str(e)) 
            return jsonify({
                "final_analysis": gpt_analysis,
            }), 500
                # ← for your server logs

                # Step 4: Gemini Flash refinement (GPT → Gemini)
        gemini_analysis = "Gemini refinement skipped"
        custom_head_style = """
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <script src="https://cdn.tailwindcss.com"></script>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
                    <style>
                        /* Custom font import and core styles */
                        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
                        
                        body {
                            font-family: 'Inter', sans-serif;
                            background-color: #000000; /* Pure Black for Console Feel */
                            color: #e2e8f0; /* Default text color for dark mode */
                        }
                        .mono {
                            font-family: 'JetBrains Mono', monospace;
                        }
                        /* Custom diagnostic panel styling */
                        .diagnostic-panel {
                            background-color: #0f172a; /* Slate-900 */
                            border: 1px solid #1e293b;
                            box-shadow: 0 0 15px rgba(252, 165, 165, 0.05); /* Subtle pink glow */
                            transition: all 0.3s ease-in-out;
                            margin-bottom: 1.5rem; /* Spacing between panels */
                            border-radius: 0.5rem; /* Rounded corners */
                            padding: 1.5rem;
                        }
                        .diagnostic-panel:hover {
                            border-color: #ef4444; /* Red-500 on hover */
                            transform: translateY(-2px);
                            box-shadow: 0 5px 20px rgba(239, 68, 68, 0.15);
                        }
                        /* Keyframe for "Ungrounded" pulse or emphasis */
                        @keyframes pulse-red {
                            0%, 100% { color: #f87171; text-shadow: 0 0 5px #ef4444; }
                            50% { color: #fee2e2; text-shadow: 0 0 10px #f87171; }
                        }
                        .pulse-text {
                            animation: pulse-red 2s infinite;
                        }
                    </style>
                </head>
                """        

        try:
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")

            gemini_prompt = (

                "Design a comprehensive, vertically scrolling *Diagnostic Report* from the analysis text below, "
                "where each numbered section is its own distinct, visually segmented panel.\n\n"

                "*MANDATORY STYLING & HEAD SECTION:*\n"
                "You MUST start the HTML file with the exact <head> block provided below. "
                "Do not create your own styles; use the classes defined in this block (like 'diagnostic-panel', 'mono', 'pulse-text') "
                "to structure the body content.\n"
                f"```html\n{custom_head_style}\n```\n\n"

                "*File Generation & Technology (Mandatory):*\n"
                "1. Generate the output as a single, fully self-contained, mobile-responsive HTML file.\n"
                "2. Mandatory use of Tailwind CSS and appropriate Font Awesome icons to illustrate concepts.\n"
                "3. The final output must be optimized for vertical scrolling on mobile devices (mobile-first layout).\n\n"

                "*Structural Constraints (Mandatory):*\n"
                "1. The report must be divided into distinct, easily scrollable panels, with each panel corresponding "
                "exactly to one of the numbered sections in the analysis text.\n"
                "2. *MANDATORY:* Include every single word of the provided analysis text verbatim within the generated panels, "
                "retaining all original section numbers and titles.\n\n"

                "*Aesthetic Directive (Crucial Adaptation):*\n"
                "1. The visual design (color palette, typography, textures, shadows, and overall mood) must be directly inspired "
                "by and aesthetically reflect the theme, tone, and emotional content of the analysis text provided. "
                "Do not default to a generic color scheme unless the text implies it.\n\n"
                f"{gpt_analysis}\n"
                "Return ONLY the final HTML file output. Do not include explanations or markdown."
            )


            gemini_response = gemini_model.generate_content(
                gemini_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 8192,
                }
            )

            gemini_analysis = gemini_response.text.strip()

        except Exception as e:
            gemini_analysis = f"Gemini refinement failed: {str(e)}"
            print("Gemini error:", str(e))
            return jsonify({
                "final_analysis": gemini_analysis,
            }), 500
        if (
            gemini_analysis
            and "failed" not in gemini_analysis.lower()
            and "skipped" not in gemini_analysis.lower()
        ):
            cleanup_video_files(
                video_id=video_id,
                video_path=video_path,
                frames_dir=FRAMES_DIR,
                txt_dir=TXT_DIR
            )
        return jsonify({
                 # optional (for debugging/logs)
            "final_analysis": gemini_analysis,     # ✅ FINAL polished result
        }), 200

    return app
