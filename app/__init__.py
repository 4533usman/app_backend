from flask import Flask, request, jsonify
import os
import uuid
import math
import cv2
import torch
import subprocess
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from openai import OpenAI
import google.generativeai as genai
from app.helpers.utils import cleanup_video_files
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
TXT_DIR = "txt_outputs"
FRAME_INTERVAL_SECONDS = 2
SEGMENT_DURATION = 60  # 1 minute per batch

os.makedirs(TXT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    attn_implementation="eager",
).to(DEVICE)

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)


def florence_caption(frame_path: str, task: str = "<MORE_DETAILED_CAPTION>") -> str:
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    try:
        pil_image = Image.open(frame_path).convert("RGB")
        inputs = processor(text=task, images=pil_image, return_tensors="pt")

        if "pixel_values" not in inputs or inputs["pixel_values"] is None:
            raise ValueError(f"Processor failed to extract pixel_values from {frame_path}")

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        if DEVICE.startswith("cuda"):
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                use_cache=False
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        try:
            parsed = processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(pil_image.width, pil_image.height)
            )
            if isinstance(parsed, dict) and task in parsed:
                return parsed[task].strip()
            if isinstance(parsed, str):
                return parsed.strip()
            return str(parsed).strip()
        except Exception as post_err:
            print(f"Post-process failed for {frame_path}: {post_err}")
            raw_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return raw_caption if raw_caption else "No description generated"

    except Exception as e:
        print(f"Error captioning {frame_path}: {e}")
        return f"Error: {str(e)}"


def create_app():
    app = Flask(__name__)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "RunPod API running"}), 200

    @app.route("/upload-video", methods=["POST"])
    def upload_video():
        if "video" not in request.files:
            return jsonify({"error": "No video provided"}), 400

        video = request.files["video"]
        video_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{video.filename}")
        video.save(video_path)

        # --- Get video duration ---
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video"}), 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps <= 0:
            return jsonify({"error": "Cannot determine video FPS"}), 500

        duration_seconds = total_frame_count / fps
        total_segments = math.ceil(duration_seconds / SEGMENT_DURATION)

        print(f"[{video_id}] Duration: {duration_seconds:.1f}s → {total_segments} segment(s)")

        # --- Accumulators ---
        all_florence_captions = []
        all_whisper_transcripts = []
        client = OpenAI(api_key=key)

        # -----------------------------------------------
        # Step 1+2: Per-segment Florence + Whisper
        # -----------------------------------------------
        for seg_idx in range(total_segments):
            start_sec = seg_idx * SEGMENT_DURATION
            seg_label = f"seg_{seg_idx + 1:03d}"
            print(f"[{video_id}] Processing {seg_label} (start={start_sec}s)")

            # -- Florence: extract frames via ffmpeg for this segment --
            seg_frames_dir = os.path.join(FRAMES_DIR, video_id, seg_label)
            os.makedirs(seg_frames_dir, exist_ok=True)

            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(start_sec),
                        "-t", str(SEGMENT_DURATION),
                        "-i", video_path,
                        "-vf", f"fps=1/{FRAME_INTERVAL_SECONDS}",
                        "-q:v", "2",
                        os.path.join(seg_frames_dir, "frame_%05d.jpg")
                    ],
                    check=True, capture_output=True, timeout=120
                )
            except subprocess.CalledProcessError as e:
                print(f"[{video_id}] ffmpeg frame extract failed for {seg_label}: {e.stderr}")

            seg_captions = []
            for frame_file in sorted(os.listdir(seg_frames_dir)):
                frame_path = os.path.join(seg_frames_dir, frame_file)
                caption = florence_caption(frame_path)
                seg_captions.append(f"{seg_label}/{frame_file}: {caption}")

            # Save this segment's Florence output
            florence_txt = os.path.join(TXT_DIR, f"{video_id}_florence_{seg_label}.txt")
            with open(florence_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(seg_captions))

            all_florence_captions.extend(seg_captions)

            # Clear GPU memory after each segment
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()

            # -- Whisper: extract audio for this segment --
            seg_audio_path = os.path.join(UPLOAD_DIR, f"{video_id}_audio_{seg_label}.mp3")
            seg_transcript = ""

            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(start_sec),
                        "-t", str(SEGMENT_DURATION),
                        "-i", video_path,
                        "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                        seg_audio_path
                    ],
                    check=True, capture_output=True, timeout=120
                )

                with open(seg_audio_path, "rb") as af:
                    whisper_response = client.audio.translations.create(
                        model="whisper-1",
                        file=af,
                        response_format="text",
                        prompt="Translate to natural English, handling Urdu-English mix code-switching naturally."
                    )
                seg_transcript = whisper_response.strip()

            except Exception as e:
                seg_transcript = f"[{seg_label} transcription failed: {str(e)}]"
                print(f"[{video_id}] Whisper failed for {seg_label}: {e}")

            finally:
                if os.path.exists(seg_audio_path):
                    os.remove(seg_audio_path)

            # Save this segment's Whisper output
            whisper_txt = os.path.join(TXT_DIR, f"{video_id}_whisper_{seg_label}.txt")
            with open(whisper_txt, "w", encoding="utf-8") as f:
                f.write(seg_transcript)

            minutes = start_sec // 60
            seconds = start_sec % 60
            all_whisper_transcripts.append(f"[{minutes}:{seconds:02d}] {seg_transcript}")

        # Combine all segments into final TXT files
        full_visual = "\n".join(all_florence_captions)
        full_transcript = "\n".join(all_whisper_transcripts)

        with open(os.path.join(TXT_DIR, f"{video_id}_visual.txt"), "w", encoding="utf-8") as f:
            f.write(full_visual)

        with open(os.path.join(TXT_DIR, f"{video_id}_audio.txt"), "w", encoding="utf-8") as f:
            f.write(full_transcript)

        # -----------------------------------------------
        # Step 3: GPT analysis on combined results
        # -----------------------------------------------
        gpt_analysis = "GPT analysis skipped"
        try:
            combined_prompt = (
                f"Visual Description from Florence-2 (frame captions):\n{full_visual}\n\n"
                f"Audio Transcript from Whisper:\n{full_transcript}\n\n"
                "You are an expert video analyst. Provide:\n"
                "1. A detailed, coherent English summary of the video content.\n"
                "2. Key topics, main ideas, and any important points discussed.\n"
                "3. A catchy, SEO-friendly YouTube title suggestion (max 70 characters).\n"
                "4. Optional: 3-5 relevant YouTube tags.\n"
                "Be concise yet comprehensive. If transcript is empty/short, note that and give a general inference."
            )

            gpt_response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional video content analyst specializing in educational, vlog, lecture, or mixed-language (Urdu-English) videos. Always produce structured, accurate, natural-English outputs."
                    },
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.4,
                max_completion_tokens=1200
            )

            gpt_analysis = gpt_response.choices[0].message.content.strip()

        except Exception as e:
            gpt_analysis = f"GPT analysis failed: {str(e)}"
            print("GPT error:", str(e))
            return jsonify({"final_analysis": gpt_analysis}), 500

        # -----------------------------------------------
        # Step 4: Gemini HTML generation
        # -----------------------------------------------
        gemini_analysis = "Gemini refinement skipped"
        custom_head_style = """
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <script src="https://cdn.tailwindcss.com"></script>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
                    <style>
                        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
                        body {
                            font-family: 'Inter', sans-serif;
                            background-color: #000000;
                            color: #e2e8f0;
                        }
                        .mono { font-family: 'JetBrains Mono', monospace; }
                        .diagnostic-panel {
                            background-color: #0f172a;
                            border: 1px solid #1e293b;
                            box-shadow: 0 0 15px rgba(252, 165, 165, 0.05);
                            transition: all 0.3s ease-in-out;
                            margin-bottom: 1.5rem;
                            border-radius: 0.5rem;
                            padding: 1.5rem;
                        }
                        .diagnostic-panel:hover {
                            border-color: #ef4444;
                            transform: translateY(-2px);
                            box-shadow: 0 5px 20px rgba(239, 68, 68, 0.15);
                        }
                        @keyframes pulse-red {
                            0%, 100% { color: #f87171; text-shadow: 0 0 5px #ef4444; }
                            50% { color: #fee2e2; text-shadow: 0 0 10px #f87171; }
                        }
                        .pulse-text { animation: pulse-red 2s infinite; }
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
                "1. The visual design must be directly inspired by and aesthetically reflect the theme, tone, and emotional content "
                "of the analysis text provided.\n\n"
                f"{gpt_analysis}\n"
                "Return ONLY the final HTML file output. Do not include explanations or markdown."
            )

            gemini_response = gemini_model.generate_content(
                gemini_prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 8192}
            )

            gemini_analysis = gemini_response.text.strip()

        except Exception as e:
            gemini_analysis = f"Gemini refinement failed: {str(e)}"
            print("Gemini error:", str(e))
            return jsonify({"final_analysis": gemini_analysis}), 500

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
            "video_id": video_id,
            "segments_processed": total_segments,
            "whisper_transcript": full_transcript,
            "final_analysis": gemini_analysis,
        }), 200

    return app
