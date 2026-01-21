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
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
TXT_DIR = "txt_outputs"
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
        whisper_transcript = "No audio or transcription available"
        whisper_success = True

        audio_path = os.path.join(UPLOAD_DIR, f"{video_id}_audio.mp3")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libmp3lame", "-q:a", "2", audio_path
            ], check=True, capture_output=True, timeout=600)

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            with open(audio_path, "rb") as audio_file:
                whisper_response = client.audio.translations.create(  # translations for English output
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                    response_format="text",
                    prompt="Translate to natural English, handling Urdu-English mix."
                )
            whisper_transcript = whisper_response.strip()

        except Exception as e:
            whisper_transcript = f"Whisper failed: {str(e)}"
            whisper_success = False

        # Clean up audio
        try:
            os.remove(audio_path)
        except:
            pass

        if not whisper_success:
            return jsonify({
                "video_id": video_id,
                "frames_saved": saved,
                "florence_results": florence_results,
                "whisper_transcript": whisper_transcript,
                "status": "Whisper failed – skipping GPT-5.2"
            }), 200

        # Save Whisper to TXT if success
        audio_txt_path = os.path.join(TXT_DIR, f"{video_id}_audio.txt")
        with open(audio_txt_path, "w", encoding="utf-8") as f:
            f.write(whisper_transcript)

        # Step 3: GPT-5.2 analysis (only if both OK)
        gpt_analysis = "GPT-5.2 analysis skipped"
        try:
            combined_prompt = (
                f"Visual Description from Florence-2:\n{open(visual_txt_path, 'r').read()}\n\n"
                f"Audio Transcript from Whisper:\n{whisper_transcript}\n\n"
                "Based on the above video visuals and audio, provide a detailed English summary, key topics, and suggested YouTube title."
            )

            gpt_response = client.chat.completions.create(
                model="gpt-5.2-thinking",  # or "gpt-5.2-pro" for deeper
                messages=[
                    {"role": "system", "content": "You are a video analyst. Combine visuals and audio for coherent insights."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )

            gpt_analysis = gpt_response.choices[0].message.content.strip()

        except Exception as e:
            gpt_analysis = f"GPT-5.2 failed: {str(e)}"

        # Optional: clean TXT files after GPT
        # os.remove(visual_txt_path)
        # os.remove(audio_txt_path)

        return jsonify({
            "video_id": video_id,
            "frames_saved": saved,
            "florence_results": florence_results,
            "whisper_transcript": whisper_transcript,
            "gpt_52_analysis": gpt_analysis,
            "status": "Full pipeline complete"
        }), 200
    return app
