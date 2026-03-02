from flask import Flask, request, jsonify
import os
import math
import shutil
import threading
import subprocess
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from app.helpers.supabase_client import update_job, insert_batch
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
SEGMENT_DURATION = 60        # seconds per batch
FRAME_INTERVAL_SECONDS = 2   # 1 frame every N seconds

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


# -------------------------------------------------------
# Florence helper
# -------------------------------------------------------
def florence_caption(frame_path: str, task: str = "<MORE_DETAILED_CAPTION>") -> str:
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    try:
        pil_image = Image.open(frame_path).convert("RGB")
        inputs = processor(text=task, images=pil_image, return_tensors="pt")

        if "pixel_values" not in inputs or inputs["pixel_values"] is None:
            raise ValueError(f"Processor failed on {frame_path}")

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
                generated_text, task=task,
                image_size=(pil_image.width, pil_image.height)
            )
            if isinstance(parsed, dict) and task in parsed:
                return parsed[task].strip()
            if isinstance(parsed, str):
                return parsed.strip()
            return str(parsed).strip()
        except Exception:
            raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return raw if raw else "No description generated"

    except Exception as e:
        print(f"Error captioning {frame_path}: {e}")
        return f"Error: {str(e)}"


# -------------------------------------------------------
# Background worker — Florence only, audio upload per segment
# -------------------------------------------------------
def process_video_background(app, job_id: str, video_path: str, duration_seconds: float):
    from app.helpers.supabase_client import upload_audio_segment

    total_segments = math.ceil(duration_seconds / SEGMENT_DURATION)

    with app.app_context():
        try:
            for seg_idx in range(total_segments):
                start_sec = seg_idx * SEGMENT_DURATION
                seg_label = f"seg_{seg_idx + 1:03d}"
                print(f"[{job_id}] ▶ {seg_label}  ({start_sec}s – {start_sec + SEGMENT_DURATION}s)")

                # ── Florence: extract frames → captions ─────────────────
                seg_frames_dir = os.path.join(FRAMES_DIR, job_id, seg_label)
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
                    print(f"[{job_id}] ffmpeg frames failed for {seg_label}: {e.stderr}")

                seg_captions = []
                for frame_file in sorted(os.listdir(seg_frames_dir)):
                    caption = florence_caption(os.path.join(seg_frames_dir, frame_file))
                    seg_captions.append(f"{frame_file}: {caption}")

                florence_text = "\n".join(seg_captions)

                # Clear GPU after each segment
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()

                # Remove frames immediately — no longer needed
                shutil.rmtree(seg_frames_dir, ignore_errors=True)

                # ── Audio: extract → upload to Supabase Storage ─────────
                seg_audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_audio_{seg_label}.mp3")
                audio_storage_path = None

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
                    # Upload to Supabase Storage bucket: audio-segments
                    audio_storage_path = upload_audio_segment(job_id, seg_label, seg_audio_path)
                    print(f"[{job_id}] ✓ Audio uploaded: {audio_storage_path}")

                except Exception as e:
                    print(f"[{job_id}] Audio upload failed for {seg_label}: {e}")

                finally:
                    # Always delete local audio file
                    if os.path.exists(seg_audio_path):
                        os.remove(seg_audio_path)

                # ── Save batch to Supabase ───────────────────────────────
                insert_batch(job_id, seg_idx, florence_text, audio_storage_path)
                print(f"[{job_id}] ✓ {seg_label} saved to Supabase")

            # All segments done — trigger edge function automatically
            # update_job(job_id, status="completed")
            print(f"[{job_id}] ✓ All {total_segments} segments complete — triggering edge function")

            edge_url = os.getenv("SUPABASE_EDGE_FUNCTION_URL")  # e.g. https://<project>.supabase.co/functions/v1/finalize
            edge_key = os.getenv("SUPABASE_KEY")

            if edge_url:
                import requests as req_lib
                try:
                    edge_res = req_lib.post(
                        edge_url,
                        json={"job_id": job_id},
                        headers={
                            "Authorization": f"Bearer {edge_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=300  # Whisper + GPT + Gemini can take up to 5 min
                    )
                    print(f"[{job_id}] Edge function response: {edge_res.status_code}")
                except Exception as e:
                    print(f"[{job_id}] Edge function call failed: {e}")
            else:
                print(f"[{job_id}] SUPABASE_EDGE_FUNCTION_URL not set — skipping auto-trigger")

        except Exception as e:
            print(f"[{job_id}] Background processing failed: {e}")
            update_job(job_id, status="failed", error=str(e))

        finally:
            # Clean up video and any leftover frames
            if os.path.exists(video_path):
                os.remove(video_path)
            job_frames_dir = os.path.join(FRAMES_DIR, job_id)
            if os.path.exists(job_frames_dir):
                shutil.rmtree(job_frames_dir, ignore_errors=True)


# -------------------------------------------------------
# App factory
# -------------------------------------------------------
def create_app():
    app = Flask(__name__)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # ── Health ─────────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "RunPod API running"}), 200

    # ── Upload video + start Florence background processing ────────────
    @app.route("/upload-video", methods=["POST"])
    def upload_video():
        if "video" not in request.files:
            return jsonify({"error": "No video provided"}), 400

        # job_id created by frontend — receive it here
        job_id = request.form.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id is required"}), 400

        video = request.files["video"]
        video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{video.filename}")
        video.save(video_path)

        # Get duration
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

        # Launch Florence-only background thread
        thread = threading.Thread(
            target=process_video_background,
            args=(app, job_id, video_path, duration_seconds),
            daemon=True
        )
        thread.start()

        return jsonify({
            "job_id": job_id,
            "total_segments": total_segments,
            "status": "processing",
            "message": f"Florence processing started for {total_segments} segment(s). Call Supabase Edge Function /finalize when ready."
        }), 202

    return app
