from flask import Flask, request, jsonify
import os
import uuid
import math
import shutil
import threading
import subprocess
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from openai import OpenAI
import google.generativeai as genai
from app.helpers.supabase_client import create_job, update_job, insert_batch, get_batches, get_job
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
# Background worker — one segment at a time
# -------------------------------------------------------
def process_video_background(app, job_id: str, video_path: str, duration_seconds: float):
    total_segments = math.ceil(duration_seconds / SEGMENT_DURATION)
    client = OpenAI(api_key=key)

    with app.app_context():
        try:
            for seg_idx in range(total_segments):
                start_sec = seg_idx * SEGMENT_DURATION
                seg_label = f"seg_{seg_idx + 1:03d}"
                print(f"[{job_id}] ▶ {seg_label}  ({start_sec}s – {start_sec + SEGMENT_DURATION}s)")

                # ── Florence ────────────────────────────────────────────
                seg_frames_dir = os.path.join(FRAMES_DIR, job_id, seg_label)
                os.makedirs(seg_frames_dir, exist_ok=True)

                try:
                    # New ffmpeg request per segment
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

                # Clear GPU after every segment's Florence pass
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()

                # Remove frames immediately — no longer needed
                shutil.rmtree(seg_frames_dir, ignore_errors=True)

                # ── Whisper ─────────────────────────────────────────────
                seg_audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_audio_{seg_label}.mp3")
                whisper_text = ""

                try:
                    # New ffmpeg request per segment
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
                        resp = client.audio.translations.create(
                            model="whisper-1",
                            file=af,
                            response_format="text",
                            prompt="Translate to natural English, handling Urdu-English mix code-switching naturally."
                        )
                    whisper_text = resp.strip()

                except Exception as e:
                    whisper_text = f"[{seg_label} transcription failed: {str(e)}]"
                    print(f"[{job_id}] Whisper failed for {seg_label}: {e}")

                finally:
                    if os.path.exists(seg_audio_path):
                        os.remove(seg_audio_path)

                # ── Save batch to Supabase ───────────────────────────────
                insert_batch(job_id, seg_idx, florence_text, whisper_text)
                print(f"[{job_id}] ✓ {seg_label} saved to Supabase")

            # All segments done
            update_job(job_id, status="completed")
            print(f"[{job_id}] ✓ All {total_segments} segments complete — ready to finalize")

        except Exception as e:
            print(f"[{job_id}] Background processing failed: {e}")
            update_job(job_id, status="failed", error=str(e))

        finally:
            # Clean up uploaded video and any leftover frames
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

    # ── Endpoint 1: Upload + start background processing ───────────────
    @app.route("/upload-video", methods=["POST"])
    def upload_video():
        if "video" not in request.files:
            return jsonify({"error": "No video provided"}), 400

        video = request.files["video"]
        video_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{video.filename}")
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

        # Create job row in Supabase
        job_id = create_job()

        # Launch background thread — one segment at a time
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
            "message": f"Video split into {total_segments} segment(s) of {SEGMENT_DURATION}s each. Call POST /finalize/<job_id> when ready."
        }), 202

    # ── Endpoint 2: Finalize — GPT + Gemini after all batches saved ────
    @app.route("/finalize/<job_id>", methods=["POST"])
    def finalize(job_id):
        # Check job status first
        job = get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        if job["status"] == "processing":
            return jsonify({
                "job_id": job_id,
                "status": "processing",
                "message": "Batches are still being processed. Try again shortly."
            }), 202

        if job["status"] == "failed":
            return jsonify({
                "job_id": job_id,
                "status": "failed",
                "error": job.get("error", "Unknown error during processing")
            }), 500

        # Fetch all saved batches
        batches = get_batches(job_id)
        if not batches:
            return jsonify({"error": "No batch data found for this job"}), 404

        # Combine all segment results
        all_florence = []
        all_whisper = []
        for batch in batches:
            seg_num = batch["segment_index"] + 1
            minutes = batch["segment_index"]
            all_florence.append(f"[Segment {seg_num}]\n{batch['florence_captions']}")
            all_whisper.append(f"[{minutes}:00] {batch['whisper_transcript']}")

        full_visual = "\n\n".join(all_florence)
        full_transcript = "\n".join(all_whisper)

        client = OpenAI(api_key=key)

        # ── GPT ────────────────────────────────────────────────────────
        gpt_analysis = "GPT analysis skipped"
        try:
            combined_prompt = (
                f"Visual Description from Florence-2 (frame captions per segment):\n{full_visual}\n\n"
                f"Audio Transcript from Whisper (per segment):\n{full_transcript}\n\n"
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
            print("GPT error:", str(e))
            return jsonify({"job_id": job_id, "final_analysis": f"GPT failed: {str(e)}"}), 500

        # ── Gemini ─────────────────────────────────────────────────────
        custom_head_style = """
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <script src="https://cdn.tailwindcss.com"></script>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
                    <style>
                        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
                        body { font-family: 'Inter', sans-serif; background-color: #000000; color: #e2e8f0; }
                        .mono { font-family: 'JetBrains Mono', monospace; }
                        .diagnostic-panel {
                            background-color: #0f172a; border: 1px solid #1e293b;
                            box-shadow: 0 0 15px rgba(252, 165, 165, 0.05);
                            transition: all 0.3s ease-in-out; margin-bottom: 1.5rem;
                            border-radius: 0.5rem; padding: 1.5rem;
                        }
                        .diagnostic-panel:hover {
                            border-color: #ef4444; transform: translateY(-2px);
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

        gemini_analysis = "Gemini refinement skipped"
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
                "3. The final output must be optimized for vertical scrolling on mobile devices.\n\n"
                "*Structural Constraints (Mandatory):*\n"
                "1. Each panel corresponds exactly to one numbered section in the analysis text.\n"
                "2. *MANDATORY:* Include every single word of the provided analysis text verbatim.\n\n"
                "*Aesthetic Directive:*\n"
                "1. The visual design must reflect the theme and emotional content of the analysis.\n\n"
                f"{gpt_analysis}\n"
                "Return ONLY the final HTML file output. Do not include explanations or markdown."
            )

            gemini_response = gemini_model.generate_content(
                gemini_prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 8192}
            )
            gemini_analysis = gemini_response.text.strip()

        except Exception as e:
            print("Gemini error:", str(e))
            return jsonify({"job_id": job_id, "final_analysis": f"Gemini failed: {str(e)}"}), 500

        # Save final result to Supabase
        update_job(job_id, response_payload={"html": gemini_analysis})

        return jsonify({
            "job_id": job_id,
            "segments_processed": len(batches),
            "whisper_transcript": full_transcript,
            "final_analysis": gemini_analysis,
        }), 200

    return app
