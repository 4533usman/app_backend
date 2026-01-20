from flask import Flask, request, jsonify
import os
import uuid
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# =====================
# RUNPOD CONFIG
# =====================
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
FRAME_INTERVAL_SECONDS = 2

# =====================
# LOAD FLORENCE ONCE (GPU)
# =====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32

# ─── Most reliable loading pattern right now ───
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
# =====================
# FLORENCE CAPTION (SAFE)
# =====================
def florence_caption(image_path, task="<MORE_DETAILED_CAPTION>"):
    """
    Caption a single image with Florence-2.
    Returns clean parsed description (recommended).
    """
    if not os.path.exists(image_path) or os.path.getsize(image_path) < 1024:
        raise ValueError(f"Invalid or empty image: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open image {image_path}: {str(e)}")

    # Processor expects list for consistency (even for batch=1)
    inputs = processor(
        text=task,
        images=[image],           # ← good as-is
        return_tensors="pt"
    )

    # Move to GPU + float16
    inputs["pixel_values"] = inputs["pixel_values"].to(device=DEVICE, dtype=torch.float16)
    inputs["input_ids"]   = inputs["input_ids"].to(device=DEVICE)   
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,       # ↑ enough for detailed caption
            num_beams=3,
            do_sample=False
        )

    # Raw decode (with special tokens) → needed for post-processing
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False   # ← keep <loc> etc for parsing
    )[0]

    # IMPORTANT: Clean parsed output (bboxes, labels, captions, etc.)
    try:
        parsed = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )
        # For caption tasks → usually just a clean string
        if task in ["<CAPTION>", "<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>"]:
            return parsed[task] if task in parsed else generated_text.strip()
        else:
            return str(parsed)  # dict for <OD>, <OCR_WITH_REGION>, etc.
    except Exception as parse_err:
        # Fallback if parsing fails (rare)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# =====================
# FLASK APP (RUNPOD)
# =====================
# =====================
# FLORENCE CAPTION DIRECT (for in-memory PIL Image - no file needed)
# =====================
def florence_caption_direct(pil_image, task="<MORE_DETAILED_CAPTION>"):
    if not isinstance(pil_image, Image.Image):
        raise ValueError("Input must be a PIL Image")

    prompt = task  # for detailed caption we usually don't need extra text

    # Processor
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt")

    if "pixel_values" not in inputs or inputs["pixel_values"] is None:
        raise ValueError("Processor failed to generate pixel_values")

    # Move to device + dtype
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if DEVICE.startswith("cuda"):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    # Generate – use greedy like your local code
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=1,           # ← change to 1 (greedy) – more stable parsing
            do_sample=False,
            use_cache=False        # ← helps in some cloud setups
        )

    # Decode with special tokens (important for post_process)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    try:
        parsed = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil_image.width, pil_image.height)
        )
        if isinstance(parsed, dict) and task in parsed:
            return parsed[task]
        return str(parsed)
    except Exception as ex:
        print(f"Post-process failed: {ex}")               # ← log this!
        # Fallback – often still usable
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
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

        video_path = os.path.join(
            UPLOAD_DIR, f"{video_id}_{video.filename}"
        )
        video.save(video_path)

        frames_folder = os.path.join(FRAMES_DIR, video_id)
        os.makedirs(frames_folder, exist_ok=True)

        # 🔥 Force FFmpeg backend (CRITICAL for RunPod)
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

            # 🔒 Validate frame before saving
            if (
                fps > 0 and
                frame_index % int(fps * FRAME_INTERVAL_SECONDS) == 0 and
                frame is not None and
                frame.size > 0
            ):
                frame_path = os.path.join(
                    frames_folder, f"frame_{saved:05d}.jpg"
                )

                success = cv2.imwrite(frame_path, frame)
                if success:
                    saved += 1

            frame_index += 1

        cap.release()

        # =====================
        # FLORENCE INFERENCE
        # =====================
        results = []

        for frame_file in sorted(os.listdir(frames_folder)):
            frame_path = os.path.join(frames_folder, frame_file)

            try:
                caption = florence_caption(frame_path)
                results.append({
                    "frame": frame_file,
                    "caption": caption
                })
            except Exception as e:
                results.append({
                    "frame": frame_file,
                    "error": str(e)
                })

        return jsonify({
            "video_id": video_id,
            "frames_saved": saved,
            "results": results
        }), 200

    # ---------------------
    # Image Upload Endpoint
    # ---------------------
    # ---------------------
# Image Upload Endpoint (Now Hardcoded Image - No Upload Required)
# ---------------------
    @app.route("/upload-image", methods=["POST"])
    def upload_image():
        # These URLs ALWAYS serve valid JPEG (tested, no hotlink block)
        HARDCODED_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"

        # Alternatives if you want variety:
        # HARDCODED_IMAGE_URL = "https://picsum.photos/800/600"               # Random high-quality photo
        # HARDCODED_IMAGE_URL = "https://images.unsplash.com/photo-1557683316-973673baf926?w=800"  # Unsplash example

        image_id = str(uuid.uuid4())

        try:
            import requests
            from io import BytesIO

            # Download
            response = requests.get(HARDCODED_IMAGE_URL, timeout=15)
            response.raise_for_status()  # Fail fast on 4xx/5xx

            content = response.content
            if len(content) < 10000:  # Rough check: real JPGs are bigger
                raise ValueError("Downloaded content too small – likely not a real image")

            image_bytes = BytesIO(content)

            # Load + strict validation
            image = Image.open(image_bytes)
            try:
                image.verify()  # This WILL raise if invalid/corrupt
            except Exception as ve:
                raise ValueError(f"Image verification failed: {str(ve)}")

            # Re-open safely after verify
            image_bytes.seek(0)
            image = Image.open(image_bytes).convert("RGB")

            # Extra safety
            if image.size[0] < 1 or image.size[1] < 1:
                raise ValueError("Image has invalid (zero) dimensions")

            # Now run Florence – processor gets a real image
            caption = florence_caption_direct(image)

            return jsonify({
                "image_id": image_id,
                "description": caption,
                "source": "hardcoded_url",
                "url": HARDCODED_IMAGE_URL
            }), 200

        except requests.exceptions.RequestException as req_err:
            return jsonify({
                "image_id": image_id,
                "error": f"Download failed: {str(req_err)}",
                "detail": "Network or URL issue"
            }), 500

        except ValueError as val_err:
            return jsonify({
                "image_id": image_id,
                "error": str(val_err),
                "detail": "Invalid image data (corrupt, wrong format, or blocked)"
            }), 400

        except Exception as e:
            return jsonify({
                "image_id": image_id,
                "error": str(e),
                "detail": "Unexpected processing error"
            }), 500
    return app
