from flask import Flask, request, jsonify
import os
import uuid
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# =====================
# RUNPOD CONFIG
# =====================
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
FRAME_INTERVAL_SECONDS = 2

DEVICE = "cuda"  # RunPod = GPU only
MODEL_ID = "microsoft/Florence-2-large"

# =====================
# LOAD FLORENCE ONCE (GPU)
# =====================
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"  # REQUIRED
).to(DEVICE).eval()


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

    inputs = processor(
        text=task,
        images=[pil_image],
        return_tensors="pt"
    )

    # This is the correct & important part
    inputs["pixel_values"] = inputs["pixel_values"].to(device=DEVICE, dtype=torch.float16)
    inputs["input_ids"]   = inputs["input_ids"].to(device=DEVICE)          # ← keeps torch.long !!!

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=3,
            do_sample=False
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    try:
        parsed = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil_image.width, pil_image.height)
        )
        if task in ["<CAPTION>", "<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>"]:
            return parsed[task] if task in parsed else generated_text.strip()
        else:
            return str(parsed)
    except Exception:
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
        # Reliable, direct image URLs that ALWAYS work (no hotlink blocks in 2026)
        HARDCODED_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

        # Alternative good options (uncomment any one if you want to change):
        # HARDCODED_IMAGE_URL = "https://picsum.photos/800/600"               # Random nice photo
        # HARDCODED_IMAGE_URL = "https://images.unsplash.com/photo-1601581875039-e8c425d10e5e?w=800"

        image_id = str(uuid.uuid4())

        try:
            import requests
            from io import BytesIO

            # Download with timeout + basic headers
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(HARDCODED_IMAGE_URL, headers=headers, timeout=15)
            response.raise_for_status()

            # Load image bytes
            image_bytes = BytesIO(response.content)

            # Step 1: Open and verify (catches invalid/corrupt downloads)
            image = Image.open(image_bytes)
            image.verify()  # This raises if the image is invalid/corrupt

            # Step 2: Re-open safely after verification + convert to RGB
            image_bytes.seek(0)  # Reset stream position
            image = Image.open(image_bytes).convert("RGB")

            # Optional quick check
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError("Image has zero dimensions")

            # Run Florence inference directly on the valid PIL image
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
                "detail": "Could not download the image"
            }), 500

        except Exception as e:
            return jsonify({
                "image_id": image_id,
                "error": str(e),
                "detail": "Image validation or processing failed"
            }), 500
    return app
