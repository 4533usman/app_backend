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
def florence_caption(image_path, task="<CAPTION>"):
    # 🔒 Validate image file
    if not os.path.exists(image_path) or os.path.getsize(image_path) < 1024:
        raise ValueError("Invalid or empty image file")

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=task,
        images=image,
        return_tensors="pt"
    )

    inputs["pixel_values"] = inputs["pixel_values"].to(
        device=DEVICE,
        dtype=torch.float16
    )
    inputs["input_ids"] = inputs["input_ids"].to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]


# =====================
# FLASK APP (RUNPOD)
# =====================
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
    @app.route("/upload-image", methods=["POST"])
    def upload_image():
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        image_id = str(uuid.uuid4())
        image_path = os.path.join(
            UPLOAD_DIR, f"{image_id}_{image_file.filename}"
        )
        image_file.save(image_path)

        try:
            description = florence_caption(image_path)
            return jsonify({
                "image_id": image_id,
                "description": description
            }), 200
        except Exception as e:
            return jsonify({
                "image_id": image_id,
                "error": str(e)
            }), 500

    return app
