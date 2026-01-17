from flask import Flask, request, jsonify
import os
import uuid
import cv2

# =====================
# CONFIG
# =====================
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
FRAME_INTERVAL_SECONDS = 2  # 1 frame every 2 seconds

# =====================
# FLASK APP
# =====================
def create_app():
    app = Flask(__name__)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "API running"}), 200

    @app.route("/upload-video", methods=["POST"])
    def upload_video():
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video = request.files["video"]
        if video.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Save video
        video_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{video.filename}")
        video.save(video_path)

        # Folder for frames
        frames_folder = os.path.join(FRAMES_DIR, video_id)
        os.makedirs(frames_folder, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video"}), 500

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = int(total_frames / fps) if fps > 0 else 0

        frame_index = 0
        saved_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save 1 frame every FRAME_INTERVAL_SECONDS
            if fps > 0 and frame_index % (fps * FRAME_INTERVAL_SECONDS) == 0:
                frame_path = os.path.join(
                    frames_folder, f"frame_{saved_frames:05d}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                saved_frames += 1

            frame_index += 1

        cap.release()

        return jsonify({
            "message": "Video uploaded and frames extracted successfully",
            "video_id": video_id,
            "filename": video.filename,
            "video_path": video_path,
            "frames_folder": frames_folder,
            "fps": fps,
            "duration_seconds": duration_seconds,
            "frames_extracted": saved_frames,
            "extraction_interval_seconds": FRAME_INTERVAL_SECONDS
        }), 200

    return app