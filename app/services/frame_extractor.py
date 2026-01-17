import subprocess
import os

def extract_frames(video_path, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-vsync", "0",
        os.path.join(output_dir, "frame_%05d.jpg")
    ], check=True)
