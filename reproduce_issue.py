import cv2
import subprocess
import os
import numpy as np

video_path = "test_no_audio.mp4"
audio_path = "test_audio.mp3"

def create_dummy_video():
    # Create a dummy video
    height, width = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
    
    # Create a blank image
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    for _ in range(20):
        out.write(frame)
    out.release()
    print(f"Created {video_path}")

def run_ffmpeg():
    print("Running ffmpeg...")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-q:a", "2", audio_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True)
        print("Success")
    except subprocess.CalledProcessError as e:
        print(f"Failed with exit code: {e.returncode}")
        print(f"Stderr: {e.stderr.decode()}")
    finally:
        pass

if __name__ == "__main__":
    try:
        create_dummy_video()
        run_ffmpeg()
    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
