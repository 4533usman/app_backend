# app/utils/cleanup.py

import os
import shutil
import glob

def cleanup_video_files(video_id, video_path, frames_dir, txt_dir):
    """
    Safely deletes all temporary files created during video processing.
    """

    try:
        # 1. Delete uploaded video
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

        # 2. Delete extracted frames directory
        frames_folder = os.path.join(frames_dir, video_id)
        if os.path.exists(frames_folder):
            shutil.rmtree(frames_folder)

        # 3. Delete TXT outputs
        txt_files = glob.glob(os.path.join(txt_dir, f"{video_id}_*.txt"))
        for txt in txt_files:
            os.remove(txt)

        print(f"[CLEANUP] Completed for video_id={video_id}")

    except Exception as e:
        # Never crash the request because of cleanup
        print(f"[CLEANUP WARNING] {str(e)}")
