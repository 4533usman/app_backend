import os

class Config:
    BASE_DIR = os.getcwd()
    INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "data/input_videos")
    OUTPUT_FRAME_DIR = os.path.join(BASE_DIR, "data/output_frames")
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
