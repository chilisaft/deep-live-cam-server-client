import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")
file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

mode = "client"
frame_processors: List[str] = []
nsfw_filter = False
video_encoder = None
video_quality = None
live_mirror = False
live_resizable = True
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = "error"
fp_ui: Dict[str, bool] = {}
camera_input_combobox = None
webcam_preview_running = False
show_fps = False
mouth_mask = False
show_mouth_mask_box = False
