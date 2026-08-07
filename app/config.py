import os
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Upload directory
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Model settings
YOLO_MODEL = "models/yolov8n.pt"
OCR_LANG = "en"
DETECTION_CONFIDENCE = 0.5