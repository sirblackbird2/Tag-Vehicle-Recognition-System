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
BRAND_MODEL_PATH = "brand_classifier.pth"
BRAND_CONFIDENCE_THRESHOLD = 0.5  # min softmax confidence to report a brand instead of "Unknown"
MOTORCYCLE_BRAND_MODEL_PATH = "motorcycle_brand_classifier.pth"
MOTORCYCLE_BRAND_CONFIDENCE_THRESHOLD = 0.5