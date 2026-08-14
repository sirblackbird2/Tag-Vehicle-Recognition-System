import base64
import re

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from app.config import (
    YOLO_MODEL, DETECTION_CONFIDENCE,
    BRAND_MODEL_PATH, BRAND_CONFIDENCE_THRESHOLD,
    MOTORCYCLE_BRAND_MODEL_PATH, MOTORCYCLE_BRAND_CONFIDENCE_THRESHOLD,
)

# Box/label colors per vehicle type (BGR, since we draw with OpenCV)
ANNOTATION_COLORS = {
    'Car': (113, 204, 46),
    'Motorcycle': (15, 196, 241),
    'Bus': (60, 76, 231),
    'Truck': (182, 89, 155),
    'Bicycle': (219, 152, 52),
}
DEFAULT_COLOR = (200, 200, 200)

# A plausible license plate: 4-8 alphanumeric characters, mixing at least one
# letter and one digit. This is intentionally permissive across formats
# (it doesn't encode any specific country's plate grammar) but is enough to
# reject obvious OCR noise like pure-digit date stickers or short garbage
# reads from grilles/badges.
PLATE_PATTERN = re.compile(r'^(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{4,8}$')


class VehicleRecognitionService:
    def __init__(self):
        print("Loading YOLO vehicle model...")
        self.yolo = YOLO(YOLO_MODEL)

        print("Loading EasyOCR...")
        self.ocr = easyocr.Reader(['en'], gpu=False)

        # Brand classifiers: one per vehicle-body type, loaded generically.
        # Each entry holds the model, its class list, threshold, and preprocessing transform.
        self.brand_classifiers = {}
        self._load_brand_classifier("car", BRAND_MODEL_PATH, BRAND_CONFIDENCE_THRESHOLD)
        self._load_brand_classifier("motorcycle", MOTORCYCLE_BRAND_MODEL_PATH, MOTORCYCLE_BRAND_CONFIDENCE_THRESHOLD)

        self.vehicle_classes = {
            2: 'Car',
            3: 'Motorcycle',
            5: 'Bus',
            7: 'Truck',
            1: 'Bicycle'
        }
        print("Models loaded!")

    def _load_brand_classifier(self, name, model_path, threshold):
        """Load a ResNet18 brand classifier and register it under `name`
        (e.g. 'car', 'motorcycle') so _classify_brand can dispatch by vehicle type."""
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            classes = checkpoint['classes']

            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            self.brand_classifiers[name] = {
                'model': model,
                'classes': classes,
                'transform': transform,
                'threshold': threshold,
            }
            print(f"{name.capitalize()} brand classifier loaded: {len(classes)} classes")
        except Exception as e:
            print(f"{name.capitalize()} brand classifier not loaded: {e}")

    def _classify_brand(self, name, roi):
        """Run the brand classifier registered under `name` on a cropped vehicle ROI.
        Returns the predicted brand, 'Unknown' if confidence is below threshold,
        or None if no classifier is loaded for this vehicle type or on error."""
        classifier = self.brand_classifiers.get(name)
        if classifier is None or roi is None or roi.size == 0:
            return None

        try:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = classifier['transform'](roi_rgb).unsqueeze(0)

            with torch.no_grad():
                outputs = classifier['model'](tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            if confidence.item() > classifier['threshold']:
                return classifier['classes'][predicted.item()]
            return "Unknown"
        except Exception as e:
            print(f"Brand classification error ({name}): {e}")
            return None

    def process_image(self, image_path, annotate=False):
        """Process image and return detected vehicles with plates.

        If annotate=True, the result also includes an 'annotated_image' key:
        a base64-encoded JPEG with bounding boxes, type/brand labels, and
        plate text drawn on it, ready to hand straight to a frontend <img>.
        """
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image"}

        vehicles = self._detect_vehicles(image)

        result = {
            'vehicles': vehicles,
            'total': len(vehicles),
        }

        if annotate:
            annotated_image = self.annotate(image, vehicles)
            result['annotated_image'] = self.encode_image_base64(annotated_image)

        return result

    def _detect_vehicles(self, image):
        """Run YOLO + plate OCR + brand classification, then de-duplicate
        overlapping boxes. Returns the final list of vehicle dicts."""
        results = self.yolo(image, conf=DETECTION_CONFIDENCE)

        # Collect all vehicle detections
        all_vehicles = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])

                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])

                    roi = image[y1:y2, x1:x2]
                    plate_text = self._read_plate(roi)
                    # Route to the classifier trained for this vehicle's body type
                    vehicle_type = self.vehicle_classes[cls_id]
                    classifier_name = "motorcycle" if vehicle_type == "Motorcycle" else "car"
                    brand = self._classify_brand(classifier_name, roi)

                    all_vehicles.append({
                        'type': vehicle_type,
                        'brand': brand,
                        'confidence': round(confidence, 3),
                        'bbox': [x1, y1, x2, y2],
                        'plate': plate_text
                    })

        # Remove overlapping detections (keep the highest confidence one)
        merged = []

        for v1 in all_vehicles:
            is_duplicate = False
            for idx, v2 in enumerate(merged):
                # Check if boxes overlap significantly
                b1 = v1['bbox']
                b2 = v2['bbox']

                # Calculate intersection area
                x_left = max(b1[0], b2[0])
                y_top = max(b1[1], b2[1])
                x_right = min(b1[2], b2[2])
                y_bottom = min(b1[3], b2[3])

                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

                    # If overlap > 70%, they're the same vehicle
                    overlap_ratio = intersection / min(area1, area2)
                    if overlap_ratio > 0.7:
                        is_duplicate = True
                        # Keep the one with higher confidence
                        if v1['confidence'] > v2['confidence']:
                            # Replace the existing one, by index (not value
                            # equality, which could match the wrong entry
                            # if two detections happen to be identical dicts)
                            merged[idx] = v1
                        break

            if not is_duplicate:
                merged.append(v1)

        return merged

    # ------------------------------------------------------------------
    # Annotation / visualization
    # ------------------------------------------------------------------

    def annotate(self, image, vehicles):
        """Draw bounding boxes and a label (type, brand, plate) for each
        detected vehicle onto a copy of the image. Returns the annotated
        image as a numpy array (BGR, same format as the input)."""
        annotated = image.copy()

        for v in vehicles:
            x1, y1, x2, y2 = v['bbox']
            color = ANNOTATION_COLORS.get(v['type'], DEFAULT_COLOR)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_parts = [v['type']]
            if v.get('brand') and v['brand'] != 'Unknown':
                label_parts.append(v['brand'])
            label = " - ".join(label_parts)
            if v.get('plate'):
                label += f" | {v['plate']}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Keep the label inside the frame even if the box is near the top edge
            label_bottom = max(y1, text_h + baseline + 8)
            label_top = label_bottom - text_h - baseline - 8

            cv2.rectangle(
                annotated,
                (x1, label_top),
                (x1 + text_w + 10, label_bottom),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 5, label_bottom - baseline - 4),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def encode_image_base64(image):
        """Encode a numpy BGR image as a base64 JPEG string."""
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return None
        return base64.b64encode(buffer).decode('utf-8')

    # ------------------------------------------------------------------
    # Plate reading
    # ------------------------------------------------------------------

    def _read_plate(self, roi):
        """Try contour-based plate cropping first, fallback to bottom-half"""
        if roi is None or roi.size == 0:
            return None

        # --- PRIMARY: Contour-based plate crop ---
        plate_roi = self._crop_plate_contour(roi)
        if plate_roi is not None:
            text = self._ocr_plate(plate_roi)
            if text:
                return text

        # --- BACKUP: Bottom-half method ---
        return self._read_plate_fallback(roi)

    def _crop_plate_contour(self, roi):
        """Find and crop the license plate using contours (with size filtering)"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Noise reduction
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

        # 2. Edge detection
        edged = cv2.Canny(bfilter, 30, 200)

        # 3. Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        # Get image dimensions for size filtering
        h, w = roi.shape[:2]
        min_area = (w * h) * 0.01  # At least 1% of the vehicle crop
        max_area = (w * h) * 0.5   # At most 50% of the vehicle crop

        for contour in contours:
            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            # If it's a rectangle (4 corners)
            if len(approx) == 4:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                area = w_box * h_box

                # --- SIZE FILTERING ---
                # Ignore rectangles that are too small (date stickers) or too large
                if area < min_area or area > max_area:
                    continue

                # Also check aspect ratio (plates are wider than they are tall)
                aspect_ratio = max(w_box, h_box) / min(w_box, h_box)
                if aspect_ratio < 1.5:  # Too square (like a date sticker)
                    continue

                # Add padding
                pad = 10
                x = max(0, x - pad)
                y = max(0, y - pad)
                w_box = min(roi.shape[1] - x, w_box + pad*2)
                h_box = min(roi.shape[0] - y, h_box + pad*2)

                plate_roi = roi[y:y+h_box, x:x+w_box]
                if plate_roi.size > 0:
                    return plate_roi

        return None

    def _clean_ocr_results(self, results):
        """Shared cleanup: filter by confidence, strip non-alphanumerics,
        uppercase, and concatenate into one candidate string."""
        all_text = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                clean = text.replace(" ", "").upper()
                clean = ''.join(c for c in clean if c.isalnum())
                if clean:
                    all_text.append(clean)

        if not all_text:
            return None

        combined = ''.join(all_text)
        combined = ''.join(c for c in combined if c.isalnum())

        # If it's longer than 8 chars, it might include the date — try the
        # leading 8 characters first (most plate formats front-load the
        # plate itself; the date/sticker text tends to trail).
        candidates = []
        if len(combined) > 8:
            candidates.append(combined[:8])
        candidates.append(combined[:8] if len(combined) > 8 else combined)

        for candidate in candidates:
            if self._looks_like_plate(candidate):
                return candidate

        # Nothing passed the plate-format check — don't return OCR noise.
        return None

    @staticmethod
    def _looks_like_plate(text):
        """Reject OCR text that's unlikely to be a real plate. Requires
        4-8 alphanumeric characters with at least one letter AND one digit,
        which filters out pure-digit date stickers and short garbage reads
        from grilles/badges/decals. This is a heuristic, not a real
        plate-format validator — adjust PLATE_PATTERN if your target
        region's plates don't fit this shape (e.g. all-digit plates)."""
        if not text:
            return False
        return bool(PLATE_PATTERN.match(text))

    def _ocr_plate(self, plate_roi):
        """Run EasyOCR on the cropped plate region"""
        # Resize for better reading
        plate_roi = cv2.resize(plate_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

        try:
            results = self.ocr.readtext(gray)
            return self._clean_ocr_results(results)
        except Exception as e:
            print(f"OCR Error: {e}")
            return None

    def _read_plate_fallback(self, roi):
        """Original bottom-half method (your proven backup)"""
        if roi is None or roi.size == 0:
            return None

        # Focus on bottom half
        h, w, _ = roi.shape
        bottom_half = roi[int(h*0.5):h, 0:w]

        # Resize for better reading
        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        try:
            results = self.ocr.readtext(gray)
            return self._clean_ocr_results(results)
        except Exception as e:
            print(f"Fallback OCR Error: {e}")
            return None