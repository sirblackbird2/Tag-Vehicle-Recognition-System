import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from app.config import YOLO_MODEL, DETECTION_CONFIDENCE

class VehicleRecognitionService:
    def __init__(self):
        print("🚀 Loading YOLO vehicle model...")
        self.yolo = YOLO(YOLO_MODEL)
        
        print("📖 Loading EasyOCR...")
        self.ocr = easyocr.Reader(['en'], gpu=False)
        
        self.vehicle_classes = {
            2: 'Car',
            3: 'Motorcycle',
            5: 'Bus',
            7: 'Truck',
            1: 'Bicycle'
        }
        print("✅ Models loaded!")
    
    def process_image(self, image_path):
        """Process image and return detected vehicles with plates"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image"}
        
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
                    
                    all_vehicles.append({
                        'type': self.vehicle_classes[cls_id],
                        'confidence': round(confidence, 3),
                        'bbox': [x1, y1, x2, y2],
                        'plate': plate_text
                    })
        
        # Remove overlapping detections (keep the highest confidence one)
        merged = []
        
        for v1 in all_vehicles:
            is_duplicate = False
            for v2 in merged:
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
                            # Replace the existing one
                            merged[merged.index(v2)] = v1
                        break
            
            if not is_duplicate:
                merged.append(v1)
        
        return {
            'vehicles': merged,
            'total': len(merged)
        }
    
    def _read_plate(self, roi):
        """Read license plate by looking at bottom half of vehicle"""
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
            
            # Collect all text pieces
            all_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    clean = text.replace(" ", "").upper()
                    clean = ''.join(c for c in clean if c.isalnum())
                    if clean:
                        all_text.append(clean)
            
            if all_text:
                combined = ''.join(all_text)
                combined = ''.join(c for c in combined if c.isalnum())
                
                # If it's longer than 8 chars, it might include the date
                if len(combined) > 8:
                    plate_part = combined[:8]
                    if len(plate_part) >= 4:
                        return plate_part
                
                if len(combined) >= 4:
                    return combined
                    
        except Exception as e:
            print(f"OCR Error: {e}")
            pass

        return None