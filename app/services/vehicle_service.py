import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from app.config import YOLO_MODEL, DETECTION_CONFIDENCE

class VehicleRecognitionService:
    def __init__(self):
        print("Loading YOLO vehicle model...")
        self.yolo = YOLO(YOLO_MODEL)
        
        print("Loading EasyOCR...")
        self.ocr = easyocr.Reader(['en'], gpu=False)
        
        # Load brand classifier
        self.brand_model = None
        self.brand_classes = []
        self.brand_transform = None
        self._load_brand_classifier()
        
        self.vehicle_classes = {
            2: 'Car',
            3: 'Motorcycle',
            5: 'Bus',
            7: 'Truck',
            1: 'Bicycle'
        }
        print("Models loaded!")
    
    def _load_brand_classifier(self):
        try:
            checkpoint = torch.load('brand_classifier.pth', map_location=torch.device('cpu'))
            self.brand_classes = checkpoint['classes']
            self.brand_model = models.resnet18(weights=None)
            self.brand_model.fc = torch.nn.Linear(self.brand_model.fc.in_features, len(self.brand_classes))
            self.brand_model.load_state_dict(checkpoint['model_state_dict'])
            self.brand_model.eval()
            
            self.brand_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            print(f"Brand classifier loaded: {len(self.brand_classes)} classes")
        except Exception as e:
            print(f"Brand classifier not loaded: {e}")
            self.brand_model = None
    
    def _classify_brand(self, roi):
        print("DEBUG: _classify_brand called")  # Add this
        if self.brand_model is None or roi is None or roi.size == 0:
            print("DEBUG: Brand model or ROI is None")  # Add this
            return None
        
        try:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = self.brand_transform(roi_rgb).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.brand_model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            if confidence.item() > 0.4:
                return self.brand_classes[predicted.item()]
            return "Unknown"
        except Exception as e:
            return None
    
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
                    print("DEBUG: Before brand classification")
                    brand = self._classify_brand(roi)
                    print(f"DEBUG: Brand returned: {brand}")
                    
                    all_vehicles.append({
                        'type': self.vehicle_classes[cls_id],
                        'brand': brand,
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
    
    def _ocr_plate(self, plate_roi):
        """Run EasyOCR on the cropped plate region"""
        # Resize for better reading
        plate_roi = cv2.resize(plate_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
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
            print(f"Fallback OCR Error: {e}")
            pass

        return None