import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os

class VehicleDetector:
    def __init__(self):
        print("🚀 Loading YOLO model...")
        self.yolo = YOLO('models/yolov8n.pt')
        
        print("📖 Loading OCR model...")
        # Simplest initialization - no parameters
        self.ocr = PaddleOCR()
        
        # Vehicle class IDs from COCO dataset
        self.vehicle_classes = {
            2: 'Car',
            3: 'Motorcycle', 
            5: 'Bus',
            7: 'Truck',
            1: 'Bicycle'
        }
        print("✅ Models loaded successfully!\n")
    
    def detect_vehicles(self, image_path):
        """Detect vehicles and read license plates from an image"""
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image"}
        
        # Run YOLO detection
        results = self.yolo(image, conf=0.5)
        vehicles = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                
                # Check if it's a vehicle
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    
                    # Crop the vehicle region
                    vehicle_roi = image[y1:y2, x1:x2]
                    
                    # Try to read license plate
                    plate_text = self._read_plate(vehicle_roi)
                    
                    vehicles.append({
                        'type': self.vehicle_classes[cls_id],
                        'confidence': round(confidence, 3),
                        'bbox': [x1, y1, x2, y2],
                        'plate': plate_text
                    })
        
        return {
            'vehicles': vehicles,
            'total': len(vehicles)
        }
    
    def _read_plate(self, roi):
        """Extract text from license plate region"""
        if roi.size == 0:
            return None
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        try:
            result = self.ocr.ocr(gray)
            if result and result[0]:
                # Handle the return format
                plate_text = result[0][0][1][0]
                return plate_text.strip()
        except:
            pass
        
        return None

# ============================================
# TEST THE CODE
# ============================================

if __name__ == "__main__":
    # Create detector
    detector = VehicleDetector()
    
    # Ask user for image path
    image_path = input("📸 Enter path to your vehicle image: ")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print("❌ File not found! Please check the path.")
    else:
        # Run detection
        print("\n🔍 Processing image...")
        result = detector.detect_vehicles(image_path)
        
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print("\n" + "="*50)
            print("VEHICLE RECOGNITION RESULTS")
            print("="*50)
            
            if result['total'] == 0:
                print("❌ No vehicles detected in the image.")
            else:
                for i, v in enumerate(result['vehicles'], 1):
                    print(f"\n🚗 Vehicle {i}:")
                    print(f"   Type: {v['type']}")
                    print(f"   Confidence: {v['confidence']}")
                    print(f"   License Plate: {v['plate'] or '❌ Not detected'}")
                    print(f"   Position: {v['bbox']}")
                
                print(f"\n📊 Total vehicles: {result['total']}")