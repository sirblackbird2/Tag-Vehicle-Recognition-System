from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.vehicle_service import VehicleRecognitionService
from app.models.schemas import PredictionResponse
import uuid
import os
from app.config import UPLOAD_DIR

router = APIRouter()
service = VehicleRecognitionService()

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.jpg"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process
    result = service.process_image(str(file_path))
    
    # Clean up
    os.remove(file_path)
    
    if "error" in result:
        raise HTTPException(400, result["error"])
    
    return result

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "Vehicle Recognition"}