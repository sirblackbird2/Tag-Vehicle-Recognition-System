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
    # Check if file exists
    if not file:
        raise HTTPException(400, "No file uploaded")
    
    # Check if there's any content
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file uploaded")
    
    # Reset file position so we can read it again later
    await file.seek(0)
    
    # Store the content for later
    file_content = content
    
    # Instead of strict content-type checking, just check if it looks like an image
    # by checking the file size and basic signature
    file_size = len(file_content)
    if file_size < 100:  # Too small to be a valid image
        raise HTTPException(400, "File too small to be an image")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.jpg"
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
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