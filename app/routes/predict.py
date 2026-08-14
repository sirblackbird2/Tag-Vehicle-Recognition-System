from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from app.services.vehicle_service import VehicleRecognitionService
from app.models.schemas import PredictionResponse
import uuid
import os
from app.config import UPLOAD_DIR

router = APIRouter()
service = VehicleRecognitionService()

# 10 MB — generous for a phone photo, small enough to bound memory use per request.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    annotate: bool = Query(
        False,
        description="If true, also return an 'annotated_image' field: a "
                     "base64-encoded JPEG with bounding boxes and labels "
                     "drawn on it, ready to render directly in a frontend.",
    ),
):
    # Check if file exists
    if not file:
        raise HTTPException(400, "No file uploaded")

    # Check if there's any content
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file uploaded")

    file_size = len(content)
    if file_size < 100:  # Too small to be a valid image
        raise HTTPException(400, "File too small to be an image")
    if file_size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File too large ({file_size} bytes). Max is {MAX_UPLOAD_BYTES} bytes."
        )

    # Save file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.jpg"

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # Process
        result = service.process_image(str(file_path), annotate=annotate)
    finally:
        # Always clean up the temp file, even if processing raises.
        if file_path.exists():
            os.remove(file_path)

    if "error" in result:
        raise HTTPException(400, result["error"])

    return result


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "Vehicle Recognition"}