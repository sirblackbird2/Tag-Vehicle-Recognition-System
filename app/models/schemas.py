from pydantic import BaseModel
from typing import List, Optional

class Vehicle(BaseModel):
    type: str
    brand: Optional[str] = None  
    confidence: float
    bbox: List[int]
    plate: Optional[str] = None

class PredictionResponse(BaseModel):
    vehicles: List[Vehicle]
    total: int
    # Base64-encoded JPEG with bounding boxes/labels drawn on it.
    # Only populated when the request is made with ?annotate=true.
    annotated_image: Optional[str] = None