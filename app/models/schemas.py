from pydantic import BaseModel
from typing import List, Optional

class Vehicle(BaseModel):
    type: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    plate: Optional[str] = None

class PredictionResponse(BaseModel):
    vehicles: List[Vehicle]
    total: int