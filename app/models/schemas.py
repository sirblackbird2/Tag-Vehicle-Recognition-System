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