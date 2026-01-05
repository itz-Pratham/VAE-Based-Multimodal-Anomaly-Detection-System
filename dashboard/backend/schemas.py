from pydantic import BaseModel
from typing import List, Optional

class InputData(BaseModel):
    """Schema for multimodal sensor data input"""
    timestamp: Optional[float] = None
    values: List[float]
    metadata: Optional[dict] = None

class PredictionResponse(BaseModel):
    """Schema for anomaly detection results"""
    is_anomaly: bool
    score: float
    latency_ms: float
    threshold: float