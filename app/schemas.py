from pydantic import BaseModel
from typing import Dict, List, Optional

class TrainResponse(BaseModel):
    train_size: int
    test_size: int
    mae: float
    mse: float
    rmse: float
    r2: float
    model_path: str

class PredictRequest(BaseModel):
    # Either provide a single record:
    record: Optional[Dict[str, float]] = None
    # Or multiple records:
    records: Optional[List[Dict[str, float]]] = None

class PredictResponse(BaseModel):
    predictions: List[float]

class EvalResponse(BaseModel):
    samples: int
    mae: float
    mse: float
    rmse: float
    r2: float
