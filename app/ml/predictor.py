import os
import joblib
import pandas as pd
from typing import List
from app.schemas import PredictRequest

MODEL_FILE = "house_price_model.pkl"
META_FILE = "model_meta.pkl"

def load_model_and_metadata(model_dir: str = "models"):
    model_path = os.path.join(model_dir, MODEL_FILE)
    meta_path = os.path.join(model_dir, META_FILE)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None
    model = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    return {"model": model, "feature_columns": metadata["feature_columns"]}

def predict_from_dict(payload: PredictRequest, model_meta: dict) -> List[float]:
    """
    Accepts PredictRequest with either 'record' (single dict) or 'records' (list).
    Returns list of predictions (floats).
    """
    model = model_meta["model"]
    feature_cols = model_meta["feature_columns"]

    if payload.record is None and payload.records is None:
        raise ValueError("Provide either 'record' or 'records' in request body.")

    # Build DataFrame
    if payload.record is not None:
        df = pd.DataFrame([payload.record])
    else:
        df = pd.DataFrame(payload.records)

    # One-hot encode categorical columns present in the request (consistent with training)
    df_enc = pd.get_dummies(df, drop_first=True)

    # Align to training feature columns
    df_aligned = df_enc.reindex(columns=feature_cols, fill_value=0)

    preds = model.predict(df_aligned)
    return [float(p) for p in preds]
