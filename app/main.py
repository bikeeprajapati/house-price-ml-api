from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from app.ml.trainer import train_and_save_model, evaluate_saved_model
from app.ml.predictor import load_model_and_metadata, predict_from_dict
from app.schemas import TrainResponse, PredictRequest, PredictResponse, EvalResponse

# Create required directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(title="House Price Prediction API",
              description="Train, evaluate and predict house prices using a linear regression model.",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return{"Welcome to home page"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train", response_model=TrainResponse)
def train_endpoint(csv_path: str = "data/data.csv"):
    """
    Trains the model using the CSV at the given path (relative to project root).
    On success saves model to models/ and returns training metrics.
    """
    # Ensure file exists
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail=f"CSV file not found at '{csv_path}'. Place your dataset there or upload via /upload-csv.")
    try:
        metrics = train_and_save_model(csv_path, model_dir=MODEL_DIR)
        return TrainResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV dataset directly via form upload. Saved to data/house_prices.csv
    """
    filename = os.path.join("data", "data.csv")
    contents = await file.read()
    with open(filename, "wb") as f:
        f.write(contents)
    return {"detail": f"Saved dataset to {filename}"}

@app.get("/evaluate", response_model=EvalResponse)
def evaluate_endpoint():
    """
    Evaluate the saved model (if exists) on the full dataset and return metrics.
    """
    model_path = os.path.join(MODEL_DIR, "house_price_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="No trained model found. Train first with /train.")
    try:
        metrics = evaluate_saved_model(os.path.join("data","data.csv"), model_dir=MODEL_DIR)
        return EvalResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    """
    Predict price for single or multiple records.
    Provide either "record" (single) or "records" (list) in the JSON body.
    """
    model_meta = load_model_and_metadata(MODEL_DIR)
    if model_meta is None:
        raise HTTPException(status_code=400, detail="No trained model found. Train first with /train.")
    try:
        preds = predict_from_dict(payload, model_meta)
        return PredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
