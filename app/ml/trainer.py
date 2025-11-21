import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.ml.data_loader import load_csv
from app.ml.preprocess import preprocess_dataframe

MODEL_FILE = "house_price_model.pkl"
META_FILE = "model_meta.pkl"

def train_and_save_model(csv_path: str, model_dir: str = "models"):
    # Load
    df = load_csv(csv_path)

    # Preprocess
    X, y, feature_cols = preprocess_dataframe(df, target_col="price")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test
    y_pred = model.predict(X_test)

    # Metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # Save model + metadata
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, MODEL_FILE)
    meta_path = os.path.join(model_dir, META_FILE)

    joblib.dump(model, model_path)
    # save metadata: feature columns
    metadata = {"feature_columns": feature_cols}
    joblib.dump(metadata, meta_path)

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "model_path": model_path
    }

def evaluate_saved_model(csv_path: str, model_dir: str = "models"):
    model_path = os.path.join(model_dir, MODEL_FILE)
    meta_path = os.path.join(model_dir, META_FILE)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Model or metadata not found. Train first.")

    model = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    feature_cols = metadata["feature_columns"]

    df = load_csv(csv_path)
    X, y, _ = preprocess_dataframe(df, target_col="price")

    # Align columns (in case dataset has different categorical encoding)
    X = X.reindex(columns=feature_cols, fill_value=0)

    y_pred = model.predict(X)

    mae = float(mean_absolute_error(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, y_pred))

    return {
        "samples": len(X),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
