from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "classifier_employee.pkl"

obj = joblib.load(MODEL_PATH)
model = obj["model"]
scaler = obj["scaler"]
threshold = float(obj["seuil"])
cols_to_scale = obj["cols_to_scale"]

FEATURES_ORDER = list(model.feature_names_in_)

app = FastAPI(
    title="HRPredict API",
    description="API de prédiction du risque de démission",
    version="1.0",
)

class PredictRequest(BaseModel):
    features: Dict[str, float]

@app.post("/predict")
def predict(data: PredictRequest):
    # Vérification des features attendues
    missing = [f for f in FEATURES_ORDER if f not in data.features]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})

    X = np.array([[data.features[f] for f in FEATURES_ORDER]], dtype=float)

    X_df = pd.DataFrame(X, columns=FEATURES_ORDER)
    X_df[cols_to_scale] = scaler.transform(X_df[cols_to_scale])
    X_scaled = X_df.values

    proba = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(proba >= threshold)

    return {"probability": proba, "prediction": pred, "threshold": threshold}
