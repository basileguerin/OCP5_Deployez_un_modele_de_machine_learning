from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from typing import Dict
from api.db import SessionLocal
from sqlalchemy import text
import uuid
from datetime import datetime
import json

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

@app.get("/metadata")
def metadata():
    return {
        "features_order": [str(x) for x in FEATURES_ORDER],
        "cols_to_scale": [str(x) for x in cols_to_scale],
        "threshold": float(threshold),
    }

@app.post("/predict")
def predict(data: PredictRequest):

    db = SessionLocal()

    try:
        # Vérification features
        missing = [f for f in FEATURES_ORDER if f not in data.features]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

        # Enregistrer input
        request_id = str(uuid.uuid4())

        db.execute(
            text("""
                INSERT INTO prediction_requests (request_id, input_features, requested_at)
                VALUES (:request_id, CAST(:features AS JSONB), :requested_at)
            """),
            {
                "request_id": request_id,
                "features": json.dumps(data.features),   # IMPORTANT
                "requested_at": datetime.utcnow(),
            }
        )

        # Préparation données
        X = np.array([[data.features[f] for f in FEATURES_ORDER]], dtype=float)

        X_df = pd.DataFrame(X, columns=FEATURES_ORDER)
        X_df[cols_to_scale] = scaler.transform(X_df[cols_to_scale])
        X_scaled = X_df.values

        # Prédiction
        proba = float(model.predict_proba(X_scaled)[0, 1])
        pred = int(proba >= threshold)

        # Enregistrer résultat
        db.execute(
            text("""
                INSERT INTO prediction_results (request_id, probability, prediction, threshold, predicted_at)
                VALUES (:request_id, :probability, :prediction, :threshold, :predicted_at)
            """),
            {
                "request_id": request_id,
                "probability": proba,
                "prediction": pred,
                "threshold": threshold,
                "predicted_at": datetime.utcnow(),
            }
        )

        db.commit()

        # Retour API
        return {
            "request_id": request_id,
            "probability": proba,
            "prediction": pred,
            "threshold": threshold
        }

    finally:
        db.close()

