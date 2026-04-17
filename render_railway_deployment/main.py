from typing import Any, Dict
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from llm_stage1 import extract_features
from llm_stage2 import interpret_prediction
from ml_model import load_model

# ---------------------------
# App initialization
# ---------------------------
app = FastAPI()
model = load_model("./best_pipeline_GradientBoosting.joblib")

# ---------------------------
# Request model
# ---------------------------
class PredictRequest(BaseModel):
    user_input: str
    json_feat: Dict[str, Any]

# ---------------------------
# Response model
# ---------------------------
class PredictResponse(BaseModel):
    status: str
    features: Any
    prediction: Any
    interpretation: str

# ---------------------------
# Utility
# ---------------------------
def clean_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common 'missing' representations into proper NaN."""
    return df.replace(["None", "none", "NaN", "nan", "", "null"], np.nan)

# ---------------------------
# Route
# ---------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> Dict[str, Any]:
    user_input = request.user_input
    json_feat = request.json_feat

    # Case 1: user left some features as "None"
    if "None" in list(json_feat.values()):

        features = extract_features(query=user_input, json_feats2=json_feat)

        df = pd.DataFrame([features])
        df = normalize_missing(df)

        if df.isnull().any().any():
            response = {
                "status": "partial",
                "features": df.to_dict(orient="records"),
                "prediction": 0,
                "interpretation": "please input all features",
            }
        else:
            price = model.predict(df)
            price_value = float(price[0]) if len(price) > 0 else None

            interpretation = interpret_prediction(df, price_value)

            response = {
                "status": "ok",
                "features": df.to_dict(orient="records"),
                "prediction": price_value,
                "interpretation": interpretation,
            }

    # Case 2: user provided all features directly
    else:
        df = pd.DataFrame([json_feat])
        df = normalize_missing(df)

        if df.isnull().any().any():
            response = {
                "status": "partial",
                "features": df.to_dict(orient="records"),
                "prediction": 0,
                "interpretation": "please input all features",
            }
        else:
            price = model.predict(df)
            price_value = float(price[0]) if len(price) > 0 else None

            interpretation = interpret_prediction(df, price_value)

            response = {
                "status": "ok",
                "features": df.to_dict(orient="records"),
                "prediction": price_value,
                "interpretation": interpretation,
            }

    return clean_for_json(response)

