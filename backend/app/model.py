from pathlib import Path

import joblib

from .preprocessing import build_feature_row
from .schemas import OrderInput


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "return_model.pkl"
return_model = joblib.load(MODEL_PATH)["model"]


def predict_risk(order: OrderInput) -> tuple[int, float]:
    row = build_feature_row(order)
    probability = float(return_model.predict_proba(row)[0, 1])
    prediction = int(return_model.predict(row)[0])
    return prediction, probability