from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model import predict_risk, return_model
from .preprocessing import FEATURE_COLUMNS
from .recommend import get_recommendations, virtual_product_lookup
from .schemas import (
    OrderInput,
    PredictResponse,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
)


app = FastAPI(
    title="E-commerce Return Prediction API",
    version="1.0.0",
    description="Predicts order returns and recommends lower-return virtual products.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": type(return_model).__name__,
        "feature_count": len(FEATURE_COLUMNS),
        "virtual_product_count": len(virtual_product_lookup),
    }


@app.post("/predict")
def predict(request: OrderInput) -> PredictResponse:
    prediction, probability = predict_risk(request)
    return PredictResponse(returned=prediction, return_probability=probability)


@app.post("/recommendations")
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    virtual_product_id, alternatives = get_recommendations(
        request.category,
        request.fabric,
        request.fit_type,
        request.price_usd,
        request.top_n,
    )
    return RecommendationResponse(
        virtual_product_id=virtual_product_id,
        recommendations=[Recommendation(**alternative) for alternative in alternatives],
    )
