from typing import List

from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    marketplace_region: str
    category: str
    fabric: str
    price_usd: float
    discount_pct: float
    is_premium: bool
    size_ordered: float
    size_usual: float
    ordered_multiple_sizes: bool
    fit_type: str
    customer_prior_orders: int
    customer_prior_return_rate: float
    avg_review_rating: float
    num_reviews: int
    reviews_read: int
    size_chart_viewed: bool
    model_shown: bool
    is_gift: bool
    device: str
    days_to_delivery: int
    return_shipping_free: bool

    model_config = {"extra": "forbid"}


class PredictResponse(BaseModel):
    returned: int
    return_probability: float


class RecommendationRequest(BaseModel):
    category: str
    fabric: str
    fit_type: str
    price_usd: float
    top_n: int = Field(default=3, ge=1, le=20)

    model_config = {"extra": "forbid"}


class Recommendation(BaseModel):
    virtual_product_id: str
    category: str
    fabric: str
    fit_type: str
    avg_price: float
    avg_rating: float
    return_rate: float
    order_count: int


class RecommendationResponse(BaseModel):
    virtual_product_id: str | None
    recommendations: List[Recommendation]
