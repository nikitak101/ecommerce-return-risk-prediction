from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .schemas import OrderInput


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "orders.csv"
RETURN_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "return_model.pkl"
SIMILARITY_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "similarity_matrix.pkl"
CATEGORICAL_COLUMNS = [
    "marketplace_region", "category", "fabric", "fit_type", "device", "price_bucket"
]


def _load_feature_columns() -> list[str]:
    return list(joblib.load(RETURN_MODEL_PATH)["feature_columns"])


def _load_price_bucket_edges() -> np.ndarray:
    artifact = joblib.load(SIMILARITY_MODEL_PATH)
    edges = artifact.get("grouping_logic", {}).get("price_bucket_edges")
    if edges is not None:
        return np.asarray(edges, dtype=float)
    prices = pd.read_csv(DATA_PATH, usecols=["price_usd"])["price_usd"]
    return np.unique(prices.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy())


FEATURE_COLUMNS = _load_feature_columns()
PRICE_BUCKET_EDGES = _load_price_bucket_edges()


def make_price_bucket(price: Iterable[float] | pd.Series) -> pd.Series:
    prices = pd.Series(price)
    if len(PRICE_BUCKET_EDGES) < 4:
        return pd.Series("medium", index=prices.index, dtype="object")
    return pd.cut(
        prices,
        bins=PRICE_BUCKET_EDGES,
        labels=["low", "medium", "high"],
        include_lowest=True,
        duplicates="drop",
    ).astype("string")


def build_feature_row(order: OrderInput) -> pd.DataFrame:
    values: dict[str, Any] = order.model_dump()
    values["size_mismatch"] = abs(values["size_ordered"] - values["size_usual"])
    values["mismatch_direction"] = values["size_ordered"] - values["size_usual"]
    values["risky_fit_combo"] = int(
        values["fit_type"] in {"slim", "oversized"} and values["size_mismatch"] > 0
    )
    values["is_new_customer"] = int(values["customer_prior_orders"] == 0)
    values["price_bucket"] = make_price_bucket([values["price_usd"]]).iloc[0]
    values["discount_price_interaction"] = values["discount_pct"] * values["price_usd"]
    values["low_engagement"] = int(
        values["reviews_read"] == 0 and not values["size_chart_viewed"]
    )

    row = pd.DataFrame([values])
    encoded = pd.get_dummies(row, columns=CATEGORICAL_COLUMNS, dtype=float)
    return encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)