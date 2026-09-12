from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "similarity_matrix.pkl"
artifact = joblib.load(ARTIFACT_PATH)
virtual_product_lookup: pd.DataFrame = artifact["virtual_product_lookup"]
similarity_matrices = artifact["similarity_matrices"]
similarity_positions = artifact["similarity_positions"]
grouping_logic = artifact["grouping_logic"]
price_bucket_edges = np.asarray(grouping_logic["price_bucket_edges"], dtype=float)
price_bucket_labels = grouping_logic.get("price_bucket_labels", ["low", "medium", "high"])


def build_virtual_product_id(category: str, fabric: str, fit_type: str, price_usd: float) -> str:
    if len(price_bucket_edges) < 4:
        price_bucket = "medium"
    else:
        price_bucket = str(pd.cut(
            pd.Series([price_usd]), bins=price_bucket_edges,
            labels=price_bucket_labels, include_lowest=True, duplicates="drop"
        ).astype("string").iloc[0])
    return f"{category}_{fabric}_{fit_type}_{price_bucket}"


def get_alternatives(virtual_product_id: str, top_n: int = 3) -> list[dict]:
    if virtual_product_id not in virtual_product_lookup.index:
        return []
    product = virtual_product_lookup.loc[virtual_product_id]
    category = product["category"]
    category_ids = virtual_product_lookup.index[similarity_positions[category]]
    query_position = list(category_ids).index(virtual_product_id)
    candidates = virtual_product_lookup.loc[category_ids].copy()
    candidates["similarity"] = similarity_matrices[category][query_position]
    candidates = candidates[candidates.index != virtual_product_id]
    candidates = candidates[candidates["return_rate"] < product["return_rate"]]
    candidates = candidates[candidates["order_count"] >= 10]
    candidates = candidates.sort_values(
        ["similarity", "return_rate"], ascending=[False, True]
    ).head(top_n)
    return [
        {
            "virtual_product_id": candidate_id,
            "category": str(row["category"]),
            "fabric": str(row["fabric"]),
            "fit_type": str(row["fit_type"]),
            "avg_price": float(row["avg_price"]),
            "avg_rating": float(row["avg_rating"]),
            "return_rate": float(row["return_rate"]),
            "order_count": int(row["order_count"]),
        }
        for candidate_id, row in candidates.iterrows()
    ]


def get_recommendations(
    category: str, fabric: str, fit_type: str, price_usd: float, top_n: int = 3
) -> tuple[str, list[dict]]:
    virtual_product_id = build_virtual_product_id(category, fabric, fit_type, price_usd)
    return virtual_product_id, get_alternatives(virtual_product_id, top_n)