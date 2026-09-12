from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "orders.csv"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "similarity_matrix.pkl"
PRICE_BUCKET_LABELS = ["low", "medium", "high"]
GROUPING_COLUMNS = ["category", "fabric", "fit_type", "price_bucket"]

PRODUCT_LOOKUP = None
SIMILARITY_MATRIX = None
SIMILARITY_POSITIONS = None
FEATURE_SCALER = None
PRICE_BUCKET_EDGES = None


def compute_price_bucket(price: pd.Series, edges=None) -> pd.Series:
    """Apply training quantile buckets, or calculate them from supplied training prices."""
    if edges is None:
        quantiles = price.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
        edges = np.unique(quantiles)
    if len(edges) < 4:
        return pd.Series("medium", index=price.index, dtype="object")
    return pd.cut(
        price,
        bins=edges,
        labels=PRICE_BUCKET_LABELS,
        include_lowest=True,
        duplicates="drop",
    ).astype("string")


def make_virtual_product_id(category, fabric, fit_type, price_bucket):
    """Reproduce the persisted grouping key for backend inference."""
    return f"{category}_{fabric}_{fit_type}_{price_bucket}"


def build_virtual_product_table(orders: pd.DataFrame):
    orders = orders.copy()
    price_quantiles = orders["price_usd"].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    price_bucket_edges = np.unique(price_quantiles)
    orders["price_bucket"] = compute_price_bucket(orders["price_usd"], price_bucket_edges)
    orders["virtual_product_id"] = orders.apply(
        lambda row: make_virtual_product_id(
            row["category"], row["fabric"], row["fit_type"], row["price_bucket"]
        ),
        axis=1,
    )

    order_counts = orders["virtual_product_id"].value_counts()
    print(f"Unique virtual products: {order_counts.size:,}")
    print(
        "Orders per virtual product (min/median/max): "
        f"{order_counts.min()} / {order_counts.median():.0f} / {order_counts.max()}"
    )
    if order_counts.min() < 10:
        print(
            "LIMITATION: some virtual products have fewer than 10 orders, so their "
            "historical return rates may be statistically noisy."
        )

    lookup = (
        orders.groupby("virtual_product_id", as_index=False)
        .agg(
            avg_price=("price_usd", "mean"),
            avg_rating=("avg_review_rating", "mean"),
            return_rate=("returned", "mean"),
            order_count=("returned", "size"),
            category=("category", "first"),
            fabric=("fabric", "first"),
            fit_type=("fit_type", "first"),
            price_bucket=("price_bucket", "first"),
        )
        .set_index("virtual_product_id")
    )
    return lookup, price_bucket_edges


def build_similarity_matrices(lookup: pd.DataFrame):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(lookup[["avg_price", "avg_rating"]])
    similarity_matrices = {}
    similarity_positions = {}
    for category, category_lookup in lookup.groupby("category", sort=True):
        positions = lookup.index.get_indexer(category_lookup.index)
        similarity_matrices[category] = cosine_similarity(scaled_features[positions])
        similarity_positions[category] = positions
    return similarity_matrices, similarity_positions, scaler


def load_artifact(artifact=None):
    """Load saved state into module globals for backend or notebook use."""
    global PRODUCT_LOOKUP, SIMILARITY_MATRIX, SIMILARITY_POSITIONS, FEATURE_SCALER
    global PRICE_BUCKET_EDGES
    artifact = artifact or joblib.load(MODEL_PATH)
    PRODUCT_LOOKUP = artifact["virtual_product_lookup"]
    SIMILARITY_MATRIX = artifact["similarity_matrices"]
    SIMILARITY_POSITIONS = artifact["similarity_positions"]
    FEATURE_SCALER = artifact["feature_scaler"]
    PRICE_BUCKET_EDGES = np.asarray(artifact["grouping_logic"]["price_bucket_edges"])


def get_alternatives(virtual_product_id, top_n=3):
    """Return same-category alternatives with a strictly lower historical return rate."""
    if PRODUCT_LOOKUP is None:
        load_artifact()
    if virtual_product_id not in PRODUCT_LOOKUP.index:
        raise KeyError(f"Unknown virtual_product_id: {virtual_product_id}")
    if top_n < 1:
        return []

    product = PRODUCT_LOOKUP.loc[virtual_product_id]
    category = product["category"]
    category_ids = PRODUCT_LOOKUP.index[SIMILARITY_POSITIONS[category]]
    query_position = list(category_ids).index(virtual_product_id)
    scores = SIMILARITY_MATRIX[category][query_position]
    candidates = PRODUCT_LOOKUP.loc[category_ids].copy()
    candidates["similarity"] = scores
    candidates = candidates[candidates.index != virtual_product_id]
    candidates = candidates[candidates["return_rate"] < product["return_rate"]]
    candidates = candidates[candidates["order_count"] >= 10]
    candidates = candidates.sort_values(
        ["similarity", "return_rate"], ascending=[False, True]
    )
    if len(candidates) < top_n:
        print(
            f"NOTE: only {len(candidates)} trustworthy alternatives found for "
            f"{virtual_product_id}; requested {top_n}."
        )
    candidates = candidates.head(top_n)
    return [
        {
            "virtual_product_id": candidate_id,
            "category": row["category"],
            "fabric": row["fabric"],
            "fit_type": row["fit_type"],
            "avg_price": float(row["avg_price"]),
            "avg_rating": float(row["avg_rating"]),
            "return_rate": float(row["return_rate"]),
            "order_count": int(row["order_count"]),
        }
        for candidate_id, row in candidates.iterrows()
    ]


def main():
    global PRODUCT_LOOKUP, SIMILARITY_MATRIX, SIMILARITY_POSITIONS, FEATURE_SCALER
    global PRICE_BUCKET_EDGES

    orders = pd.read_csv(RAW_PATH)
    required = {
        "category",
        "fabric",
        "fit_type",
        "price_usd",
        "avg_review_rating",
        "returned",
    }
    if not required.issubset(orders.columns):
        raise ValueError(f"Raw data is missing columns: {required - set(orders.columns)}")

    PRODUCT_LOOKUP, PRICE_BUCKET_EDGES = build_virtual_product_table(orders)
    SIMILARITY_MATRIX, SIMILARITY_POSITIONS, FEATURE_SCALER = build_similarity_matrices(
        PRODUCT_LOOKUP
    )
    artifact = {
        "similarity_matrices": SIMILARITY_MATRIX,
        "similarity_positions": SIMILARITY_POSITIONS,
        "virtual_product_lookup": PRODUCT_LOOKUP,
        "feature_scaler": FEATURE_SCALER,
        "grouping_logic": {
            "columns": GROUPING_COLUMNS,
            "price_bucket_edges": PRICE_BUCKET_EDGES.tolist(),
            "price_bucket_labels": PRICE_BUCKET_LABELS,
            "virtual_product_id_format": "{category}_{fabric}_{fit_type}_{price_bucket}",
            "price_bucket_method": "training-data 1/3 and 2/3 quantiles with pd.cut",
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    load_artifact(artifact)

    print(f"Virtual-product table shape: {PRODUCT_LOOKUP.shape}")
    print(f"Saved recommendation artifact: {MODEL_PATH}")
    print("\nExample recommendations:")
    examples = PRODUCT_LOOKUP[PRODUCT_LOOKUP["order_count"] >= 10].sort_values(
        "order_count", ascending=False
    ).head(3)
    for virtual_product_id in examples.index:
        recommendations = get_alternatives(virtual_product_id, top_n=3)
        print(f"{virtual_product_id} recommendations:")
        for recommendation in recommendations:
            print(
                f"  {recommendation['virtual_product_id']} "
                f"(order_count={recommendation['order_count']})"
            )


if __name__ == "__main__":
    main()
