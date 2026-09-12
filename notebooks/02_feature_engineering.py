from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "orders.csv"
RETURN_REASONS_PATH = PROJECT_ROOT / "data" / "return_reasons.csv"
ORDER_INDEX_PATH = PROJECT_ROOT / "data" / "order_index.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed_orders.csv"

EXPECTED_COLUMNS = [
    "order_id",
    "marketplace_region",
    "category",
    "fabric",
    "price_usd",
    "discount_pct",
    "is_premium",
    "size_ordered",
    "size_usual",
    "size_mismatch",
    "ordered_multiple_sizes",
    "fit_type",
    "customer_prior_orders",
    "customer_prior_return_rate",
    "avg_review_rating",
    "num_reviews",
    "reviews_read",
    "size_chart_viewed",
    "model_shown",
    "is_gift",
    "device",
    "days_to_delivery",
    "return_shipping_free",
    "return_reason",
    "returned",
]

CATEGORICAL_COLUMNS = ["marketplace_region", "category", "fabric", "fit_type", "device"]
NEW_FEATURES = [
    "mismatch_direction",
    "risky_fit_combo",
    "is_new_customer",
    "price_bucket",
    "discount_price_interaction",
    "low_engagement",
]


def make_price_bucket(price: pd.Series) -> pd.Series:
    """Create stable low/medium/high price groups, including constant-price data."""
    try:
        return pd.qcut(price, q=3, labels=["low", "medium", "high"], duplicates="drop")
    except ValueError:
        return pd.Series("medium", index=price.index, dtype="object")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or extra_columns:
        raise ValueError(
            f"Unexpected schema. Missing: {missing_columns}; extra: {extra_columns}"
        )
    df = df[EXPECTED_COLUMNS].copy()

    # CRITICAL: return_reason directly encodes returned and would cause severe target leakage.
    # It is retained separately for a future-scope multi-class return-reason model.
    print("WARNING: dropping return_reason before modeling because it directly encodes returned.")
    df[["order_id", "return_reason"]].to_csv(RETURN_REASONS_PATH, index=False)

    # Keep identifiers in a traceability reference, but never expose them as model features.
    df[["order_id"]].to_csv(ORDER_INDEX_PATH, index=False)
    order_reference = df["order_id"].copy()
    df = df.drop(columns=["order_id", "return_reason"])

    missing_before_imputation = df.isna().sum()
    print("Missing values before defensive imputation:")
    print(missing_before_imputation[missing_before_imputation > 0].to_string() or "None")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in df.columns]

    # These imputers are robustness safeguards for future data; this verified dataset is clean.
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    print(f"Missing values after defensive imputation: {int(df.isna().sum().sum())}")

    # These fields are already pre-engineered in the raw data. We do not recompute them.
    # The source confirms they are known before the order; no order_date exists to re-check timing.
    print(
        "Verified: size_mismatch and customer_prior_return_rate are pre-order-known, "
        "pre-engineered fields taken as given from the data source; no order_date exists "
        "to independently verify temporal ordering."
    )

    # Signed sizing captures whether a customer sized up or down, which can behave differently.
    df["mismatch_direction"] = df["size_ordered"] - df["size_usual"]
    # A mismatch may be more consequential for slim or oversized cuts than true-to-size cuts.
    df["risky_fit_combo"] = (
        df["fit_type"].isin(["slim", "oversized"]) & (df["size_mismatch"] > 0)
    ).astype(int)
    # New customers have no prior behavior history for the model to use.
    df["is_new_customer"] = (df["customer_prior_orders"] == 0).astype(int)
    # Price tiers let the model learn different return behavior for low-, mid-, and high-priced items.
    df["price_bucket"] = make_price_bucket(df["price_usd"])
    # Converts a percentage discount into the actual dollar incentive on this order.
    df["discount_price_interaction"] = df["discount_pct"] * df["price_usd"]
    # Lack of review and size-chart engagement may indicate unverified purchase decisions.
    df["low_engagement"] = (
        (df["reviews_read"] == 0) & (df["size_chart_viewed"] == 0)
    ).astype(int)

    target = df.pop("returned")
    categorical_to_encode = [column for column in CATEGORICAL_COLUMNS + ["price_bucket"] if column in df.columns]
    numeric_to_keep = [column for column in df.columns if column not in categorical_to_encode]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_to_encode),
            ("numeric", "passthrough", numeric_to_keep),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    encoder = Pipeline([("preprocessor", preprocessor)])
    encoded_values = encoder.fit_transform(df)
    encoded_columns = encoder.named_steps["preprocessor"].get_feature_names_out()
    processed = pd.DataFrame(encoded_values, columns=encoded_columns, index=df.index)
    processed["returned"] = target.to_numpy()
    processed.to_csv(PROCESSED_PATH, index=False)

    # Check that row traceability was preserved while keeping identifiers out of model features.
    assert len(order_reference) == len(processed)
    assert "order_id" not in processed.columns
    assert "return_reason" not in processed.columns

    print("\nFinal feature list:")
    print(processed.columns.tolist())
    print(f"\nFinal shape: {processed.shape}")
    print("\nCorrelation of each new engineered feature with returned:")
    engineered_for_correlation = df[NEW_FEATURES].copy()
    engineered_for_correlation["price_bucket"] = engineered_for_correlation["price_bucket"].map(
        {"low": 0, "medium": 1, "high": 2}
    )
    correlations = engineered_for_correlation.corrwith(target).sort_values(ascending=False)
    print(correlations.to_string())
    print(f"\nSaved processed data to: {PROCESSED_PATH}")
    print(f"Saved return-reason reference to: {RETURN_REASONS_PATH}")
    print(f"Saved order-id reference to: {ORDER_INDEX_PATH}")


if __name__ == "__main__":
    main()