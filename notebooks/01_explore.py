from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "orders.csv"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "outputs"


def find_binary_target(frame: pd.DataFrame) -> str:
    """Find the most likely binary return/target column."""
    preferred = [
        column
        for column in frame.columns
        if any(term in column.lower() for term in ("return", "target", "label", "outcome"))
    ]
    exact_matches = [
        column
        for column in frame.columns
        if column.lower() in {"returned", "return", "is_returned", "target", "label", "outcome"}
    ]
    if exact_matches and all(frame[column].dropna().nunique() == 2 for column in exact_matches):
        return exact_matches[0]

    binary_columns = [
        column for column in frame.columns if frame[column].dropna().nunique() == 2
    ]

    for column in preferred:
        if column in binary_columns:
            return column
    for column in binary_columns:
        if "id" not in column.lower():
            return column
    raise ValueError("Could not find a binary target column.")


def target_as_binary(series: pd.Series) -> pd.Series:
    """Convert common binary encodings to 0/1 for rate calculations."""
    if pd.api.types.is_numeric_dtype(series):
        values = sorted(series.dropna().unique())
        if values == [0, 1]:
            return series.astype(float)
        return series.map({values[0]: 0, values[-1]: 1}).astype(float)

    normalized = series.astype("string").str.strip().str.lower()
    positive = {"1", "true", "yes", "y", "returned", "return"}
    negative = {"0", "false", "no", "n", "not_returned", "not returned"}

    def convert_value(value: str) -> int | pd.NA:
        if value in positive:
            return 1
        if value in negative:
            return 0
        return pd.NA

    result = normalized.map(convert_value)
    if result.isna().any():
        values = list(normalized.dropna().unique())
        result = normalized.map({values[0]: 0, values[-1]: 1})
    return result.astype(float)


def choose_category(frame: pd.DataFrame, target_column: str) -> str:
    categorical = frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    categorical = [column for column in categorical if column != target_column]
    if not categorical:
        raise ValueError("Could not find a categorical column for the category chart.")

    preferred_terms = ("category", "product", "item", "type", "department", "segment")
    preferred = [column for column in categorical if any(term in column.lower() for term in preferred_terms)]
    return preferred[0] if preferred else categorical[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    target_column = find_binary_target(df)
    category_column = choose_category(df, target_column)
    target_binary = target_as_binary(df[target_column])

    print("Shape:")
    print(df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head())
    print("\nNull counts per column:")
    print(df.isna().sum())
    print(f"\nTarget column: {target_column}")
    print("Normalized target value counts:")
    print(df[target_column].value_counts(normalize=True, dropna=False))

    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns
    print("\nUnique value counts for categorical columns:")
    for column in categorical_columns:
        print(f"{column}: {df[column].nunique(dropna=False)}")

    numeric_columns = df.select_dtypes(include="number").columns
    print("\nSummary statistics for numeric columns:")
    print(df[numeric_columns].describe())

    correlation = df[numeric_columns].corr()
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    axis.set_yticks(range(len(correlation.columns)), correlation.columns)
    axis.set_title("Correlation Heatmap of Numeric Features")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(figure)

    chart_data = pd.DataFrame({category_column: df[category_column], "return_rate": target_binary})
    chart_data = chart_data.groupby(category_column, dropna=False)["return_rate"].mean().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 6))
    chart_data.plot(kind="bar", ax=axis, color="#d95f02")
    axis.set_title(f"Return Rate by {category_column}")
    axis.set_xlabel(category_column)
    axis.set_ylabel("Return rate")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "return_rate_by_category.png", dpi=150)
    plt.close(figure)

    missing_columns = df.columns[df.isna().any()].tolist()
    class_balance = df[target_column].value_counts(normalize=True, dropna=False).mul(100).round(2)
    print("\nSummary:")
    print(f"Rows: {len(df):,}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Target column: {target_column}")
    print("Class balance (%):")
    print(class_balance.to_string())
    print(f"Columns with missing values: {missing_columns or 'None'}")
    print(f"Category chart column: {category_column}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()