from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed_orders.csv"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "return_model.pkl"
PLOT_PATH = PROJECT_ROOT / "notebooks" / "outputs" / "feature_importance.png"


def evaluate_model(name, model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)
    probabilities = model.predict_proba(features_test)[:, 1]
    metrics = {
        "Precision": precision_score(target_test, predictions, zero_division=0),
        "Recall": recall_score(target_test, predictions, zero_division=0),
        "F1-score": f1_score(target_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(target_test, probabilities),
    }
    print(f"\n{name}")
    print("Confusion matrix:")
    print(confusion_matrix(target_test, predictions))
    print(pd.Series(metrics).to_string(float_format=lambda value: f"{value:.4f}"))
    return model, metrics


def get_feature_importance(model, feature_columns):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_columns)
    return pd.Series(abs(model.coef_[0]), index=feature_columns)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    data = pd.read_csv(DATA_PATH)
    target_column = "returned"
    if target_column not in data.columns:
        raise ValueError(f"Expected target column '{target_column}' in {DATA_PATH}")

    feature_columns = [column for column in data.columns if column != target_column]
    features = data[feature_columns]
    target = data[target_column].astype(int)
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=42,
    )
    negative_count = int((target_train == 0).sum())
    positive_count = int((target_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=42,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBClassifier": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    print(f"Training rows: {len(features_train):,}; test rows: {len(features_test):,}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")
    fitted_models = {}
    all_metrics = {}
    for name, model in models.items():
        fitted_models[name], all_metrics[name] = evaluate_model(
            name, model, features_train, features_test, target_train, target_test
        )

    comparison = pd.DataFrame(all_metrics).T[["Precision", "Recall", "F1-score", "ROC-AUC"]]
    print("\nModel comparison:")
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    # ROC-AUC evaluates ranking across classification thresholds; accuracy can look strong
    # on this imbalanced dataset while mostly predicting the majority class.
    print(
        "\nSelection note: ROC-AUC is preferred to accuracy here because it measures "
        "positive/negative ranking across thresholds, while accuracy can hide poor "
        "return detection by favoring the majority non-return class."
    )
    winner_name = comparison["ROC-AUC"].idxmax()
    winner = fitted_models[winner_name]
    importance = get_feature_importance(winner, feature_columns).sort_values(ascending=False)

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 7))
    importance.head(15).sort_values().plot(kind="barh", ax=axis, color="#2563eb")
    axis.set_title(f"Top Feature Importance: {winner_name}")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)

    # The source CSV is already the fitted output of the feature-engineering pipeline.
    # Bundle a fitted pass-through pipeline so inference preserves this processed schema.
    encoders = FunctionTransformer(validate=False).fit(features_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": winner, "encoders": encoders, "feature_columns": feature_columns},
        MODEL_PATH,
    )

    print("\nFinal summary:")
    print(f"Winning model: {winner_name}")
    print(comparison.loc[winner_name].to_string(float_format=lambda value: f"{value:.4f}"))
    print("Top 5 most important features:")
    print(importance.head(5).to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Feature importance plot: {PLOT_PATH}")
    print(f"Saved model bundle: {MODEL_PATH}")


if __name__ == "__main__":
    main()