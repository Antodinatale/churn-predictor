"""
model.py
Handles model training, evaluation, and persistence.
Bridges the analytical notebook and the production API.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report

from features import prepare_features

MODEL_PATH = Path(__file__).parent / "model.pkl"


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Trains a Random Forest on the engineered features and returns the model.

    Uses class_weight='balanced' to handle potential class imbalance
    between churned and non-churned users.
    """
    X, y = prepare_features(df)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate_model(model: RandomForestClassifier, df: pd.DataFrame) -> dict:
    """
    Evaluates the model with 5-fold cross-validation.
    Returns a dict with mean accuracy, precision, recall, and F1.
    """
    X, y = prepare_features(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        return_train_score=False,
    )

    metrics = {
        "accuracy":  round(results["test_accuracy"].mean(), 4),
        "precision": round(results["test_precision"].mean(), 4),
        "recall":    round(results["test_recall"].mean(), 4),
        "f1":        round(results["test_f1"].mean(), 4),
    }
    return metrics


def save_model(model: RandomForestClassifier, path: Path = MODEL_PATH) -> None:
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: Path = MODEL_PATH) -> RandomForestClassifier:
    if not path.exists():
        raise FileNotFoundError(
            f"model.pkl not found at {path}. "
            "Run the notebook to train and save the model first."
        )
    return joblib.load(path)


if __name__ == "__main__":
    # Quick training smoke-test when run directly
    data_path = Path("/data/raw/github_users.csv")
    if not data_path.exists():
        print("No data found. Run scraper.py first.")
    else:
        df = pd.read_csv(data_path)
        print(f"Training on {len(df)} records...")
        model = train_model(df)
        metrics = evaluate_model(model, df)
        print("Cross-validation results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        save_model(model)
