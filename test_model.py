# test_model.py

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from target import (
    load_raw_data,
    FEATURE_COLUMNS,
    TARGET_COL,
    MODEL_PATH
)


def main():
    print("Loading model bundle…")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    print("Loading dataset…")
    df = load_raw_data()

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COL])
    y_true = df[TARGET_COL].astype(int)

    # Take only the features the model was trained on
    X = df[feature_columns].copy()

    # Run the model on the entire dataset
    print("Running predictions…")
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Evaluate
    print("\n==== MODEL EVALUATION ====\n")
    print(f"Accuracy:      {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:       {roc_auc_score(y_true, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
