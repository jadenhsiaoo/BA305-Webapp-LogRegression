# target.py  (OPTIMIZED, CLASS-BALANCED LOGISTIC REGRESSION)
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ----------------------------------------------
# CONFIG
# ----------------------------------------------

DATA_PATH = "Anonymize_Loan_Default_data.csv"
MODEL_PATH = "loan_default_model.pkl"
TARGET_COL = "repay_fail"

# Your UI features — keep these
FEATURE_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "int_rate",
    "installment",
    "revol_bal",
]

# ----------------------------------------------
# DATA LOADING
# ----------------------------------------------

def load_raw_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1")
    return df

# ----------------------------------------------
# TRAINING (OPTIMIZED, CLASS-BALANCED LOGISTIC REGRESSION)
# ----------------------------------------------

def train_and_save_model(csv_path: str = DATA_PATH,
                         model_path: str = MODEL_PATH):

    df = load_raw_data(csv_path)

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X = df[FEATURE_COLUMNS].copy()

    # Compute means for prediction-time empty inputs
    feature_means = X.mean(numeric_only=True).to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # PIPELINE FOR HIGH ACCURACY
    # -------------------------

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), FEATURE_COLUMNS)
        ]
    )

    # L1 model for feature selection – class-weighted to handle imbalance
    base_l1_selector = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        max_iter=5000
    )

    # Final elastic-net logistic regression – also class-weighted
    final_clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        class_weight="balanced",
        max_iter=5000
    )

    # Full model pipeline
    clf = Pipeline(steps=[
        ("preprocessor", preprocess),
        ("poly", PolynomialFeatures(
            degree=2,
            interaction_only=True,
            include_bias=False
        )),
        ("select", SelectFromModel(base_l1_selector)),  # automatic feature reduction
        ("model", final_clf)
    ])

    clf.fit(X_train, y_train)

    # -------------------------
    # EVALUATION
    # -------------------------

    y_proba = clf.predict_proba(X_test)[:, 1]

    # Use a lower threshold to improve recall for the minority class
    threshold = 0.40
    y_pred = (y_proba >= threshold).astype(int)

    print("\n=== Optimized, Class-Balanced Logistic Regression ===")
    print(f"Classification threshold used for report: {threshold:.2f}")
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # -------------------------
    # SAVE MODEL BUNDLE
    # -------------------------

    bundle = {
        "model": clf,
        "feature_means": feature_means,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump(bundle, model_path)
    print(f"\nSaved optimized logistic regression model to: {model_path}\n")

    return bundle

def evaluate_thresholds(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]

    for t in [0.2, 0.3, 0.4, 0.5]:
        preds = (probs >= t).astype(int)
        print(f"\n----- Threshold {t} -----")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))

# ----------------------------------------------
# LOADING
# ----------------------------------------------

def load_model_bundle(model_path: str = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        return train_and_save_model(DATA_PATH, model_path)
    return joblib.load(model_path)

# ----------------------------------------------
# PREDICTION (with missing → mean substitution)
# ----------------------------------------------

def _fill_missing_with_means(raw_inputs: dict,
                             feature_means: dict,
                             feature_columns: list[str]) -> dict:
    filled = {}

    for f in feature_columns:
        val = raw_inputs.get(f, "")

        if val is None or str(val).strip() == "":
            filled[f] = float(feature_means[f])
        else:
            try:
                filled[f] = float(val)
            except (ValueError, TypeError):
                filled[f] = float(feature_means[f])

    return filled


def predict_single(raw_inputs: dict, bundle=None, return_filled: bool = False):

    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]
    feature_means = bundle["feature_means"]
    feature_columns = bundle["feature_columns"]

    filled = _fill_missing_with_means(raw_inputs, feature_means, feature_columns)

    X = pd.DataFrame([filled], columns=feature_columns)

    prob = float(model.predict_proba(X)[0, 1])

    if return_filled:
        return prob, filled
    return prob


# ----------------------------------------------
# RUN TRAINING VIA CLI
# ----------------------------------------------

if __name__ == "__main__":
    train_and_save_model()
