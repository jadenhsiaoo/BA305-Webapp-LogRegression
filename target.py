# target.py  (OPTIMIZED, CLASS-BALANCED LOGISTIC REGRESSION + GRAPH OUTPUT)
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ----------------------------------------------
# CONFIG
# ----------------------------------------------

DATA_PATH = "Anonymize_Loan_Default_data.csv"
MODEL_PATH = "loan_default_model.pkl"
TARGET_COL = "repay_fail"

# UI Features
FEATURE_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "int_rate",
    "installment",
    "funded_amnt",
    "total_rec_prncp",
]


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------
# DATA LOADING
# ----------------------------------------------

def load_raw_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1")
    return df

# ----------------------------------------------
# TRAINING (OPTIMIZED + GRAPH GENERATION)
# ----------------------------------------------

def train_and_save_model(csv_path: str = DATA_PATH,
                         model_path: str = MODEL_PATH):

    df = load_raw_data(csv_path)
    df = df.dropna(subset=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    X = df[FEATURE_COLUMNS].copy()

    feature_means = X.mean(numeric_only=True).to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), FEATURE_COLUMNS)
        ]
    )

    base_l1_selector = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        max_iter=5000
    )

    final_clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        class_weight="balanced",
        max_iter=5000
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocess),
        ("poly", PolynomialFeatures(
            degree=2,
            interaction_only=True,
            include_bias=False
        )),
        ("select", SelectFromModel(base_l1_selector)),
        ("model", final_clf)
    ])

    clf.fit(X_train, y_train)

    # -------------------------
    # EVALUATION
    # -------------------------

    y_proba = clf.predict_proba(X_test)[:, 1]
    threshold = 0.35
    y_pred = (y_proba >= threshold).astype(int)

    print("\n=== Optimized, Class-Balanced Logistic Regression ===")
    print(f"Classification threshold used for report: {threshold:.2f}")
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # -------------------------
    # SAVE GRAPH OUTPUTS
    # -------------------------

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(RESULTS_DIR / "pr_curve.png", dpi=200)
    plt.close()

    # 3. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # 4. Feature Importance from L1 Model
    selector = clf.named_steps["select"].estimator_
    feature_mask = clf.named_steps["select"].get_support()
    poly = clf.named_steps["poly"]

    poly_feature_names = poly.get_feature_names_out(FEATURE_COLUMNS)
    selected_features = np.array(poly_feature_names)[feature_mask]

    importance = np.abs(selector.coef_[0][feature_mask])

    # Sort by magnitude
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(selected_features[sorted_idx], importance[sorted_idx])
    plt.xlabel("Coefficient Importance")
    plt.title("Selected Feature Importances (L1-based)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=200)
    plt.close()

    print("\nSaved evaluation graphs in /results folder.\n")

    # -------------------------
    # SAVE MODEL BUNDLE
    # -------------------------

    bundle = {
        "model": clf,
        "feature_means": feature_means,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump(bundle, model_path)
    print(f"Saved optimized logistic regression model to: {model_path}\n")

    return bundle

# ----------------------------------------------
# LOADING / PREDICTION
# ----------------------------------------------

def load_model_bundle(model_path: str = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        return train_and_save_model(DATA_PATH, model_path)
    return joblib.load(model_path)


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
            except:
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
# RUN FROM CLI
# ----------------------------------------------

if __name__ == "__main__":
    train_and_save_model()
