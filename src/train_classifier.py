"""End-to-end training and evaluation script for AI4I 2020 Failure-Risk Classification.

Follows the requirements in prd.md, techstack.md, and architecture.md:
1. Strict exclusion of identifier and leakage fields (UDI, Product ID, TWF, HDF, PWF, OSF, RNF).
2. Stratified train/validation/test split (70% train, 15% validation, 15% test).
3. Learned transformations (OneHotEncoder for Type, StandardScaler for numerics) fit strictly on training set.
4. Baseline DummyClassifier vs. LogisticRegression vs. RandomForestClassifier.
5. Imbalanced metrics evaluation: Precision, Recall, F1, PR-AUC (Average Precision), ROC-AUC, Confusion Matrix.
6. Decision threshold selection performed exclusively on the Validation set.
7. Single final evaluation on the untouched Test set.
8. Artifact serialized to models/ai4i_classifier_pipeline.joblib and metrics saved to reports/metrics/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METADATA_DIR = MODELS_DIR / "metadata"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Approved features and strictly excluded leakage proxies
APPROVED_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET_COL = "Machine failure"


def build_preprocessor() -> ColumnTransformer:
    """Creates scikit-learn ColumnTransformer for categorical and continuous features."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES)
        ],
        remainder="drop"
    )


def train_and_evaluate(data_path: Path = DATA_PATH):
    if not data_path.exists():
        raise FileNotFoundError(
            f"AI4I dataset not found at {data_path}. Please place 'ai4i2020.csv' in 'data/raw/'."
        )

    print(f"Loading AI4I 2020 from {data_path}...")
    df = pd.read_csv(data_path)

    # Validate target and feature availability
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' missing!"
    for feat in APPROVED_FEATURES:
        assert feat in df.columns, f"Approved feature '{feat}' missing!"

    X = df[APPROVED_FEATURES].copy()
    y = df[TARGET_COL].astype(int).values

    print(f"Features: {APPROVED_FEATURES}")
    print(f"Class distribution: Failures={y.sum()} ({y.mean()*100:.2f}%), Normal={(y==0).sum()}")

    # 1. Stratified Split: 70% Train, 15% Val, 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    val_ratio_adjusted = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, stratify=y_train_val, random_state=42
    )

    print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 2. Define pipelines
    preprocessor = build_preprocessor()

    models = {
        "DummyBaseline": DummyClassifier(strategy="stratified", random_state=42),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    val_results = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Fit strictly on Train
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict on Validation
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)

        val_results[name] = {
            "precision": float(precision_score(y_val, y_val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_val_pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_val, y_val_proba)),
            "roc_auc": float(roc_auc_score(y_val, y_val_proba)),
            "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist()
        }
        print(f"[{name} - Val] PR-AUC: {val_results[name]['pr_auc']:.4f}, F1: {val_results[name]['f1']:.4f}, Recall: {val_results[name]['recall']:.4f}")

    # 3. Model & Threshold Selection (on Validation set only)
    selected_name = "RandomForest"
    selected_pipeline = fitted_pipelines[selected_name]

    # Threshold optimization on Validation
    y_val_proba = selected_pipeline.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    
    # Select threshold maximizing F1 on validation
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    selected_threshold = float(thresholds[best_idx])
    
    print(f"Optimal threshold chosen on validation: {selected_threshold:.4f} (Val F1: {f1_scores[best_idx]:.4f})")

    # 4. Final Single Evaluation on untouched Test set
    y_test_proba = selected_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    y_test_pred_tuned = (y_test_proba >= selected_threshold).astype(int)

    test_metrics = {
        "model_name": selected_name,
        "selected_threshold": selected_threshold,
        "test_pr_auc": float(average_precision_score(y_test, y_test_proba)),
        "test_roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "default_threshold_0.5": {
            "precision": float(precision_score(y_test, y_test_pred_default, zero_division=0)),
            "recall": float(recall_score(y_test, y_test_pred_default, zero_division=0)),
            "f1": float(f1_score(y_test, y_test_pred_default, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred_default).tolist()
        },
        "tuned_threshold": {
            "precision": float(precision_score(y_test, y_test_pred_tuned, zero_division=0)),
            "recall": float(recall_score(y_test, y_test_pred_tuned, zero_division=0)),
            "f1": float(f1_score(y_test, y_test_pred_tuned, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred_tuned).tolist()
        }
    }

    print("\n--- Final Test Evaluation ---")
    print(f"Test PR-AUC: {test_metrics['test_pr_auc']:.4f}")
    print(f"Tuned Threshold ({selected_threshold:.3f}) -> F1: {test_metrics['tuned_threshold']['f1']:.4f}, Recall: {test_metrics['tuned_threshold']['recall']:.4f}, Precision: {test_metrics['tuned_threshold']['precision']:.4f}")

    # 5. Save serialized pipeline and metadata
    model_artifact_path = MODELS_DIR / "ai4i_classifier_pipeline.joblib"
    joblib.dump(selected_pipeline, model_artifact_path)
    print(f"Saved fitted pipeline to: {model_artifact_path}")

    metadata = {
        "model_name": selected_name,
        "version": "1.0.0",
        "task": "failure_risk_classification",
        "dataset": "UCI AI4I 2020 Predictive Maintenance",
        "approved_features": APPROVED_FEATURES,
        "target": TARGET_COL,
        "split_strategy": "Stratified 70/15/15, random_state=42",
        "selected_threshold": selected_threshold,
        "metrics_summary": test_metrics,
        "validation_comparison": val_results
    }

    metadata_path = METADATA_DIR / "ai4i_classifier_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metrics_path = METRICS_DIR / "ai4i_classification_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved metrics to: {metrics_path}")
    return test_metrics


if __name__ == "__main__":
    train_and_evaluate()
