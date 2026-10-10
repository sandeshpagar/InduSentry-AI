"""Inference interface for AI4I 2020 Machine Failure-Risk Classifier."""

from typing import Union, Dict, Any, List
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from src.data_validation import (
    validate_ai4i_record,
    AI4I_APPROVED_FEATURES,
    AI4I_EXCLUDED_COLUMNS
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ai4i_classifier_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata" / "ai4i_classifier_metadata.json"

_CACHED_PIPELINE = None
_CACHED_METADATA = None


def load_classifier(model_path: Path = DEFAULT_MODEL_PATH, metadata_path: Path = DEFAULT_METADATA_PATH):
    """Loads and caches the fitted classifier pipeline and metadata."""
    global _CACHED_PIPELINE, _CACHED_METADATA
    if _CACHED_PIPELINE is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path}. Train the model first.")
        _CACHED_PIPELINE = joblib.load(model_path)
    
    if _CACHED_METADATA is None and Path(metadata_path).exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            _CACHED_METADATA = json.load(f)

    return _CACHED_PIPELINE, _CACHED_METADATA


def predict_failure_risk(
    data: Union[Dict[str, Any], pd.DataFrame],
    threshold: float = None,
    model_path: Path = DEFAULT_MODEL_PATH
) -> Union[Dict[str, Any], pd.DataFrame]:
    """Computes failure-risk predictions for a single record or a batch DataFrame.
    
    Adheres to architecture.md response contract:
    - task: failure_risk_classification
    - dataset_context: AI4I 2020
    - predicted_label: 0 or 1
    - model_score: risk probability
    - threshold: decision boundary
    - alert_level: NOMINAL / CAUTION / HIGH_ALERT
    - message: Advisory limitation notice
    """
    pipeline, metadata = load_classifier(model_path)
    if threshold is None:
        threshold = metadata.get("selected_threshold", 0.5) if metadata else 0.5

    is_single_record = isinstance(data, dict)
    if is_single_record:
        df_input = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df_input = data.copy()
    else:
        raise ValueError("Input data must be a dict or a pandas DataFrame.")

    # Validate input schema
    is_valid, errors = validate_ai4i_record(df_input)
    if not is_valid:
        # Check if the only error is target leakage / excluded columns: strip them safely
        clean_errors = [e for e in errors if "Target leakage warning" not in e]
        if clean_errors:
            raise ValueError(f"Schema validation failed: {'; '.join(clean_errors)}")

    # Ensure strictly only approved features are passed to the pipeline in exact order
    X_inference = df_input[AI4I_APPROVED_FEATURES].copy()

    # Probability of class 1 (Machine failure)
    probabilities = pipeline.predict_proba(X_inference)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results: List[Dict[str, Any]] = []
    for prob, pred in zip(probabilities, predictions):
        score_val = float(np.round(prob, 4))
        
        if score_val >= threshold:
            alert = "HIGH_ALERT"
            msg = "High failure risk detected based on operating conditions. Immediate inspection advised."
        elif score_val >= (threshold * 0.7):
            alert = "CAUTION"
            msg = "Elevated risk score approaching threshold. Monitor operating parameters."
        else:
            alert = "NOMINAL"
            msg = "Operating parameters within nominal risk boundaries for this model."

        results.append({
            "task": "failure_risk_classification",
            "dataset_context": "UCI AI4I 2020 Predictive Maintenance",
            "predicted_label": int(pred),
            "model_score": score_val,
            "threshold": float(np.round(threshold, 4)),
            "alert_level": alert,
            "message": f"{msg} (Note: Row-level snapshot classification, not a future-time RUL prediction)."
        })

    if is_single_record:
        return results[0]
    
    result_df = df_input.copy()
    for key in ["predicted_label", "model_score", "alert_level", "message"]:
        result_df[key] = [r[key] for r in results]
    return result_df
