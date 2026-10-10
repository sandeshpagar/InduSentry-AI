"""Inference interface for NASA C-MAPSS FD001 Remaining Useful Life (RUL) Regressor."""

from typing import Union, Dict, Any
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from src.data_validation import (
    validate_cmapss_trajectory,
    CMAPSS_REQUIRED_COLUMNS
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "cmapss_rul_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata" / "cmapss_rul_metadata.json"

_CACHED_RUL_PIPELINE = None
_CACHED_RUL_METADATA = None


def load_rul_regressor(model_path: Path = DEFAULT_MODEL_PATH, metadata_path: Path = DEFAULT_METADATA_PATH):
    """Loads and caches the fitted RUL regressor pipeline and metadata."""
    global _CACHED_RUL_PIPELINE, _CACHED_RUL_METADATA
    if _CACHED_RUL_PIPELINE is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"RUL model artifact not found at: {model_path}. Train the model first.")
        _CACHED_RUL_PIPELINE = joblib.load(model_path)
    
    if _CACHED_RUL_METADATA is None and Path(metadata_path).exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            _CACHED_RUL_METADATA = json.load(f)

    return _CACHED_RUL_PIPELINE, _CACHED_RUL_METADATA


def predict_remaining_useful_life(
    trajectory_df: pd.DataFrame,
    warning_threshold_cycles: float = 30.0,
    model_path: Path = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """Estimates remaining useful operating cycles for a monitored engine trajectory.
    
    Adheres to architecture.md response contract:
    - task: remaining_useful_life_regression
    - dataset_context: NASA C-MAPSS FD001
    - predicted_remaining_cycles: float
    - unit: operating_cycles
    - warning_threshold_cycles: float
    - alert_level: NOMINAL / CAUTION / HIGH_ALERT
    - message: Advisory limitation notice
    """
    pipeline, metadata = load_rul_regressor(model_path)
    
    # Validate trajectory schema and monotonicity
    is_valid, errors = validate_cmapss_trajectory(trajectory_df)
    if not is_valid:
        raise ValueError(f"Trajectory validation failed: {'; '.join(errors)}")

    feature_cols = metadata.get("feature_columns") if metadata else None
    if feature_cols is None:
        # Default features: settings + sensors (exclude unit_nr, cycle)
        feature_cols = [c for c in trajectory_df.columns if c not in ["unit_nr", "cycle", "RUL"]]

    # Extract the last observed cycle in the trajectory (latest condition)
    latest_observation = trajectory_df.sort_values(by="cycle").iloc[[-1]]
    X_latest = latest_observation[feature_cols].values

    # Predict remaining cycles
    pred_cycles = float(pipeline.predict(X_latest)[0])
    pred_cycles = max(0.0, float(np.round(pred_cycles, 1)))

    # Determine alert level
    if pred_cycles <= warning_threshold_cycles:
        alert = "HIGH_ALERT"
        msg = f"Critical RUL warning: Engine has an estimated {pred_cycles:.1f} cycles remaining (below {warning_threshold_cycles:.0f} cycle threshold)."
    elif pred_cycles <= (warning_threshold_cycles * 1.5):
        alert = "CAUTION"
        msg = f"Watch advisory: Engine has {pred_cycles:.1f} cycles remaining, approaching inspection threshold."
    else:
        alert = "NOMINAL"
        msg = f"Estimated RUL is {pred_cycles:.1f} operating cycles. Engine within normal lifecycle window."

    return {
        "task": "remaining_useful_life_regression",
        "dataset_context": "NASA C-MAPSS FD001",
        "predicted_remaining_cycles": pred_cycles,
        "unit": "operating_cycles",
        "warning_threshold_cycles": float(warning_threshold_cycles),
        "alert_level": alert,
        "message": f"{msg} (Note: Trajectory regression estimate based on C-MAPSS FD001 turbofan simulation)."
    }
