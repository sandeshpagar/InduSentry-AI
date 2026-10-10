"""Unit and integration tests for NASA C-MAPSS RUL forecasting regression."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.rul_prediction import (
    predict_remaining_useful_life,
    load_rul_regressor,
    DEFAULT_MODEL_PATH
)
from src.data_validation import CMAPSS_REQUIRED_COLUMNS


def create_sample_trajectory(num_cycles: int = 25) -> pd.DataFrame:
    """Helper creating a synthetic valid C-MAPSS FD001 engine trajectory."""
    data = {
        "unit_nr": [1] * num_cycles,
        "cycle": list(range(1, num_cycles + 1)),
        "setting_1": [0.001] * num_cycles,
        "setting_2": [0.0002] * num_cycles,
        "setting_3": [100.0] * num_cycles
    }
    for i in range(1, 22):
        data[f"s_{i}"] = [500.0 + i + (c * 0.1) for c in range(num_cycles)]
    return pd.DataFrame(data)


def test_rul_artifact_path_exists():
    """Confirms that the serialized RUL pipeline artifact is present."""
    assert Path(DEFAULT_MODEL_PATH).exists(), "RUL model artifact is missing!"


def test_load_rul_regressor():
    """Loads regressor and checks expected pipeline structure."""
    pipeline, metadata = load_rul_regressor()
    assert hasattr(pipeline, "predict")
    assert "scaler" in pipeline.named_steps
    assert "regressor" in pipeline.named_steps
    if metadata:
        assert metadata["task"] == "remaining_useful_life_regression"
        assert metadata["unit"] == "operating_cycles"


def test_predict_valid_trajectory():
    """Inference on a valid trajectory returns structured response and positive cycles."""
    traj_df = create_sample_trajectory(num_cycles=30)
    result = predict_remaining_useful_life(traj_df, warning_threshold_cycles=30.0)
    
    assert isinstance(result, dict)
    assert result["task"] == "remaining_useful_life_regression"
    assert result["dataset_context"] == "NASA C-MAPSS FD001"
    assert result["unit"] == "operating_cycles"
    assert isinstance(result["predicted_remaining_cycles"], float)
    assert result["predicted_remaining_cycles"] >= 0.0
    assert result["alert_level"] in ("NOMINAL", "CAUTION", "HIGH_ALERT")
    assert "C-MAPSS FD001" in result["message"]


def test_engine_id_not_used_in_model():
    """Confirms unit_nr is never in feature columns."""
    _, metadata = load_rul_regressor()
    if metadata:
        features = metadata["feature_columns"]
        assert "unit_nr" not in features, "Target leakage: unit_nr found in features!"
        assert "RUL" not in features, "Target leakage: RUL found in features!"


def test_invalid_trajectory_raises_error():
    """Non-monotonic cycles or missing sensors must trigger validation errors."""
    bad_traj = pd.DataFrame({
        "unit_nr": [1, 1],
        "cycle": [5, 2] # Non-monotonic
    })
    with pytest.raises(ValueError, match="Trajectory validation failed"):
        predict_remaining_useful_life(bad_traj)
