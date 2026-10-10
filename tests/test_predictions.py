"""Unit and integration tests for AI4I classifier predictions and leakage defense."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.prediction import (
    predict_failure_risk,
    load_classifier,
    DEFAULT_MODEL_PATH
)
from src.data_validation import AI4I_APPROVED_FEATURES


def test_classifier_artifact_exists():
    """Confirms that the serialized model pipeline artifact exists."""
    assert Path(DEFAULT_MODEL_PATH).exists(), "Model pipeline artifact is missing!"


def test_load_classifier():
    """Loads pipeline and verifies expected stages (preprocessor + classifier)."""
    pipeline, metadata = load_classifier()
    assert hasattr(pipeline, "predict_proba")
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    if metadata:
        assert metadata["task"] == "failure_risk_classification"
        assert "selected_threshold" in metadata


def test_predict_single_valid_record():
    """Predicts a valid single record dictionary and verifies contract shape."""
    record = {
        "Type": "M",
        "Air temperature [K]": 298.2,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1408,
        "Torque [Nm]": 46.3,
        "Tool wear [min]": 3.0
    }
    result = predict_failure_risk(record)
    assert isinstance(result, dict)
    assert result["task"] == "failure_risk_classification"
    assert result["dataset_context"] == "UCI AI4I 2020 Predictive Maintenance"
    assert result["predicted_label"] in (0, 1)
    assert 0.0 <= result["model_score"] <= 1.0
    assert result["alert_level"] in ("NOMINAL", "CAUTION", "HIGH_ALERT")
    assert "Row-level snapshot classification" in result["message"]


def test_predict_batch_dataframe():
    """Predicts a batch DataFrame and verifies matched columns added."""
    data = [
        {
            "Type": "L",
            "Air temperature [K]": 298.9,
            "Process temperature [K]": 309.1,
            "Rotational speed [rpm]": 2861,
            "Torque [Nm]": 4.6,
            "Tool wear [min]": 143.0
        },
        {
            "Type": "H",
            "Air temperature [K]": 298.4,
            "Process temperature [K]": 308.9,
            "Rotational speed [rpm]": 1782,
            "Torque [Nm]": 23.9,
            "Tool wear [min]": 24.0
        }
    ]
    df = pd.DataFrame(data)
    result_df = predict_failure_risk(df)
    assert len(result_df) == 2
    assert "predicted_label" in result_df.columns
    assert "model_score" in result_df.columns
    assert "alert_level" in result_df.columns


def test_leakage_protection_in_prediction():
    """Providing target leakage fields (UDI, TWF, Machine failure) should not break prediction or leak."""
    record_with_leakage = {
        "UDI": 9999,
        "Product ID": "L99999",
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0,
        "TWF": 1,
        "Machine failure": 1
    }
    result = predict_failure_risk(record_with_leakage)
    assert isinstance(result, dict)
    assert "model_score" in result
    assert result["predicted_label"] in (0, 1)


def test_invalid_input_raises_error():
    """Missing required features should raise a ValueError."""
    bad_record = {
        "Type": "L",
        "Air temperature [K]": 300.0
    }
    with pytest.raises(ValueError, match="Schema validation failed"):
        predict_failure_risk(bad_record)
