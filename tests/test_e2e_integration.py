"""End-to-end integration test validating complete workflow across all modules."""

import pytest
import pandas as pd
from pathlib import Path

from src.data_validation import validate_ai4i_record, validate_cmapss_trajectory
from src.prediction import predict_failure_risk
from src.rul_prediction import predict_remaining_useful_life
from src.alerts import evaluate_classifier_alert, evaluate_rul_alert, format_alert_log_entry


def test_full_ai4i_workflow_integration():
    """Validates raw record -> schema check -> prediction -> alert -> log entry."""
    raw_record = {
        "Type": "M",
        "Air temperature [K]": 298.2,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1408,
        "Torque [Nm]": 46.3,
        "Tool wear [min]": 3.0
    }
    
    # 1. Validation
    is_valid, errors = validate_ai4i_record(raw_record)
    assert is_valid is True
    assert len(errors) == 0

    # 2. Prediction
    pred_res = predict_failure_risk(raw_record, threshold=0.5)
    assert "model_score" in pred_res
    assert "predicted_label" in pred_res

    # 3. Alert evaluation
    alert = evaluate_classifier_alert(pred_res["model_score"], threshold=0.5)
    assert alert["alert_level"] in ("NOMINAL", "CAUTION", "HIGH_ALERT")

    # 4. Log entry formatting
    log_entry = format_alert_log_entry(
        task=pred_res["task"],
        context=pred_res["dataset_context"],
        metric_value=pred_res["model_score"],
        metric_unit="risk_score",
        alert_level=alert["alert_level"],
        reason=alert["reason"]
    )
    assert log_entry["task"] == "failure_risk_classification"
    assert "timestamp" in log_entry

    # 5. Serializability to DataFrame / CSV
    df = pd.DataFrame([log_entry])
    csv_str = df.to_csv(index=False)
    assert "timestamp" in csv_str
    assert "failure_risk_classification" in csv_str


def test_full_cmapss_workflow_integration():
    """Validates raw trajectory -> validation -> RUL forecast -> alert -> log entry."""
    raw_traj = {
        "unit_nr": [1, 1, 1],
        "cycle": [1, 2, 3],
        "setting_1": [0.0, 0.0, 0.0],
        "setting_2": [0.0, 0.0, 0.0],
        "setting_3": [100.0, 100.0, 100.0]
    }
    for i in range(1, 22):
        raw_traj[f"s_{i}"] = [500.0 + i, 501.0 + i, 502.0 + i]
    
    df_traj = pd.DataFrame(raw_traj)

    # 1. Validation
    is_valid, errors = validate_cmapss_trajectory(df_traj)
    assert is_valid is True
    assert len(errors) == 0

    # 2. RUL Prediction
    rul_res = predict_remaining_useful_life(df_traj, warning_threshold_cycles=30.0)
    assert "predicted_remaining_cycles" in rul_res
    assert rul_res["unit"] == "operating_cycles"

    # 3. Alert evaluation
    alert = evaluate_rul_alert(rul_res["predicted_remaining_cycles"], threshold_cycles=30.0)
    assert alert["alert_level"] in ("NOMINAL", "CAUTION", "HIGH_ALERT")

    # 4. Log entry formatting
    log_entry = format_alert_log_entry(
        task=rul_res["task"],
        context=rul_res["dataset_context"],
        metric_value=rul_res["predicted_remaining_cycles"],
        metric_unit="operating_cycles",
        alert_level=alert["alert_level"],
        reason=alert["reason"]
    )
    assert log_entry["task"] == "remaining_useful_life_regression"
    assert "timestamp" in log_entry
