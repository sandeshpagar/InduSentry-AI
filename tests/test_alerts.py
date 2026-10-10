"""Unit tests for InduSentry AI Early Warning Engine (src/alerts.py)."""

import pytest
from src.alerts import evaluate_classifier_alert, evaluate_rul_alert, format_alert_log_entry


def test_classifier_alert_thresholds():
    """Verifies threshold boundary transitions for failure risk classifier."""
    # Nominal case (< 70% of 0.5 = < 0.35)
    res_nom = evaluate_classifier_alert(model_score=0.20, threshold=0.5)
    assert res_nom["alert_level"] == "NOMINAL"
    assert "below alert threshold" in res_nom["reason"]

    # Caution case (between 0.35 and 0.5)
    res_caut = evaluate_classifier_alert(model_score=0.40, threshold=0.5)
    assert res_caut["alert_level"] == "CAUTION"
    assert "within watch boundary" in res_caut["reason"]

    # High Alert case (>= 0.5)
    res_high = evaluate_classifier_alert(model_score=0.65, threshold=0.5)
    assert res_high["alert_level"] == "HIGH_ALERT"
    assert "exceeds alert threshold" in res_high["reason"]


def test_rul_alert_thresholds():
    """Verifies threshold boundary transitions for RUL forecast (in cycles)."""
    # Nominal case (> 1.5 * 30 = > 45 cycles)
    res_nom = evaluate_rul_alert(predicted_remaining_cycles=80.0, threshold_cycles=30.0)
    assert res_nom["alert_level"] == "NOMINAL"

    # Caution case (between 30 and 45 cycles)
    res_caut = evaluate_rul_alert(predicted_remaining_cycles=38.0, threshold_cycles=30.0)
    assert res_caut["alert_level"] == "CAUTION"

    # Critical High Alert case (<= 30 cycles)
    res_crit = evaluate_rul_alert(predicted_remaining_cycles=15.0, threshold_cycles=30.0)
    assert res_crit["alert_level"] == "HIGH_ALERT"
    assert "critical threshold" in res_crit["reason"]


def test_tasks_remain_independent():
    """Classifier and RUL functions must not interfere with each other's units and logic."""
    clf_res = evaluate_classifier_alert(model_score=0.8, threshold=0.5)
    rul_res = evaluate_rul_alert(predicted_remaining_cycles=0.8, threshold_cycles=30.0)

    # In classifier, high score = high alert
    assert clf_res["alert_level"] == "HIGH_ALERT"
    assert clf_res["metric_unit"] == "risk_score"

    # In RUL, low cycles = high alert
    assert rul_res["alert_level"] == "HIGH_ALERT"
    assert rul_res["metric_unit"] == "operating_cycles"


def test_format_alert_log_entry():
    """Validates structure and data types of standardized history log entries."""
    entry = format_alert_log_entry(
        task="Test Task",
        context="Synthetic Test",
        metric_value=42.123,
        metric_unit="cycles",
        alert_level="CAUTION",
        reason="Testing format",
        source="Test Source"
    )
    assert "timestamp" in entry
    assert entry["task"] == "Test Task"
    assert entry["output_value"] == 42.12
    assert entry["unit"] == "cycles"
    assert entry["alert_level"] == "CAUTION"
