"""Task-specific Advisory Early Warning Engine for InduSentry AI.

Maintains strict separation between AI4I Failure-Risk Classification and NASA C-MAPSS RUL Regression.
All alerts are strictly advisory and non-controlling.
"""

from typing import Dict, Any
from datetime import datetime


def evaluate_classifier_alert(model_score: float, threshold: float = 0.5) -> Dict[str, Any]:
    """Evaluates early warning status for AI4I failure-risk classification score."""
    score = float(model_score)
    thresh = float(threshold)

    if score >= thresh:
        level = "HIGH_ALERT"
        reason = f"Model risk score ({score:.3f}) exceeds alert threshold ({thresh:.3f})."
        advisory = "High probability of machine failure under current operating conditions. Immediate inspection recommended."
    elif score >= (thresh * 0.7):
        level = "CAUTION"
        reason = f"Model risk score ({score:.3f}) is within watch boundary (>= 70% of {thresh:.3f})."
        advisory = "Elevated risk parameters detected. Schedule routine inspection."
    else:
        level = "NOMINAL"
        reason = f"Model risk score ({score:.3f}) is below alert threshold ({thresh:.3f})."
        advisory = "Operating parameters within nominal risk boundaries for this model."

    return {
        "alert_level": level,
        "reason": reason,
        "advisory_message": advisory,
        "threshold": thresh,
        "metric_value": score,
        "metric_unit": "risk_score"
    }


def evaluate_rul_alert(predicted_remaining_cycles: float, threshold_cycles: float = 30.0) -> Dict[str, Any]:
    """Evaluates early warning status for NASA C-MAPSS RUL forecast."""
    cycles = float(predicted_remaining_cycles)
    thresh = float(threshold_cycles)

    if cycles <= thresh:
        level = "HIGH_ALERT"
        reason = f"Estimated RUL ({cycles:.1f} cycles) is at or below critical threshold ({thresh:.1f} cycles)."
        advisory = "Urgent maintenance intervention advised before estimated run-to-failure cycle exhaustion."
    elif cycles <= (thresh * 1.5):
        level = "CAUTION"
        reason = f"Estimated RUL ({cycles:.1f} cycles) is within watch window (<= 1.5x of {thresh:.1f} cycles)."
        advisory = "Engine approaching maintenance window. Prepare replacement parts."
    else:
        level = "NOMINAL"
        reason = f"Estimated RUL ({cycles:.1f} cycles) is well above threshold ({thresh:.1f} cycles)."
        advisory = "Engine operating within acceptable lifespan margin."

    return {
        "alert_level": level,
        "reason": reason,
        "advisory_message": advisory,
        "threshold": thresh,
        "metric_value": cycles,
        "metric_unit": "operating_cycles"
    }


def format_alert_log_entry(
    task: str,
    context: str,
    metric_value: float,
    metric_unit: str,
    alert_level: str,
    reason: str,
    source: str = "Manual Input"
) -> Dict[str, Any]:
    """Standardizes session-state history record structure."""
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "dataset_context": context,
        "source": source,
        "output_value": round(float(metric_value), 2),
        "unit": metric_unit,
        "alert_level": alert_level,
        "reason": reason
    }
