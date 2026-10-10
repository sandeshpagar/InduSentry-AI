"""Unit tests for AI4I and NASA C-MAPSS schema validation."""

import pytest
import pandas as pd
import numpy as np
from src.data_validation import (
    validate_ai4i_record,
    validate_cmapss_trajectory,
    AI4I_APPROVED_FEATURES,
    CMAPSS_REQUIRED_COLUMNS
)


# ==========================================
# AI4I Validation Tests
# ==========================================

def test_valid_ai4i_dict_record():
    """A clean dictionary matching approved features should pass validation."""
    valid_record = {
        "Type": "M",
        "Air temperature [K]": 298.2,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1408,
        "Torque [Nm]": 46.3,
        "Tool wear [min]": 3.0
    }
    is_valid, errors = validate_ai4i_record(valid_record)
    assert is_valid is True
    assert len(errors) == 0


def test_ai4i_missing_required_columns():
    """Omitting an approved feature should cause validation failure."""
    incomplete_record = {
        "Type": "M",
        "Air temperature [K]": 298.2,
        # missing process temp, rotational speed, torque, tool wear
    }
    is_valid, errors = validate_ai4i_record(incomplete_record)
    assert is_valid is False
    assert any("Missing required feature columns" in err for err in errors)


def test_ai4i_rejects_empty_data():
    """Empty dict or DataFrame should be rejected."""
    is_valid, errors = validate_ai4i_record({})
    assert is_valid is False
    assert any("empty" in err for err in errors)

    is_valid_df, errors_df = validate_ai4i_record(pd.DataFrame())
    assert is_valid_df is False


def test_ai4i_flags_target_leakage_columns():
    """Presence of excluded columns (e.g. failure modes or IDs) must trigger a leakage warning/rejection."""
    record_with_leakage = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0,
        "TWF": 1,              # Tool wear failure flag (leakage proxy)
        "Machine failure": 1   # Target label
    }
    is_valid, errors = validate_ai4i_record(record_with_leakage)
    assert is_valid is False
    assert any("Target leakage warning" in err for err in errors)


def test_ai4i_physical_bounds_and_types():
    """Negative tool wear, negative temperatures, or infinite values must be caught."""
    invalid_record = {
        "Type": "L",
        "Air temperature [K]": -50.0,       # Kelvin must be positive
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": -5.0            # Wear cannot be negative
    }
    is_valid, errors = validate_ai4i_record(invalid_record)
    assert is_valid is False
    assert any("must be positive" in err for err in errors)
    assert any("cannot be negative" in err for err in errors)


# ==========================================
# C-MAPSS Trajectory Validation Tests
# ==========================================

def test_valid_cmapss_trajectory():
    """A trajectory matching C-MAPSS columns with monotonic cycles passes."""
    data = {
        "unit_nr": [1, 1, 1],
        "cycle": [1, 2, 3],
        "setting_1": [0.0, 0.0, 0.0],
        "setting_2": [0.0, 0.0, 0.0],
        "setting_3": [100.0, 100.0, 100.0]
    }
    # Add 21 sensor columns
    for i in range(1, 22):
        data[f"s_{i}"] = [500.0 + i, 501.0 + i, 502.0 + i]

    df = pd.DataFrame(data)
    is_valid, errors = validate_cmapss_trajectory(df)
    assert is_valid is True
    assert len(errors) == 0


def test_cmapss_non_monotonic_cycles():
    """Non-increasing cycle counts for an engine unit must be rejected."""
    data = {
        "unit_nr": [1, 1, 1],
        "cycle": [1, 3, 2], # Disordered
        "setting_1": [0.0, 0.0, 0.0],
        "setting_2": [0.0, 0.0, 0.0],
        "setting_3": [100.0, 100.0, 100.0]
    }
    for i in range(1, 22):
        data[f"s_{i}"] = [500.0, 500.0, 500.0]

    df = pd.DataFrame(data)
    is_valid, errors = validate_cmapss_trajectory(df)
    assert is_valid is False
    assert any("monotonic increasing" in err for err in errors)


def test_cmapss_missing_columns():
    """Trajectory missing sensor columns is rejected."""
    df = pd.DataFrame({
        "unit_nr": [1, 1],
        "cycle": [1, 2],
        "setting_1": [0.0, 0.0]
    })
    is_valid, errors = validate_cmapss_trajectory(df)
    assert is_valid is False
    assert any("Missing required C-MAPSS trajectory columns" in err for err in errors)
