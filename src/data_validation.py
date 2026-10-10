"""Schema and data validation for AI4I 2020 and NASA C-MAPSS FD001 datasets.

Enforces strict task separation and guards against target leakage as defined in architecture.md.
"""

from typing import Any, Tuple, List
import pandas as pd
import numpy as np

# Approved candidate features for AI4I Failure Risk Classification
AI4I_APPROVED_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Strict target leakage / identifier columns that must NEVER be used as prediction inputs
AI4I_EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Machine failure"
]

VALID_MACHINE_TYPES = {"L", "M", "H"}

# Expected columns for C-MAPSS FD001 raw trajectories
CMAPSS_SETTINGS = ["setting_1", "setting_2", "setting_3"]
CMAPSS_SENSORS = [f"s_{i}" for i in range(1, 22)]
CMAPSS_REQUIRED_COLUMNS = ["unit_nr", "cycle"] + CMAPSS_SETTINGS + CMAPSS_SENSORS


def validate_ai4i_record(data: Any) -> Tuple[bool, List[str]]:
    """Validates an AI4I input record (dict or pandas DataFrame) against schema constraints.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    if data is None:
        return False, ["Input data cannot be None."]

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        return False, [f"Unsupported input type: {type(data).__name__}. Expected dict or pandas.DataFrame."]

    if df.empty:
        return False, ["Input dataset is empty."]

    # Check for target leakage or proxy fields
    leakage_present = [col for col in AI4I_EXCLUDED_COLUMNS if col in df.columns]
    if leakage_present:
        errors.append(
            f"Target leakage warning: input contains excluded fields ({', '.join(leakage_present)}). "
            "These must be removed from model predictors."
        )

    # Check for required approved features
    missing_cols = [col for col in AI4I_APPROVED_FEATURES if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required feature columns: {', '.join(missing_cols)}")
        return False, errors

    # Check Type values
    invalid_types = df[~df["Type"].astype(str).str.upper().isin(VALID_MACHINE_TYPES)]
    if not invalid_types.empty:
        errors.append(f"Invalid machine Type observed. Allowed values: {sorted(list(VALID_MACHINE_TYPES))}.")

    # Check numerical bounds & finiteness
    numeric_features = [col for col in AI4I_APPROVED_FEATURES if col != "Type"]
    for col in numeric_features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must contain numeric values.")
            continue

        if df[col].isnull().any():
            errors.append(f"Column '{col}' contains missing or NaN values.")

        if np.isinf(df[col]).any():
            errors.append(f"Column '{col}' contains infinite values.")

        # Physical sanity checks
        if "temperature" in col.lower() and (df[col] <= 0).any():
            errors.append(f"Physical constraint violation: '{col}' must be positive (Kelvin).")
        if col in ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"] and (df[col] < 0).any():
            errors.append(f"Physical constraint violation: '{col}' cannot be negative.")

    return len(errors) == 0, errors


def validate_cmapss_trajectory(df: Any) -> Tuple[bool, List[str]]:
    """Validates a NASA C-MAPSS FD001 engine trajectory dataset.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    if df is None:
        return False, ["Input trajectory cannot be None."]

    if not isinstance(df, pd.DataFrame):
        return False, [f"Unsupported input type: {type(df).__name__}. Expected pandas.DataFrame."]

    if df.empty:
        return False, ["Input trajectory dataset is empty."]

    # Check required columns
    missing_cols = [col for col in CMAPSS_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required C-MAPSS trajectory columns: {', '.join(missing_cols)}")
        return False, errors

    # Validate unit_nr and cycle are integer-like and positive
    for col in ["unit_nr", "cycle"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric.")
        elif (df[col] <= 0).any():
            errors.append(f"Column '{col}' must contain strictly positive values.")

    # Check cycle monotonicity per unit_nr
    if "unit_nr" in df.columns and "cycle" in df.columns:
        for unit_id, group in df.groupby("unit_nr"):
            if not group["cycle"].is_monotonic_increasing:
                errors.append(f"Engine unit {unit_id}: Operating cycles must be strictly ordered and monotonic increasing.")
                break

    # Check sensor values for nulls or infinite values
    sensor_cols = CMAPSS_SETTINGS + CMAPSS_SENSORS
    for col in sensor_cols:
        if df[col].isnull().any():
            errors.append(f"Trajectory column '{col}' contains missing values.")
            break
        if np.isinf(df[col]).any():
            errors.append(f"Trajectory column '{col}' contains infinite values.")
            break

    return len(errors) == 0, errors
