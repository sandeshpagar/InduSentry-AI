"""End-to-end training and evaluation script for NASA C-MAPSS FD001 RUL regression.

Follows the requirements in prd.md, techstack.md, and architecture.md:
1. Parse engine unit ID, cycle, 3 operating settings, and 21 sensors.
2. Derives RUL target per row as max(cycle per engine) - current cycle.
3. Strict engine-separated split: Train units (80 engines) vs Validation units (20 engines).
4. Drops engine ID (unit_nr) from predictors; fits StandardScaler strictly on training units.
5. Compares DummyRegressor (mean) vs RandomForestRegressor vs HistGradientBoostingRegressor.
6. Evaluates MAE and RMSE in operating cycles on held-out validation engines and official test set.
7. Serializes pipeline to models/cmapss_rul_pipeline.joblib and exports metrics to reports/metrics/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NASA_CMAPSS"
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
METADATA_DIR = MODELS_DIR / "metadata"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# 26 Raw columns specification
SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s_{i}" for i in range(1, 22)]
COLUMN_NAMES = ["unit_nr", "cycle"] + SETTINGS + SENSORS

# In FD001, sensors with constant/zero variance (s_1, s_5, s_10, s_16, s_18, s_19, setting_3) can be dropped or kept.
# We keep standard operational features and non-constant sensors:
EXCLUDED_FROM_FEATURES = ["unit_nr", "cycle", "RUL"]


def load_and_prep_training_data(filepath: Path = TRAIN_FILE) -> pd.DataFrame:
    """Loads space-separated C-MAPSS training file and derives row-level RUL."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    
    # Derive RUL: max_cycle for engine minus current cycle
    max_cycles = df.groupby("unit_nr")["cycle"].max().reset_index()
    max_cycles.rename(columns={"cycle": "max_cycle"}, inplace=True)
    df = df.merge(max_cycles, on="unit_nr", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def train_and_evaluate_rul(data_dir: Path = DATA_DIR):
    if not (data_dir / "train_FD001.txt").exists():
        raise FileNotFoundError(
            f"train_FD001.txt not found in {data_dir}. Run 'python src/download_cmapss.py' first."
        )

    print("Loading and preparing training trajectories...")
    df_train_all = load_and_prep_training_data(data_dir / "train_FD001.txt")
    
    # 1. Engine-Separated Train/Val Split (80 Train units, 20 Val units)
    unique_units = df_train_all["unit_nr"].unique()
    np.random.seed(42)
    val_units = np.random.choice(unique_units, size=20, replace=False)
    train_units = [u for u in unique_units if u not in val_units]

    print(f"Total Engines: {len(unique_units)} | Train Engines: {len(train_units)} | Val Engines: {len(val_units)}")
    # Verify zero engine overlap
    assert len(set(train_units).intersection(set(val_units))) == 0, "Engine leakage detected across splits!"

    train_mask = df_train_all["unit_nr"].isin(train_units)
    df_train = df_train_all[train_mask].copy()
    df_val = df_train_all[~train_mask].copy()

    # Define feature set
    feature_cols = [c for c in df_train_all.columns if c not in EXCLUDED_FROM_FEATURES]
    X_train = df_train[feature_cols].values
    y_train = df_train["RUL"].values

    X_val = df_val[feature_cols].values
    y_val = df_val["RUL"].values

    # Optional RUL clipping (commonly clipped to 125 cycles in literature to avoid early flat-line saturation)
    # y_train = np.minimum(y_train, 125)

    print(f"Features count: {len(feature_cols)} ({', '.join(feature_cols[:6])}...)")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # 2. Benchmark Models
    candidates = {
        "DummyBaseline": DummyRegressor(strategy="mean"),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
    }

    val_metrics = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])
        # Fit strictly on train engines
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict on validation engines
        y_val_pred = pipeline.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_val_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))

        val_metrics[name] = {"val_mae_cycles": mae, "val_rmse_cycles": rmse}
        print(f"[{name}] Val MAE: {mae:.2f} cycles | Val RMSE: {rmse:.2f} cycles")

    # Select best candidate on validation MAE
    best_name = "RandomForest" if val_metrics["RandomForest"]["val_mae_cycles"] <= val_metrics["HistGradientBoosting"]["val_mae_cycles"] else "HistGradientBoosting"
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nSelected Model: {best_name}")

    # 3. Test Evaluation on Official Test Trajectories (if available)
    test_metrics = {}
    test_file = data_dir / "test_FD001.txt"
    rul_file = data_dir / "RUL_FD001.txt"

    if test_file.exists() and rul_file.exists():
        df_test = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
        rul_true = pd.read_csv(rul_file, sep=r"\s+", header=None)[0].values

        # Predict from the final observed cycle of each test engine
        last_cycles = df_test.groupby("unit_nr").last().reset_index()
        X_test_last = last_cycles[feature_cols].values
        
        y_test_pred = best_pipeline.predict(X_test_last)
        test_mae = float(mean_absolute_error(rul_true, y_test_pred))
        test_rmse = float(np.sqrt(mean_squared_error(rul_true, y_test_pred)))

        test_metrics = {
            "test_engines_count": len(rul_true),
            "test_mae_cycles": test_mae,
            "test_rmse_cycles": test_rmse
        }
        print(f"Official Test Terminal RUL Evaluation: MAE={test_mae:.2f} cycles | RMSE={test_rmse:.2f} cycles")

    # 4. Save Artifact & Metrics
    model_artifact_path = MODELS_DIR / "cmapss_rul_pipeline.joblib"
    joblib.dump(best_pipeline, model_artifact_path)
    print(f"Saved fitted RUL pipeline to: {model_artifact_path}")

    metadata = {
        "model_name": best_name,
        "version": "1.0.0",
        "task": "remaining_useful_life_regression",
        "dataset": "NASA C-MAPSS FD001",
        "unit": "operating_cycles",
        "feature_columns": feature_cols,
        "train_engines_count": len(train_units),
        "val_engines_count": len(val_units),
        "validation_metrics": val_metrics,
        "official_test_metrics": test_metrics,
        "warning_threshold_cycles": 30.0
    }

    with open(METADATA_DIR / "cmapss_rul_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(METRICS_DIR / "cmapss_rul_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)

    print("Metadata and metrics successfully saved!")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_rul()
