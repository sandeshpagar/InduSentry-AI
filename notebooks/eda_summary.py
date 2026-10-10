"""AI4I 2020 Data Exploration and Preparation."""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_PATH = Path("../data/raw/ai4i2020.csv")
FIGURES_PATH = Path("../reports/figures")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# %%
# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# %%
# Target distribution
target_counts = df["Machine failure"].value_counts()
target_pct = df["Machine failure"].value_counts(normalize=True) * 100
print("Class Distribution:")
print(pd.DataFrame({"Count": target_counts, "Percentage": target_pct}))

# %%
# Summary statistics for approved operating variables
approved_numeric = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
print("\nDescriptive Statistics:")
print(df[approved_numeric].describe().round(2))

# %%
# Machine Type breakdown
print("\nMachine Type Breakdown:")
print(df["Type"].value_counts())

# Failure modes distribution
failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
print("\nFailure Modes Counts (Target proxies - Excluded from X):")
print(df[failure_modes].sum())
