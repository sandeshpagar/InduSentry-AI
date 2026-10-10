# InduSentry AI
**Industrial Machine Failure Early Warning System**  
*Applied Machine Learning for Industry Solutions (PECO311C)*

InduSentry AI is a local-first machine learning application designed to provide early warning indicators for industrial machinery. It implements two strictly decoupled machine learning pipelines:
1. **Path A — Failure-Risk Classification (UCI AI4I 2020):** Supervised classification predicting row-level machine failure from operating parameters (`Type`, temperatures, rotational speed, torque, tool wear).
2. **Path B — Remaining Useful Life Forecasting (NASA C-MAPSS FD001):** Trajectory regression estimating remaining operating cycles for monitored turbofan engines.

---

## 1. Project Layout

```text
AMLISPro/
├── prd.md                    # Product Requirements Document
├── techstack.md              # Technical stack specifications
├── architecture.md           # System architecture & contracts
├── design.md                 # UI/UX design specifications
├── phasewiseprompt.md        # 5-Phase implementation prompts
├── README.md                 # Setup and execution guide
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── data/
│   ├── raw/                  # Official raw datasets (AI4I, C-MAPSS)
│   └── processed/            # Processed artifacts
├── notebooks/                # Jupyter research & EDA notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_failure_classification.ipynb
│   └── 03_rul_forecasting.ipynb
├── src/                      # Core logic
│   ├── __init__.py
│   ├── data_validation.py    # Schema contracts & anti-leakage checks
│   ├── download_ai4i.py      # AI4I official downloader
│   ├── download_cmapss.py    # C-MAPSS official downloader
│   ├── train_classifier.py   # AI4I classification training pipeline
│   ├── prediction.py         # AI4I inference interface
│   ├── train_rul.py          # C-MAPSS trajectory training pipeline
│   ├── rul_prediction.py     # C-MAPSS inference interface
│   └── alerts.py             # Advisory early warning engine
├── app/                      # Streamlit application
│   ├── streamlit_app.py      # Multi-page dashboard
│   └── components/
│       ├── __init__.py
│       └── ui_blocks.py      # Reusable dark industrial UI components
├── models/                   # Serialized model pipelines & metadata
│   ├── ai4i_classifier_pipeline.joblib
│   ├── cmapss_rul_pipeline.joblib
│   └── metadata/
├── reports/                  # Evaluation metrics and summaries
│   ├── project-summary.md    # Comprehensive academic final report
│   ├── metrics/              # JSON evaluation metrics
│   └── figures/              # Figures and plots
└── tests/                    # Pytest test suite (25 tests)
    ├── test_data_validation.py
    ├── test_predictions.py
    ├── test_rul.py
    ├── test_alerts.py
    └── test_e2e_integration.py
```

---

## 2. Environment Setup (Windows PowerShell)

Ensure Python 3.11 or 3.12 is installed on Windows:

```powershell
# 1. Create Python virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Dataset Setup

Download the official public datasets using the provided utilities:

```powershell
# Download UCI AI4I 2020 dataset
python src/download_ai4i.py

# Download NASA C-MAPSS FD001 dataset
python src/download_cmapss.py
```

Files will be placed into `data/raw/ai4i2020.csv` and `data/raw/NASA_CMAPSS/`.

---

## 4. Model Training & Evaluation

Train both pipelines without data leakage:

```powershell
# Train AI4I failure classifier (stratified 70/15/15 split, validation threshold tuning)
python src/train_classifier.py

# Train C-MAPSS RUL regressor (engine-separated 80/20 split, evaluated in operating cycles)
python src/train_rul.py
```

Serialized models will be saved to `models/*.joblib`, and metrics to `reports/metrics/*.json`.

---

## 5. Running the Streamlit Dashboard

Launch the application:

```powershell
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser. The dashboard features:
- **Overview:** System readiness chips, analyzed record counts, active warnings, and session prediction feed.
- **Failure Risk:** Single-record form and batch CSV upload with live risk scores and prediction badges.
- **Sensor Analytics:** Interactive Plotly distribution and degradation curve visualizations.
- **RUL Forecast:** Trajectory input, cycle forecasting, and advisory warnings.
- **Model Evaluation:** Real confusion matrix tables, PR-AUC, precision, recall, and cycle MAE/RMSE.
- **Alerts & History:** Filterable session event log with CSV export.

---

## 6. Running the Automated Test Suite

Run the full pytest suite (25 unit and integration tests):

```powershell
python -m pytest -v
```

---

## 7. Scope & Operational Limitations

- **Prototype Context:** This is an academic research software prototype built using public synthetic and simulated benchmark datasets.
- **No Physical Hardware:** It does not connect to physical machinery, PLCs, SCADA, or industrial controllers.
- **Advisory Only:** All early warning outputs are non-controlling decision aids.
