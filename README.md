# InduSentry AI
**Industrial Machine Failure Early Warning System**  
*Applied Machine Learning for Industry Solutions (PECO311C)*

InduSentry AI is a local-first machine learning prototype that provides early warning indicators for industrial machinery. It implements two strictly separated machine learning workflows:
1. **Failure-Risk Classification (AI4I 2020):** Supervised classification predicting row-level machine failure from operating parameters (`Type`, temperatures, rotational speed, torque, tool wear).
2. **Remaining Useful Life Forecasting (NASA C-MAPSS FD001):** Run-to-failure trajectory regression estimating remaining operating cycles for monitored turbine engines.

---

## 1. Project Structure

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
│   └── processed/            # Cleaned/engineered data
├── notebooks/                # Jupyter research & EDA notebooks
├── src/                      # Core logic (validation, preprocessing, training, inference, alerts)
├── app/                      # Streamlit application & reusable UI components
├── models/                   # Serialized model pipelines & metadata
├── reports/                  # Metrics and figure outputs
└── tests/                    # Pytest unit and integration test suite
```

---

## 2. Environment Setup (Windows PowerShell)

Ensure Python 3.11 or 3.12 is installed. Open PowerShell in the project directory:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Dataset Setup

Place the official datasets in `data/raw/`:

### UCI AI4I 2020 Predictive Maintenance
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bda)
- File destination: `data/raw/ai4i2020.csv`

### NASA C-MAPSS (FD001)
- Source: [NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- File destination:
  - `data/raw/NASA_CMAPSS/train_FD001.txt`
  - `data/raw/NASA_CMAPSS/test_FD001.txt`
  - `data/raw/NASA_CMAPSS/RUL_FD001.txt`

---

## 4. Running the Dashboard

```powershell
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`.

---

## 5. Running Tests

```powershell
python -m pytest -v
```
