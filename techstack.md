# Technology Stack

## 1. Stack decision

Use a **Python-first, local-first ML application**. Jupyter notebooks are the main environment for exploration, training and evaluation. Streamlit is the dashboard. The first release does not need a separate Java/Node backend, a remote database or a cloud service.

This keeps the work aligned with the Applied Machine Learning subject: data preparation, model selection, validation, failure-risk classification and RUL forecasting remain the core technical contribution.

## 2. Recommended stack

| Layer | Choice | Purpose | MVP status |
|---|---|---|---|
| Language | Python 3.11 or 3.12 | ML and application logic | Required |
| Environment | `venv` / `.venv` | Isolated dependencies | Required |
| Notebook environment | JupyterLab / Jupyter Notebook | EDA, experiments and evaluation | Required |
| Data handling | pandas, NumPy | CSV parsing, transformations, feature preparation | Required |
| ML preprocessing/models | scikit-learn | Pipelines, encoders, models and metrics | Required |
| EDA charts | Matplotlib, Seaborn | Static plots for analysis/report | Required |
| Dashboard charts | Plotly | Interactive sensor charts and evaluation plots | Recommended |
| UI | Streamlit | Local interactive web dashboard | Required |
| Model storage | joblib | Save/load trusted fitted pipelines | Required |
| Tests | pytest | Unit and integration tests | Recommended |
| Linting | Ruff | Optional linting/formatting | Optional |
| Version control | Git and GitHub | Change history and backup | Recommended |
| Dataset A | UCI AI4I 2020 | Failure-risk classification | Required |
| Dataset B | NASA C-MAPSS FD001 first | RUL regression from trajectories | Required for complete objectives |

Use a stable Python version supported by the installed library releases. Do not upgrade the project's runtime blindly to the newest Python release. Record the exact versions that have been installed and verified.

## 3. Initial dependencies

A reasonable starter `requirements.txt` is:

```text
jupyterlab
ipykernel
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
joblib
pytest
```

This is an initial dependency list, not a tested lockfile. Install it in the virtual environment, run smoke tests, then pin the working versions. Do not add TensorFlow or PyTorch at the beginning; classical baselines and tree-based models are enough for the MVP. Add a sequence model only as an optional extension after the baseline is verified.

## 4. Dataset strategy

### AI4I 2020 Predictive Maintenance Dataset

Official source: <https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bda>

- Task: classification of the row-level `Machine failure` target.
- Candidate inputs: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- Exclude `UDI`, `Product ID` and failure mode fields `TWF`, `HDF`, `PWF`, `OSF`, `RNF` from predictors.
- Use a stratified train/validation/test plan; do not fit encoders/scalers on validation or test data.
- This dataset alone does not establish a future-time warning window or RUL.

### NASA C-MAPSS

Information and download: <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>

- Task: RUL regression using engine trajectories.
- Begin with FD001; document the raw file schema and engine/cycle ordering.
- Derive RUL targets from complete training trajectories, following the documented convention.
- Separate fitting/validation by engine identifier; never randomly split individual rows from one engine across both sets.
- For official test evaluation, predict from the last observed test trajectory and compare with the supplied terminal RUL labels.

## 5. Model strategy

### Failure classification

1. `DummyClassifier` baseline to establish a simple reference.
2. Logistic Regression using a preprocessing pipeline.
3. Random Forest Classifier.
4. Optional `HistGradientBoostingClassifier` or another compatible tabular model if justified.

Compare models with validation metrics. Focus on recall, precision, F1 and PR-AUC / average precision because failure labels are imbalanced. Accuracy is supplementary and must not be the only selection measure. Use class weights or other imbalance strategies only after documenting the approach. Select any alert threshold on validation data and reserve the final test set for a single final evaluation.

If the UI calls a value a probability, investigate calibration. Otherwise label it a model score/risk estimate and explain the interpretation.

### RUL regression

1. A simple regression baseline.
2. `RandomForestRegressor` or `HistGradientBoostingRegressor` using documented cycle/sensor features.
3. Optional feature-window experiments when they can be built without future-data leakage.

Report MAE and RMSE in operating cycles and include error analysis. Keep engine identities out of the predictive features. A prediction request must provide the same feature schema and context expected by the trained model; don't use a single isolated row if the model requires rolling-window features.

## 6. Application architecture choices

- **Local-first:** run notebooks and Streamlit on the student's laptop.
- **No database for MVP:** save metrics and metadata as JSON/CSV, keep current-session alerts in Streamlit session state, and allow CSV export.
- **No live sensor feed:** use manual values, a supported CSV or a trajectory example. Do not call this real-time industrial monitoring.
- **No Supabase requirement:** remote authentication and persistence do not support the core single-user educational objective.
- **No separate REST API for MVP:** import reusable Python inference modules directly into Streamlit. Consider FastAPI only if a separate web/mobile client later needs a model API.
- **Model artifacts:** save the fitted preprocessing and estimator together when practical. Only load artifacts generated by the trusted project training workflow.

## 7. IDE and repository practices

- Use Google Antigravity IDE or Antigravity 2.0 with the project directory opened as the workspace.
- Use PowerShell or the IDE's integrated terminal on Windows.
- Create `.venv` inside the project root and add it to `.gitignore`.
- Keep the Markdown specifications in the root: `prd.md`, `techstack.md`, `architecture.md`, `design.md`, `phasewiseprompt.md`.
- Keep raw datasets outside Git by default; provide official download instructions instead of committing large archives.
- Exclude `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, local secrets, temporary files and generated large data/model artifacts as appropriate.
- Track exact working dependency versions after the environment passes tests.

## 8. Security and responsible use

- Python pickle/joblib loading can execute code. Only load model artifacts created and reviewed within this project.
- Treat CSV uploads as data: validate types, required columns, row counts and numeric values before model inference.
- Never put credentials or private paths into source control.
- The system is advisory and not a certified industrial safety controller.
- Do not describe scores obtained on synthetic/simulated public data as validated results for factory machinery.

## 9. Useful documentation

- Python: <https://docs.python.org/3/>
- pandas: <https://pandas.pydata.org/docs/>
- scikit-learn: <https://scikit-learn.org/stable/>
- JupyterLab: <https://jupyterlab.readthedocs.io/>
- Streamlit: <https://docs.streamlit.io/>
- Plotly Python: <https://plotly.com/python/>
- pytest: <https://docs.pytest.org/>
- Google Antigravity IDE overview: <https://www.antigravity.google/docs/ide/overview/>
