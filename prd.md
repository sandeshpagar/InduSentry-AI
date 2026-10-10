# Product Requirements Document (PRD)

## 1. Product identity

- **Project title:** Industrial Machine Failure Early Warning System
- **Subject:** Applied Machine Learning for Industry Solutions (PECO311C)
- **Product type:** Local-first machine-learning prototype with an industrial monitoring dashboard
- **Primary users:** Student developer and project evaluator; maintenance operator is the demonstration persona
- **MVP form:** Python ML workflows in Jupyter Notebook plus a Streamlit dashboard

## 2. Product vision

Demonstrate how machine-learning models can use operating data to identify records associated with machine failure and estimate Remaining Useful Life (RUL) from run-to-failure trajectories. Present the outputs through a clear operator-facing dashboard that supports inspection and maintenance planning.

The first release is a **software prototype using public synthetic/simulated datasets**. It does not connect to physical sensors, industrial control systems, or safety-critical equipment.

## 3. Problem statement

Unexpected machinery failure may cause downtime, repair costs, production delays and safety risks. Reactive maintenance identifies problems only after a symptom or failure has occurred. This project explores how historical operating conditions and degradation trajectories can support more proactive inspection.

The product implements two separate ML tasks:

1. **Failure-risk classification:** predict the AI4I dataset's row-level `Machine failure` label from the approved operating variables.
2. **Remaining Useful Life forecasting:** estimate remaining operating cycles from NASA C-MAPSS engine trajectories.

These tasks use different datasets and targets. They must remain visibly distinct throughout training, evaluation, prediction and dashboard presentation. The AI4I dataset alone does **not** support a defensible estimate of when a machine will fail in a defined future time window or its true RUL.

## 4. Goals and success criteria

### Goals

- Build a reproducible end-to-end supervised ML workflow for failure-risk classification.
- Compare a baseline model with candidate classifiers using metrics appropriate for imbalanced labels.
- Build a separate RUL regression workflow on sequential run-to-failure data.
- Convert model results into configurable, transparent, advisory warnings.
- Deliver an intuitive, professional Streamlit dashboard inspired by the supplied dark industrial dashboard reference.
- Make the system explainable and demonstrable: show model provenance, metrics, input validation and known limitations.
- Provide reproducible setup instructions, tests and a final project report.

### Measurable completion criteria

- A fresh Python environment can be set up from the documented dependency file.
- The AI4I data-validation and EDA workflows complete successfully.
- The classifier trains without identifiers or failure-mode columns leaking target information.
- The classification report includes a confusion matrix, precision, recall, F1-score, PR-AUC / average precision when probability scores are available, and accuracy as a secondary metric.
- The RUL workflow derives targets correctly, separates engines between fitting and validation, and reports MAE/RMSE in operating cycles.
- The dashboard loads saved artifacts and supports the two tasks separately.
- Invalid files or missing artifacts produce clear user-facing errors rather than fabricated outputs.
- Warning thresholds are configurable and covered by tests.
- All charts and headline values come from loaded datasets, saved evaluation results, current user inputs or clearly labelled demo placeholders.
- The report states that public synthetic/simulated dataset results do not prove real-factory performance.

## 5. Scope

### In scope for MVP

- Dataset setup instructions and schema validation
- Exploratory data analysis (EDA)
- AI4I failure-risk classifier
- NASA C-MAPSS RUL regressor, starting with FD001
- Reusable preprocessing and inference modules
- Configurable warning logic
- Streamlit pages: Overview, Failure Risk, Sensor Analytics, RUL Forecast, Model Evaluation, Alerts & History
- Manual input and supported CSV uploads
- Session-level prediction/alert history and downloadable CSV export
- Automated unit tests and basic integration checks
- Project documentation and screenshots for evaluation

### Out of scope for MVP

- Live connection to machine sensors, PLC, SCADA, MQTT, OPC-UA or an IoT gateway
- Automatic machine shutdown or machine-control commands
- Industrial certification or safety guarantees
- Claiming proven reductions in real maintenance costs or downtime
- Production authentication, multi-user authorization, database, cloud deployment or mobile app
- A single model that merges AI4I classification with C-MAPSS RUL
- Deep learning / LSTM as a mandatory component. It may be explored later only if simpler baselines are sound and time permits.

## 6. Dataset and target specifications

### 6.1 AI4I 2020 Predictive Maintenance Dataset

Official source: <https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bda>

Use for failure-risk classification.

- **Target:** `Machine failure`
- **Approved candidate inputs:** `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- **Exclude from predictors:** `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
- Use a stratified train/validation/test strategy, record split settings and fit transformations on training data only.
- Never describe the row-level classification result as “failure within the next N minutes/hours/cycles.”
- If the dashboard displays a risk probability, document the interpretation and evaluate calibration, or label it accurately as a model score.

### 6.2 NASA C-MAPSS

Dataset information: <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>

Use for RUL regression; start with FD001.

- Treat each engine as one ordered operating-cycle trajectory.
- For run-to-failure training trajectories, derive each row's RUL from the engine's last training cycle minus its current cycle, unless another target convention is justified and documented.
- Do not use engine ID as a predictive feature.
- Split by engine identifier for validation; never distribute rows from one engine across both fitting and validation sets.
- When evaluating official test trajectories, predict from each trajectory's final observed window/row and use the provided terminal-RUL labels for scoring.
- Document feature-window construction, RUL target convention, clipping (if any), and metrics.

## 7. User stories

### Student / evaluator

- As a student, I want training notebooks to explain the workflow so that I can defend the methodology.
- As an evaluator, I want metrics and split details so that I can assess the models, not just the UI.
- As a student, I want saved artifacts so that I can run the dashboard without retraining every time.
- As an evaluator, I want the two ML tasks and their dataset limits clearly explained.

### Operator demonstration persona

- I want to enter a supported AI4I record or upload a valid CSV and see an explained classifier result.
- I want to select a supported C-MAPSS trajectory and see an RUL estimate in operating cycles.
- I want warnings to explain what triggered them and which model produced them.
- I want to inspect relevant data and export the results of the current session.

## 8. Functional requirements

### FR-01: Data validation

- Check required columns, data types, finite numeric values and empty files before analysis or inference.
- Report missing columns, unsupported schemas and invalid values in plain language.
- Apply sensible upload size/row-count limits for a local student application.

### FR-02: Data exploration

- Display dataset dimensions, types, missing-value summary, label distribution and descriptive statistics.
- Plot relevant variable distributions and relationships.
- Do not imply that unordered AI4I records form continuous per-machine histories.

### FR-03: Classifier training and evaluation

- Train a DummyClassifier baseline and at least one candidate classifier.
- Use a scikit-learn pipeline for preprocessing and model fitting.
- Focus on recall, precision, F1 and PR-AUC / average precision; report the confusion matrix and accuracy as secondary context.
- Choose any decision threshold using validation data only. Do not tune it on the final test set.
- Record model settings, feature list, random seed, split approach and evaluation metrics.

### FR-04: RUL training and evaluation

- Parse C-MAPSS engine ID, cycle, operating settings and sensor readings.
- Build the documented RUL target and prediction features without using future information unavailable at inference time.
- Evaluate on held-out engines and, when applicable, official C-MAPSS test trajectories with supplied terminal RUL labels.
- Report MAE and RMSE in operating cycles, plus actual-versus-predicted and residual plots.

### FR-05: Prediction interface

- Support a single-record AI4I prediction and batch CSV classification.
- Support a separate C-MAPSS RUL workflow requiring the matching trajectory/feature schema that the fitted model expects.
- Display the task name, dataset context, model artifact/version and relevant output units.
- Never invent a prediction if a model artifact is missing or input validation fails.

### FR-06: Warning logic

- Keep classifier and RUL warning rules separate.
- Use configurable thresholds selected/documented from validation or justified as demonstration settings.
- Explain alert reason, score/estimate, unit and advisory nature.
- Avoid an absolute “healthy” guarantee. Prefer “No high-risk alert from this model” where appropriate.
- Never create a machine-control action.

### FR-07: Dashboard pages

1. **Overview:** model availability, active data mode, current-session records and warnings, recent predictions.
2. **Failure Risk:** manual input, file upload, validation, prediction result and downloadable batch outputs.
3. **Sensor Analytics:** distributions and relationships for AI4I; cycle-based trends for valid C-MAPSS trajectories.
4. **RUL Forecast:** trajectory input, RUL estimate in cycles and dedicated RUL warning state.
5. **Model Evaluation:** saved classification and RUL metrics/plots, clearly labelled by dataset and model.
6. **Alerts & History:** session-based warning log, filtering and CSV export.

### FR-08: Reproducibility and testing

- Save fitted model and preprocessing pipeline together when feasible.
- Store evaluation metrics and model metadata in readable JSON/CSV files.
- Include a requirements file and setup/run instructions.
- Add tests for data validation, feature schemas, task routing, prediction output shape and alert thresholds.

## 9. Non-functional requirements

- **Usability:** clear labels, measurement units, task separation, accessible contrast and helpful error states.
- **Reliability:** invalid inputs fail safely without crashing the whole app.
- **Maintainability:** separate data loading, preprocessing, training, inference, warning logic and UI.
- **Reproducibility:** record random seeds, split identifiers, versions, features and dataset source.
- **Performance:** local batch predictions should feel interactive on a typical student laptop; measure actual runtime rather than promising an untested SLA.
- **Security:** load only trusted model artifacts, validate uploads, and keep secrets out of the repository.
- **Transparency:** display limitations and do not overstate what model scores mean.

## 10. UX and design requirements

- Use the supplied reference image only as visual inspiration: dark graphite dashboard, vertical left navigation, compact metric cards, charts and tabular records.
- Do not copy the reference's LEGO branding, sales terminology, customer names, invoice rows or business KPIs.
- Use industrial-machine labels, model states, risk scores, sensor charts and maintenance warning records appropriate to this project.
- Use status colours consistently, but always pair colour with text or an icon.
- Provide empty states while data or models are unavailable; do not fill the interface with fabricated “live” values.
- Detailed design rules are defined in `design.md`.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| AI4I class imbalance | Report recall/precision/PR-AUC and confusion matrix; consider class weights; tune only on validation data. |
| Target leakage | Exclude IDs and failure-mode flags; audit feature columns before training. |
| Inflated RUL evaluation | Split by engine and use appropriate official test RUL labels. |
| Misleading probability wording | Calibrate if probability interpretation is required; otherwise call it a model score. |
| False alarms or missed failures | Document threshold trade-offs and keep alerts advisory. |
| Dashboard ahead of model readiness | Start with honest empty states, then integrate saved models after evaluation. |
| Dependency conflicts | Use a dedicated virtual environment and pin tested versions after verification. |

## 12. Delivery plan

1. **Phase 1 — Foundation and UI shell:** repository/environment, project structure, design system, navigation, dataset validation and honest empty states.
2. **Phase 2 — AI4I classifier:** EDA, baseline/candidate training, evaluation, model artifact and inference tests.
3. **Phase 3 — C-MAPSS RUL:** trajectory processing, engine-based validation, regressor evaluation and saved artifact.
4. **Phase 4 — Dashboard integration:** real inference in both workflows, warning logic, analytics, evaluation displays and session history.
5. **Phase 5 — Verification and release:** full testing, walkthroughs, accessibility/responsiveness review, documentation and report evidence.

Every phase must end with a detailed walkthrough and verification report. The developer agent must stop and wait for user review/approval before moving to the next phase.

## 13. References

- UCI Machine Learning Repository. AI4I 2020 Predictive Maintenance Dataset. <https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bda>
- NASA. C-MAPSS Jet Engine Simulated Data. <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- scikit-learn documentation. <https://scikit-learn.org/stable/>
- Streamlit documentation. <https://docs.streamlit.io/>
