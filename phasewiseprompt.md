# Antigravity Starter Prompt and Five-Phase Implementation Prompts

Use this file as the controlled development plan for the Industrial Machine Failure Early Warning System. Run **one phase at a time**. At the end of each phase, Antigravity must produce a detailed verification walkthrough and stop until the user explicitly approves moving to the next phase.

## How to use this file

1. Create/open the project directory in Antigravity.
2. Put these five documents in the project root: `prd.md`, `techstack.md`, `architecture.md`, `design.md`, `phasewiseprompt.md`.
3. Put the supplied dashboard image at `assets/dashboard-design-reference.png` if you want the design reference to remain available inside the workspace. In this kit the reference is already copied to `assets/`.
4. Paste the **Starter Prompt** first. It should only inspect the project specifications and propose the plan.
5. After reviewing the proposed plan, paste **Phase 1 Prompt**. After its walkthrough is verified, explicitly ask Antigravity to run Phase 2, and continue in the same way.
6. Do not ask the agent to implement all phases at once. The phase gates are intentional.

---

# 0. Starter Prompt — Read the specifications and prepare

Copy this prompt into Antigravity as the first message:

> I am building a college project titled **Industrial Machine Failure Early Warning System** for Applied Machine Learning for Industry Solutions.
>
> Before changing or creating any code, read these project specifications in full:
>
> - `prd.md`
> - `techstack.md`
> - `architecture.md`
> - `design.md`
> - `phasewiseprompt.md`
>
> Treat these documents as the project's current requirements. The dashboard screenshot in `design.md` is visual reference only. Do not reuse its LEGO branding, e-commerce labels, sales KPIs, names or records.
>
> **Your task in this first turn is planning only. Do not implement code yet.**
>
> 1. Inspect the current workspace and report whether it is empty or contains existing work. Do not overwrite any existing user files.
> 2. Summarize the intended system and explain the distinction between AI4I failure-risk classification and NASA C-MAPSS RUL forecasting.
> 3. Present the five implementation phases from `phasewiseprompt.md`, the deliverable for each and the dependencies between them.
> 4. Recommend the actual project tree to create, matching `architecture.md` unless there is a concrete conflict.
> 5. Identify environment prerequisites and how datasets will be placed in `data/raw/`. Do not invent model metrics or predictions if datasets are absent.
> 6. Point out ambiguities or technical risks before coding.
>
> Then stop and wait for my instruction to begin Phase 1. Do not automatically continue to another phase.

---

# Global rules — apply to every phase

Paste the relevant phase prompt as a complete task. The following rules apply to every phase even if not repeated in the task body:

1. Read the five Markdown specifications before making changes and inspect the actual workspace first.
2. Implement **only the currently requested phase**. Do not begin work from later phases “for convenience.”
3. Preserve working code. Do not replace or delete user files without explaining why and obtaining approval when a destructive change is needed.
4. Keep the app local-first: Python, Jupyter, scikit-learn, Streamlit and Plotly. No separate backend, Supabase, database, login system, cloud service or live industrial sensor integration in the MVP.
5. Keep the two ML tasks separate: AI4I `Machine failure` classification and NASA C-MAPSS RUL regression. The outputs must retain their own dataset context, model metadata, units and limits.
6. Never fabricate measured metrics, trained model artifacts, live data, model predictions, warning history or successful tests. If the data/model is unavailable, show an honest not-ready state and report the blocker.
7. Prevent target leakage. For AI4I exclude `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF` and `RNF` from the predictor features. For RUL split by engine, and only use information available at inference time.
8. Fit preprocessing on training data only. Keep final test data untouched for final evaluation. Choose alert thresholds using validation data or document them as provisional demonstration settings.
9. Keep the app advisory. Do not send machine-control commands or claim certified safety/real-factory performance.
10. Build readable, modular, typed where practical, commented code. Handle input and artifact errors with actionable messages.
11. Run the relevant checks and show actual command output or accurate summaries. Do not report a check as passed if it was skipped or failed.
12. **At the end of every phase, provide the Detailed Verification Walkthrough defined below and stop.** Ask me to confirm the walkthrough before progressing. Never automatically start the next phase.

## Required end-of-phase response: Detailed Verification Walkthrough

At the end of every phase, include the following sections, tailored to the work actually completed:

1. **Phase result:** completed / partially completed / blocked, with reasons.
2. **Changed files:** a concise list and each file's purpose.
3. **What now works:** describe implemented behavior, not just intended behavior.
4. **Environment/setup commands:** exact PowerShell commands for Windows where relevant, including current directory and environment activation. Do not assume command names that have not been created.
5. **Run walkthrough:** numbered steps that a student can follow in Antigravity/terminal/browser. Include what to click, what input to use and what should appear.
6. **Automated checks:** exact commands run, pass/fail/skipped status and key results. Include lint/type checks only if configured.
7. **Manual verification checklist:** observable expected results and common failure cases to test.
8. **Evidence:** where screenshots, generated metric files, logs or test reports have been saved. Do not claim screenshots exist unless created.
9. **Known issues and limitations:** clearly separate blockers from optional improvements.
10. **Approval gate:** end by asking me to verify the walkthrough and explicitly approve the next phase. Do not start it.

---

# Phase 1 Prompt — Project foundation, design system and UI shell

> Implement **Phase 1 only: project foundation, dataset validation scaffolding, and the Streamlit dashboard shell**.
>
> Read all five specifications, inspect the workspace and preserve any existing work. Use `design.md` as the visual and interaction spec. Do not train ML models in this phase.
>
> ## Tasks
>
> 1. Set up the recommended repository structure from `architecture.md`, including `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `app/`, `models/`, `reports/` and `tests/`. Add `README.md`, `requirements.txt` and `.gitignore` if missing.
> 2. Add the initial dependency list in `techstack.md`; document how to create and activate `.venv` on Windows PowerShell. Do not state that dependencies or versions work until you test them.
> 3. Implement the dashboard shell with the six pages: Overview, Failure Risk, Sensor Analytics, RUL Forecast, Model Evaluation, Alerts & History. The pages can show polished, helpful empty states where real artifacts are not available. Do not populate fake metrics or fake machine readings.
> 4. Match the supplied reference's visual language only: dark graphite canvas, left sidebar, compact status cards, chart panels and a records table. Use industrial project content and the palette/layout rules in `design.md`; do not copy LEGO/e-commerce text or content.
> 5. Implement reusable UI components for page headers, status badges, metric cards, empty states and error notices as appropriate. Keep ML/inference/business logic out of UI components.
> 6. Implement initial input-schema validation helpers for AI4I and C-MAPSS as described in `architecture.md`. For test fixtures only, use small clearly identified synthetic fixtures; never describe them as the real source datasets.
> 7. Add tests covering missing required columns, invalid/empty input and the relevant base validation behavior.
> 8. Add data download/setup instructions with the official UCI and NASA source links. If the datasets are not available locally, report that clearly rather than manufacturing data.
> 9. Make the app start using the documented command, for example `streamlit run app/streamlit_app.py`, and make the relevant tests runnable using `python -m pytest -q`.
>
> ## Acceptance criteria
>
> - All six pages render, navigation is clear and no unimplemented page claims to have working predictions.
> - The visual style follows `design.md`.
> - The app has meaningful empty states when models, datasets or metrics are absent.
> - Data validation returns readable messages and tests cover core cases.
> - Setup and run instructions match the repository actually created.
> - No ML model training is performed in this phase.
>
> Finish with the complete **Detailed Verification Walkthrough**. Show me how to create/activate the environment, install dependencies, run tests, launch the app, navigate each page and verify empty states/validation. Report actual results and stop for my approval before Phase 2.

### Phase 1 walkthrough checkpoints

- [ ] The app launches from a fresh terminal using documented instructions.
- [ ] The sidebar shows exactly the project-relevant pages.
- [ ] Each page title and explanation is correct.
- [ ] Model-not-ready and dataset-not-loaded states are informative and honest.
- [ ] The supplied image is used only as visual reference; no sales/LEGO content remains.
- [ ] Invalid/missing-column fixtures produce useful validation errors.
- [ ] Tests run and results are recorded.
- [ ] No fake ML metrics are shown.

---

# Phase 2 Prompt — AI4I failure-risk classification

> Implement **Phase 2 only: the AI4I failure-risk classification workflow**. Do not implement the NASA RUL model yet and do not do the full dashboard integration planned for Phase 4.
>
> Inspect the current code and Phase 1 walkthrough first. Preserve the design shell and working validation/tests. If Phase 1 issues remain, fix only issues that block Phase 2 and report the corrections.
>
> ## Tasks
>
> 1. Add or update `notebooks/01_data_exploration.ipynb` and `notebooks/02_failure_classification.ipynb`. Keep notebooks clear enough for a student to explain in an oral evaluation.
> 2. Use the official AI4I 2020 dataset from UCI. Check `data/raw/` first. If the dataset is absent and it cannot be downloaded, do not fabricate it: provide exact download and placement instructions and report that training is blocked pending the file.
> 3. Inspect dimensions, column types, missing values, target balance, summary statistics and useful feature relationships. Save report-quality figures to `reports/figures/` where possible.
> 4. Use `Machine failure` as the target and only the approved candidate inputs in the specification. Explicitly exclude `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF` and `RNF` from `X`.
> 5. Create reproducible stratified train/validation/test splits. Keep preprocessing fitted inside a scikit-learn pipeline so validation/test data do not influence preprocessing.
> 6. Fit a DummyClassifier baseline and compare Logistic Regression with a Random Forest (a compatible gradient-boosting candidate is optional). Report precision, recall, F1, confusion matrix, PR-AUC / average precision where supported, and accuracy as supplementary context.
> 7. Choose any decision threshold using validation data only and record why. Do not tune on the final test set. If the model score is shown as a probability, assess/document calibration; otherwise label it “model score” or “estimated risk score.”
> 8. Save the selected fitted preprocessing/model pipeline under `models/` plus readable model metadata and metrics under `reports/metrics/`. Record feature names, model settings, seed, split details, threshold and dataset source.
> 9. Implement reusable classifier inference functions in `src/` and tests for one valid record, batch predictions, missing columns and protection against leakage-prone input fields.
> 10. Do not replace overview cards with fabricated metrics. The dashboard may continue to show “not integrated yet” until Phase 4.
>
> ## Acceptance criteria
>
> - Training can be reproduced using the documented command/notebook.
> - No leakage-prone or ID columns are used as predictors.
> - Metrics come from actual validation/test runs, never guessed values.
> - The test set is kept separate until final evaluation.
> - Model and preprocessing are saved together or with an explicit artifact contract.
> - Classifier inference handles valid and invalid inputs.
>
> Finish with the complete **Detailed Verification Walkthrough**. Include the exact data file expected, notebook/run commands, how to reproduce training, where to inspect the confusion matrix and metrics, how to run inference tests and how to check the feature list for leakage. If training was blocked by missing data, explain exactly what I must download and where to put it. Stop and wait for approval before Phase 3.

### Phase 2 walkthrough checkpoints

- [ ] The dataset file is from the documented UCI source.
- [ ] EDA shows label imbalance and approved feature distributions.
- [ ] Feature list excludes IDs and failure-mode target proxies.
- [ ] The baseline and candidate classifier were actually run.
- [ ] Precision, recall, F1, confusion matrix and PR-AUC/AP are saved where supported.
- [ ] Threshold/model selection uses validation data, not final test results.
- [ ] A trained artifact loads and predicts a valid sample.
- [ ] Invalid input fails with an actionable message.
- [ ] Automated test results are reported accurately.

---

# Phase 3 Prompt — NASA C-MAPSS Remaining Useful Life forecasting

> Implement **Phase 3 only: the NASA C-MAPSS RUL regression workflow**, beginning with FD001. Keep the AI4I classifier unchanged and separate.
>
> Inspect the current project and Phase 2 walkthrough. Do not integrate RUL into the dashboard yet; that is Phase 4.
>
> ## Tasks
>
> 1. Create or complete `notebooks/03_rul_forecasting.ipynb` and reusable modules such as `src/train_rul.py` and `src/rul_prediction.py`.
> 2. Use the official NASA C-MAPSS source. Check for the expected FD001 training, test and terminal RUL label files in `data/raw/NASA_CMAPSS/`. If absent, report the exact files needed and do not fabricate trajectories or training results.
> 3. Parse each trajectory with engine ID, operating cycle, operating settings and sensor columns; sort each engine by cycle and validate the expected shape.
> 4. Derive training RUL from the last observed cycle of each complete training engine minus the current cycle. Document any target clipping or alternate convention; do not apply undocumented transformations.
> 5. Exclude engine ID as a predictive feature. Define the prediction schema clearly: use the last observed sensor/operating row and/or documented rolling-window features that can be computed at inference from the available past trajectory only.
> 6. Split training/validation by engine ID, not random row. Ensure that all rows for a held-out engine remain out of model fitting and preprocessing fitting.
> 7. Train a simple regression baseline and a RandomForestRegressor or HistGradientBoostingRegressor. Report MAE and RMSE in operating cycles, actual-versus-predicted plots and residual analysis.
> 8. Evaluate the official FD001 test trajectories using the provided terminal RUL labels when the files are available. Make sure the test prediction corresponds to the final observed point/window for each test engine.
> 9. Save the chosen fitted model and metadata, including feature schema, target convention, window definition, split identifiers, dataset subset and metrics.
> 10. Add RUL unit/integration tests covering feature construction, engine split separation, valid trajectory inference and invalid input.
>
> ## Acceptance criteria
>
> - The RUL target and feature convention are explicitly documented.
> - No engine is shared between fitting and validation splits.
> - No future samples or engine IDs leak into predictors.
> - Reported MAE/RMSE are measured from actual held-out trajectories and expressed in cycles.
> - The trained artifact requires the same schema/window features used during training.
> - AI4I classification artifacts and results are not overwritten or merged with RUL.
>
> Finish with the complete **Detailed Verification Walkthrough**. Show how the files are organized, how RUL labels are calculated, how engine-separated splitting is checked, how to train/evaluate, where MAE/RMSE and plots are saved, and how to test a sample trajectory. If blocked by missing NASA data, provide precise download/placement instructions. Stop before Phase 4.

### Phase 3 walkthrough checkpoints

- [ ] The subset is clearly identified as C-MAPSS FD001.
- [ ] RUL target convention matches the complete training trajectories.
- [ ] No engine appears in both fitting and validation sets.
- [ ] No engine ID or future trajectory value is used as an input feature.
- [ ] Test evaluation uses the supplied terminal RUL values where available.
- [ ] MAE and RMSE are shown in operating cycles.
- [ ] RUL artifact metadata defines exactly what input context it expects.
- [ ] Tests cover a valid and invalid trajectory input.

---

# Phase 4 Prompt — Integrate both models into the dashboard

> Implement **Phase 4 only: connect the existing saved ML workflows to the Streamlit dashboard and implement the actual warning/history behavior**. Do not retrain models automatically on app startup, and do not start final release documentation beyond updating necessary usage details.
>
> First inspect the current Phase 1 UI, Phase 2 classifier artifacts, Phase 3 RUL artifacts and their walkthroughs. Reuse the tested inference functions and metadata rather than rewriting model logic inside Streamlit.
>
> ## Tasks
>
> 1. Connect the Failure Risk page to the saved AI4I classifier for single-record and supported CSV-batch predictions.
> 2. Connect RUL Forecast to the saved C-MAPSS regressor using its actual required input schema and trajectory/window logic. Reject inputs that do not satisfy the model contract.
> 3. Connect Sensor Analytics to real dataset data: use distribution/relationship plots for AI4I and ordered cycle plots only for valid C-MAPSS trajectories.
> 4. Connect Model Evaluation to saved JSON/CSV metrics and generated figures. Clearly label dataset, model and units. Show an honest empty state if an evaluation artifact is absent.
> 5. Implement `src/alerts.py` or equivalent task-specific warning logic. Keep classification and RUL thresholds independent; document threshold origin and alert rationale.
> 6. Add an Overview page based on actual current-session counts and actual model/data readiness. Never display fabricated machine counts, uptime or fake historic alerts.
> 7. Implement Alerts & History using `st.session_state` or an equivalent session store. Include timestamp, task, dataset/model, output, units, threshold comparison, severity and reason. Add filters and CSV export.
> 8. Show progress for batch predictions. Report accepted/rejected rows accurately; do not silently ignore invalid records.
> 9. Add tests for task routing, saved-artifact loading, prediction output schemas, threshold boundary cases, session history and CSV export where practical.
> 10. Ensure all six pages follow `design.md` and no page uses static sample output as though it were a real prediction.
>
> ## Acceptance criteria
>
> - Single-row and batch AI4I predictions use the actual saved classifier pipeline.
> - RUL forecasts use the actual saved RUL pipeline and matching feature context.
> - Results identify the right task and dataset and show units/limitations.
> - Classifier and RUL warning rules are independent and tested at boundaries.
> - Metrics displayed are loaded from actual evaluation artifacts.
> - Alerts/history are session-based and export works.
> - Missing artifacts and invalid inputs produce graceful, actionable states.
> - No dashboard result claims live-sensor or future-window capability unsupported by the data.
>
> Finish with the complete **Detailed Verification Walkthrough**. Include exact app launch commands, click-by-click checks for every page, one valid and one invalid AI4I case, a batch CSV test, one valid and one invalid RUL trajectory test, warning-threshold boundary tests, model evaluation checks, alert-history checks and an export check. Report commands and results accurately. Stop before Phase 5 and wait for my approval.

### Phase 4 walkthrough checkpoints

- [ ] App uses saved artifacts; it does not retrain at launch.
- [ ] AI4I form schema matches approved classifier features.
- [ ] Batch results can be previewed and exported.
- [ ] RUL requires a supported trajectory/window.
- [ ] Charts reflect the correct kind of data and show units.
- [ ] Metrics are actual saved evaluation results.
- [ ] Task-specific alert thresholds work independently.
- [ ] History reflects actual predictions from the current session.
- [ ] Missing models and invalid inputs do not crash the app.

---

# Phase 5 Prompt — End-to-end verification, hardening and final handover

> Implement **Phase 5 only: final project verification, bug fixes, usability polish and handover documentation**. Do not add major features or replace the architecture.
>
> Inspect all five specification documents and review the completion walkthroughs from Phases 1–4. Run the application and tests as they exist; do not assume a test passes because source code looks correct.
>
> ## Tasks
>
> 1. Create a concise end-to-end test plan covering environment setup, app launch, page rendering, valid/invalid AI4I input, batch prediction, valid/invalid RUL trajectory, evaluation metrics, warning thresholds and CSV export.
> 2. Run all automated tests, fix genuine bugs, rerun relevant tests and record exact outcomes. Identify skipped tests and why they were skipped.
> 3. Verify that the app runs using a fresh documented setup, to the extent the current environment permits. If a clean environment cannot be recreated in this workspace, state that limitation honestly.
> 4. Check design consistency against `design.md`: dark industrial styling, correct navigation/content, responsive behavior, accessible contrast, readable labels and clear empty/error states.
> 5. Check the model result wording: no unsupported future-window prediction, no fabricated real-time data, no conflation of AI4I classifier with C-MAPSS RUL, no real-factory performance claims.
> 6. Confirm `.gitignore` excludes `.venv`, cache files, secrets and large local datasets/model artifacts according to the repository policy.
> 7. Finish `README.md` with setup, official dataset download/placement, training commands, artifact paths, app launch, tests, limitations and troubleshooting.
> 8. Add a report-ready `reports/project-summary.md` or similar with the actual methodology, model comparison tables and measured metrics if training was completed. If a workflow is blocked, clearly mark it as incomplete and do not manufacture results.
> 9. If screenshots can be created in the actual runtime, save representative final screenshots under `reports/screenshots/`. Do not claim screenshots exist unless they were captured.
> 10. Provide a final repository inventory and note any remaining issue that needs the student to act.
>
> ## Acceptance criteria
>
> - The environment/app/test instructions are accurate for the created project.
> - All test outcomes are recorded truthfully and critical failures are fixed or clearly blocked.
> - The final dashboard is coherent, usable and aligned with `design.md`.
> - Evaluation metrics and project claims are traceable to actual saved artifacts.
> - The final report identifies datasets, split strategy, limitations and next steps.
> - No real machine controls or unsupported live monitoring claims are present.
>
> Finish with the complete **Detailed Verification Walkthrough**, now serving as the final handover checklist. Include exact commands, a page-by-page manual verification walkthrough, test outcomes, key artifact paths, screenshots/evidence available, known limitations, and the final status of every PRD acceptance criterion. End by asking me to review the final handover; do not silently start extra work.

### Phase 5 final walkthrough checkpoints

- [ ] Fresh setup and app commands are verified or blocked with a clear reason.
- [ ] Full automated suite results are recorded.
- [ ] All pages and key scenarios were manually checked.
- [ ] Model metrics and alert behavior are traceable to real runs.
- [ ] No leakage, task confusion or unsupported claims remain known and undocumented.
- [ ] README and final summary match the actual implementation.
- [ ] Screenshot/evidence locations are accurate.
- [ ] Remaining limitations and required student actions are explicit.

---

# Final reminder for the agent

**Phase completion is not permission to continue.** After each phase, stop at the walkthrough/approval gate. Wait until the user explicitly requests the next phase. Do not fabricate data, model results, successful tests or screenshots to make the project appear finished.
