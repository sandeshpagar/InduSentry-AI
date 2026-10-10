# Dashboard Design Specification

## 1. Purpose and design reference

This document describes the visual and interaction design for the **Industrial Machine Failure Early Warning System** dashboard. The supplied screenshot is a reference for layout, density and dark industrial styling only. It is not a source of product functionality or dashboard content.

![Reference image supplied by the project owner](assets/dashboard-design-reference.png)

### What to take from the reference

- A dark graphite interface with a persistent vertical left navigation.
- A clear page heading and compact top utility area.
- A grid of short, useful status/metric cards.
- A central chart panel alongside a secondary activity/alert panel.
- A lower table for recent records.
- Compact spacing, consistent card edges and restrained accent colours.

### What not to copy

Do not reuse the LEGO mark, sales/purchases/returns, marketing goals, customers, invoices, product IDs, sales data, names or financial values. Replace all of these with machine data, model status, prediction results, sensor analytics and maintenance warnings appropriate to this project.

## 2. Design principles

1. **ML output first:** the interface should help users understand a model result, not bury it beneath decoration.
2. **Honest status:** missing datasets, missing models and unevaluated metrics must have useful empty states. Never fabricate “live” readings or performance values.
3. **Separate tasks:** AI4I failure-risk classification and NASA C-MAPSS RUL forecasting must be visibly separate.
4. **Explain alerts:** every alert should show its task, reason, threshold/units and model/dataset context.
5. **Visual hierarchy:** summary cards first, analysis charts second, detailed tables lower on the page.
6. **Accessible colour:** do not communicate risk using colour alone. Pair status colours with text and/or an icon.
7. **Responsive layout:** on narrow screens, stack cards and charts, reduce table columns or enable horizontal scrolling, and preserve all key actions.
8. **Low-motion, high-clarity:** prefer subtle hover/focus states over ornamental animations, 3D effects or glowing effects.

## 3. Visual system

Use these colours as the starting palette; adjust only to improve contrast and consistency.

| Purpose | Suggested value | Use |
|---|---|---|
| App background | `#171A1C` | Main application canvas |
| Sidebar background | `#101315` | Navigation area |
| Card/panel background | `#222729` | KPI cards, charts, tables |
| Raised panel | `#292F31` | Table headers, popovers and selected areas |
| Primary text | `#F3F6F7` | Headings and key values |
| Secondary text | `#AAB4B8` | Captions, metadata and helper text |
| Primary accent | `#38C8CC` | Selected navigation, primary actions, neutral data highlights |
| Success / lower concern | `#43C887` | A defined low-risk/success state |
| Caution / watch | `#F0B44D` | Intermediate warning state |
| High alert | `#F06565` | High warning state |
| Borders | `#384043` | Subtle panel and table separation |

Use a readable sans-serif font. Streamlit's default font is acceptable; do not add external fonts unless there is a clear benefit. Maintain readable body text, clear heading levels and adequate contrast. Use one consistent border radius (approximately 8–12 px) and subtle shadows only when they improve separation.

## 4. Global application shell

### Sidebar

Recommended order:

1. Project mark/icon and shortened product name
2. Overview
3. Failure Risk
4. Sensor Analytics
5. RUL Forecast
6. Model Evaluation
7. Alerts & History
8. Divider
9. About / Data & Model Status

The sidebar should remain visually stable across pages. The selected page gets a restrained accent background or left indicator. Do not include irrelevant e-commerce items such as Products, Customers, Marketing, Invoices or Accounting.

### Header

Each page has:

- Page title and one-line explanation of the task.
- Dataset/task context where applicable.
- A compact status chip for data/model readiness.
- Optional refresh/reload control only if it performs a real action.

A search box is not required for MVP. Do not add controls that have no behavior.

### Main content grid

- Desktop: use a two- or three-column arrangement where meaningful; keep adequate whitespace.
- Tablet: reduce columns without making text or cards too narrow.
- Mobile: stack panels, keep main actions easy to reach and permit table scrolling.
- Prefer native Streamlit layout APIs and small reusable styling helpers over large blocks of fragile CSS.

## 5. Page designs

### 5.1 Overview

**Purpose:** show whether the system is ready and summarize the current session.

Recommended structure:

- Header: “Machine Monitoring Overview” plus subheading that the app is a local ML prototype.
- Top metric cards:
  - Classification Model: Ready / Not trained
  - RUL Model: Ready / Not trained
  - Records analyzed this session
  - Warnings generated this session
- Middle row:
  - A real, data-derived chart such as failure-risk category counts when predictions exist, otherwise an empty state.
  - Recent warnings or model/data status panel.
- Lower row:
  - Recent Predictions table: time, task, input source, result, unit/score if available, alert status.
- Quick actions: run a prediction, upload data, open model evaluation.

Never use fake “machines online”, “uptime percentage”, “97% health” or other unsupported operational claims.

### 5.2 Failure Risk

**Purpose:** run the AI4I row-level classifier.

Controls:

- Mode selector: Single Record / CSV Batch.
- Single-record inputs for `Type`, air temperature (K), process temperature (K), rotational speed (rpm), torque (Nm) and tool wear (min).
- CSV upload and preview in batch mode.
- Validation summary before inference.
- Explicit Analyze / Run Prediction button.

Result panel:

- Predicted label and model-estimated score/probability if supported.
- Configured decision threshold and resulting alert level.
- Model artifact and dataset context.
- Brief note: this is an AI4I row-level classification estimate, not a prediction that failure will occur within a chosen future time window.

Do not present the numeric result as a real-time sensor measurement. Use a neutral status until the model returns a result.

### 5.3 Sensor Analytics

**Purpose:** explore operating data with appropriate visualizations.

For AI4I:

- Dataset summary, variable distributions, missing-value summary and scatterplots between operating variables.
- Avoid a continuous time series because AI4I does not provide reliable per-machine ordered histories.

For C-MAPSS:

- Engine selector and cycle range for supported trajectory records.
- Selected sensor/operating variable vs. cycle plot.
- Optional rolling summaries only when derived from past observed data.

Always show dataset source and filters applied. Label axis units where known. Plots must be derived from loaded data, not fabricated.

### 5.4 RUL Forecast

**Purpose:** estimate remaining operating cycles for a supported C-MAPSS trajectory.

Controls:

- Dataset/subset label (initially FD001).
- Supported trajectory upload or documented demo example selector.
- Input validation showing engine/trajectory length and required schema.
- Run RUL Estimate button.
- Configurable warning threshold measured in operating cycles.

Results:

- Estimated remaining useful life in cycles.
- Threshold comparison and independent RUL warning state.
- Relevant sensor trajectory/degradation chart.
- Model ID, target convention and advisory disclaimer.

Do not allow an arbitrary isolated set of sensor values if the trained model requires sequence or rolling-window features.

### 5.5 Model Evaluation

**Purpose:** demonstrate scientific model evaluation for the project evaluator.

Two separate sections/tabs:

**Failure classifier**
- Precision, recall, F1, PR-AUC / average precision and accuracy.
- Confusion matrix and precision-recall curve.
- Model selected, dataset context, feature list, split strategy and threshold selection notes.

**RUL regressor**
- MAE and RMSE in cycles.
- Actual vs. predicted RUL plot and residual/error distribution.
- Engine-based split summary and target convention.

Load metrics generated by the training workflow. If an evaluation artifact is missing, display “Metrics not available yet” and steps to run evaluation rather than inventing sample scores.

### 5.6 Alerts & History

**Purpose:** let the user inspect model-generated warnings.

Table columns should include:

- Timestamp
- Task (Failure Risk / RUL)
- Dataset/model context
- Score or remaining cycles (with unit)
- Alert level
- Reason / threshold comparison
- Reviewed state, if session-level review is implemented

Add filters for task and severity, a clear empty state, and an export-to-CSV action. Store history in Streamlit session state for the MVP; state that it will not persist after a full app restart. Do not claim notification delivery by email/SMS unless implemented separately.

### 5.7 About / Data & Model Status

Include:

- Dataset sources and local file status.
- Model artifact status and version/metadata where available.
- Current application mode: uploaded dataset, saved local artifacts or labelled demo mode.
- Supported limitations: no live telemetry, no physical machine control, AI4I has no future-window target, C-MAPSS simulated RUL results are not factory validation.

## 6. Components to implement once and reuse

- `PageHeader`: title, caption and task/model status.
- `StatusBadge`: text plus state icon/colour.
- `MetricCard`: label, actual value, optional unit and explanatory caption.
- `EmptyState`: what is missing and the next action.
- `ErrorNotice`: readable validation or artifact error.
- `DatasetInfo`: data source, rows and schema summary.
- `PredictionResult`: model output, unit, threshold and caveat.
- `AlertTable`: consistent event display and export.
- `ChartPanel`: title, explanation and chart container.

Prefer native Streamlit components; use small CSS additions for layout/theme only. Keep all business logic out of UI components.

## 7. Interaction and state rules

- A form runs inference only when the user submits it; typing in a field must not trigger a costly model call on every keystroke.
- Show progress when processing a batch or trajectory.
- Store current-session prediction/alert history in `st.session_state`.
- Preserve current task and input values after harmless reruns where practical.
- Clear or invalidate old results when the relevant input changes so the page cannot display a result for different inputs.
- Catch expected validation/artifact errors and show a helpful next step.
- Keep classifier and RUL thresholds independent.
- Export only the rows actually processed; do not silently drop invalid rows. If partial processing is supported, report accepted/rejected row counts and reasons.

## 8. Responsive and accessibility requirements

- All form inputs have labels and units.
- All warnings include a text label; colour/icon is supplementary.
- Text and chart labels remain legible against the dark background.
- Keyboard focus and built-in controls remain usable.
- Long tables may scroll horizontally on small screens.
- Charts need descriptive titles and axis labels, not colour alone to convey meaning.
- Avoid flashing, auto-refresh loops and decorative movement.

## 9. Functional design acceptance checklist

- [ ] Sidebar includes only project-relevant destinations.
- [ ] All six pages render without an exception.
- [ ] Selected navigation state is obvious.
- [ ] Overview values are actual session/artifact values or clear empty states.
- [ ] Classifier input fields and CSV schema match the trained pipeline.
- [ ] RUL input validation matches the trained feature/window contract.
- [ ] Classification and RUL outputs are never merged or mislabelled.
- [ ] Metric page loads saved evaluation results and indicates when missing.
- [ ] Alerts show task, reason, unit/threshold and advisory limitation.
- [ ] History export works and exported columns are explained.
- [ ] Narrow layouts remain usable.
- [ ] No reference-image sales content or fabricated industrial metrics remain.
