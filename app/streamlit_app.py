"""InduSentry AI - Industrial Machine Failure Early Warning System.

Streamlit multi-page dashboard following the PRD, architecture.md, and design.md specs.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import (
    validate_ai4i_record,
    validate_cmapss_trajectory,
    AI4I_APPROVED_FEATURES
)
from app.components import (
    apply_theme_styling,
    render_header,
    render_status_chip,
    render_empty_state,
    render_error_notice
)

# 1. Page Configuration
st.set_page_config(
    page_title="InduSentry AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Dark Industrial Styling
apply_theme_styling()

# 3. Session State Initialization
if "session_records_count" not in st.session_state:
    st.session_state.session_records_count = 0
if "session_alerts" not in st.session_state:
    st.session_state.session_alerts = []

# 4. Sidebar Navigation
with st.sidebar:
    st.markdown("### ⚙️ **InduSentry AI**")
    st.caption("Early Warning Monitoring System")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "Overview",
            "Failure Risk",
            "Sensor Analytics",
            "RUL Forecast",
            "Model Evaluation",
            "Alerts & History"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("##### System Status")
    render_status_chip("Classifier", "Not trained", "neutral")
    render_status_chip("RUL Regressor", "Not trained", "neutral")
    st.caption("Environment: Phase 1 UI Shell")


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if selected_page == "Overview":
    render_header(
        title="Machine Monitoring Overview",
        subtitle="System readiness, session telemetry, and early warning status",
        task_badge="System Telemetry"
    )

    # Top KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="AI4I Classifier", value="Not trained", help="Train pipeline in Phase 2")
    with col2:
        st.metric(label="C-MAPSS RUL Regressor", value="Not trained", help="Train pipeline in Phase 3")
    with col3:
        st.metric(label="Records Analyzed", value=str(st.session_state.session_records_count))
    with col4:
        st.metric(label="Active Alerts", value=str(len(st.session_state.session_alerts)))

    st.markdown("#### Real-time Monitoring Feed")
    render_empty_state(
        title="No Telemetry or Predictions Processed Yet",
        message="Session has just started. Once models are trained in subsequent phases, live or batch records will populate here.",
        next_step="Navigate to 'Failure Risk' or 'RUL Forecast' to validate inputs or proceed to Phase 2 training."
    )

    st.markdown("#### Recent Prediction Events")
    if len(st.session_state.session_alerts) == 0:
        st.info("No prediction events recorded in this session.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.session_alerts), use_container_width=True)


# ==========================================
# PAGE 2: FAILURE RISK (AI4I CLASSIFIER)
# ==========================================
elif selected_page == "Failure Risk":
    render_header(
        title="Failure Risk Classification",
        subtitle="Row-level failure identification on AI4I operating variables",
        task_badge="Path A: AI4I 2020"
    )

    mode = st.radio("Input Mode", ["Single Operating Record", "Batch CSV Upload"], horizontal=True)

    if mode == "Single Operating Record":
        with st.form("ai4i_single_record_form"):
            st.markdown("##### Enter Machine Operating Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
                air_temp = st.number_input("Air temperature [K]", min_value=200.0, max_value=400.0, value=298.1)
            with col2:
                process_temp = st.number_input("Process temperature [K]", min_value=200.0, max_value=450.0, value=308.6)
                rot_speed = st.number_input("Rotational speed [rpm]", min_value=0.0, max_value=4000.0, value=1551.0)
            with col3:
                torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=200.0, value=42.8)
                tool_wear = st.number_input("Tool wear [min]", min_value=0.0, max_value=500.0, value=0.0)

            submit_val = st.form_submit_button("Validate & Analyze Record")

        if submit_val:
            record = {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rot_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear
            }
            is_valid, errors = validate_ai4i_record(record)
            if is_valid:
                st.success("✅ Input parameters conform to the AI4I approved feature contract.")
                render_empty_state(
                    title="Inference Engine Not Available (Phase 1)",
                    message="Schema validation passed. Machine failure classifier pipeline will be trained and integrated in Phase 2.",
                    next_step="Proceed to Phase 2 to fit and serialize the AI4I model."
                )
            else:
                render_error_notice(errors)

    else:
        uploaded_file = st.file_uploader("Upload CSV conforming to AI4I schema", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.markdown(f"**Loaded Preview:** {len(batch_df)} rows, {len(batch_df.columns)} columns")
                st.dataframe(batch_df.head(5), use_container_width=True)
                
                is_valid, errors = validate_ai4i_record(batch_df)
                if is_valid:
                    st.success("✅ Batch CSV schema validated successfully.")
                    render_empty_state(
                        title="Batch Inference Engine Pending",
                        message="Batch CSV passed validation. Model integration is scheduled for Phase 4.",
                        next_step="Proceed to Phase 2 to build the classifier pipeline."
                    )
                else:
                    render_error_notice(errors)
            except Exception as e:
                render_error_notice([f"Failed to read CSV: {str(e)}"])


# ==========================================
# PAGE 3: SENSOR ANALYTICS
# ==========================================
elif selected_page == "Sensor Analytics":
    render_header(
        title="Operating & Sensor Analytics",
        subtitle="Inspection of physical operating distributions and degradation curves",
        task_badge="Telemetry Inspection"
    )

    dataset_tab = st.radio("Analytics Target", ["AI4I Machine Operating Data", "C-MAPSS Jet Engine Degradation"], horizontal=True)

    if dataset_tab == "AI4I Machine Operating Data":
        render_empty_state(
            title="Raw AI4I Dataset Pending",
            message="Raw dataset `data/raw/ai4i2020.csv` is not yet present. Operating distributions and correlation heatmaps will render once EDA is completed in Phase 2.",
            next_step="Download `ai4i2020.csv` from UCI ML repository into `data/raw/`."
        )
    else:
        render_empty_state(
            title="C-MAPSS Trajectories Pending",
            message="Engine run-to-failure cycle trajectories from `data/raw/NASA_CMAPSS/` will be analyzed here in Phase 3.",
            next_step="Place `train_FD001.txt` in `data/raw/NASA_CMAPSS/`."
        )


# ==========================================
# PAGE 4: RUL FORECAST (NASA C-MAPSS FD001)
# ==========================================
elif selected_page == "RUL Forecast":
    render_header(
        title="Remaining Useful Life (RUL) Forecasting",
        subtitle="Trajectory-based cycle forecasting for monitored turbofan engines",
        task_badge="Path B: NASA C-MAPSS FD001"
    )

    st.markdown("##### Monitored Trajectory Input")
    st.caption("Expects ordered sequence: unit ID, cycle, 3 operating settings, and 21 sensor measurements.")

    uploaded_traj = st.file_uploader("Upload engine run-to-failure or test trajectory (CSV / Space-delimited)", type=["csv", "txt"])
    if uploaded_traj is not None:
        try:
            traj_df = pd.read_csv(uploaded_traj, sep=r"\s+", header=None)
            st.write(f"Trajectory uploaded: {len(traj_df)} rows")
            is_valid, errors = validate_cmapss_trajectory(traj_df)
            if not is_valid:
                render_error_notice(errors)
        except Exception as e:
            render_error_notice([f"Error parsing trajectory: {str(e)}"])
    else:
        render_empty_state(
            title="RUL Forecaster Awaiting Model Artifact (Phase 3)",
            message="No active trajectory loaded. C-MAPSS FD001 regressor will be trained and evaluated in Phase 3.",
            next_step="Implement Phase 3 RUL regression pipeline."
        )


# ==========================================
# PAGE 5: MODEL EVALUATION
# ==========================================
elif selected_page == "Model Evaluation":
    render_header(
        title="Model Scientific Evaluation & Metrics",
        subtitle="Independent validation results, confusion matrices, and error distributions",
        task_badge="Evaluation Artifacts"
    )

    eval_tab1, eval_tab2 = st.tabs(["AI4I Failure Classifier", "NASA C-MAPSS RUL Regressor"])

    with eval_tab1:
        st.markdown("##### Task A: AI4I 2020 Binary Classifier")
        st.caption("Evaluated on stratified held-out test split.")
        render_empty_state(
            title="Classification Metrics Not Generated",
            message="Precision, recall, F1-score, PR-AUC, and confusion matrix will be computed from real model training in Phase 2.",
            next_step="Execute Phase 2 AI4I training pipeline."
        )

    with eval_tab2:
        st.markdown("##### Task B: C-MAPSS FD001 Regressor")
        st.caption("Evaluated on engine-separated validation set and official test terminal RUL.")
        render_empty_state(
            title="RUL Regression Metrics Not Generated",
            message="MAE and RMSE in operating cycles, residual plots, and degradation tracking will be computed in Phase 3.",
            next_step="Execute Phase 3 C-MAPSS training pipeline."
        )


# ==========================================
# PAGE 6: ALERTS & HISTORY
# ==========================================
elif selected_page == "Alerts & History":
    render_header(
        title="Advisory Alerts & Session History",
        subtitle="Audit log of early warning events and threshold triggers",
        task_badge="Advisory Log"
    )

    col_filter1, col_filter2 = st.columns([0.4, 0.6])
    with col_filter1:
        st.selectbox("Filter Task", ["All Tasks", "AI4I Failure Risk", "NASA C-MAPSS RUL"])
    with col_filter2:
        st.selectbox("Filter Severity", ["All Severities", "High Alert", "Caution", "Nominal"])

    if len(st.session_state.session_alerts) == 0:
        render_empty_state(
            title="No Alerts Triggered in Current Session",
            message="Advisory warnings generated by the ML pipelines will be logged in session memory and can be exported as CSV.",
            next_step="Run predictions in Phase 4 to record warning events."
        )
    else:
        st.dataframe(pd.DataFrame(st.session_state.session_alerts), use_container_width=True)

    st.download_button(
        label="Download Alert History (CSV)",
        data=pd.DataFrame(st.session_state.session_alerts).to_csv(index=False),
        file_name="indusentry_alert_history.csv",
        mime="text/csv",
        disabled=len(st.session_state.session_alerts) == 0
    )
