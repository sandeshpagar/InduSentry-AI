"""InduSentry AI - Industrial Machine Failure Early Warning System.

Streamlit multi-page dashboard integrated with saved scikit-learn models and early warning engine.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import (
    validate_ai4i_record,
    validate_cmapss_trajectory,
    AI4I_APPROVED_FEATURES,
    CMAPSS_SETTINGS,
    CMAPSS_SENSORS
)
from src.prediction import predict_failure_risk, DEFAULT_MODEL_PATH as CLF_MODEL_PATH
from src.rul_prediction import predict_remaining_useful_life, DEFAULT_MODEL_PATH as RUL_MODEL_PATH
from src.alerts import evaluate_classifier_alert, evaluate_rul_alert, format_alert_log_entry
from app.components.ui_blocks import (
    apply_theme_styling,
    render_header,
    render_status_chip,
    render_empty_state,
    render_error_notice,
    render_prediction_badge
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

# 3. Model Readiness Inspection
clf_ready = CLF_MODEL_PATH.exists()
rul_ready = RUL_MODEL_PATH.exists()

# 4. Session State Initialization
if "session_records_count" not in st.session_state:
    st.session_state.session_records_count = 0
if "session_alerts" not in st.session_state:
    st.session_state.session_alerts = []

# 5. Sidebar Navigation
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
    render_status_chip("Classifier", "Ready" if clf_ready else "Not trained", "success" if clf_ready else "neutral")
    render_status_chip("RUL Regressor", "Ready" if rul_ready else "Not trained", "success" if rul_ready else "neutral")
    st.caption("Environment: Phase 4 Integrated")


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if selected_page == "Overview":
    render_header(
        title="Machine Monitoring Overview",
        subtitle="Operational telemetry, model readiness, and early warning status",
        task_badge="System Telemetry"
    )

    # Top KPI cards
    active_alerts = len([a for a in st.session_state.session_alerts if a["alert_level"] in ("HIGH_ALERT", "CAUTION")])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="AI4I Classifier", value="Ready" if clf_ready else "Not trained", delta="Active" if clf_ready else None)
    with col2:
        st.metric(label="C-MAPSS RUL Regressor", value="Ready" if rul_ready else "Not trained", delta="Active" if rul_ready else None)
    with col3:
        st.metric(label="Records Analyzed", value=str(st.session_state.session_records_count))
    with col4:
        st.metric(label="Active Warnings", value=str(active_alerts), delta=f"{active_alerts} events" if active_alerts > 0 else None, delta_color="inverse")

    st.markdown("#### Real-time Monitoring Feed")
    if len(st.session_state.session_alerts) == 0:
        render_empty_state(
            title="No Predictions Processed in Current Session",
            message="Models are loaded and awaiting machine telemetry. Analyze records in 'Failure Risk' or 'RUL Forecast' to populate the audit stream.",
            next_step="Navigate to 'Failure Risk' to run single or batch classification."
        )
    else:
        recent_df = pd.DataFrame(st.session_state.session_alerts).tail(10)
        st.dataframe(recent_df, use_container_width=True)

    st.markdown("#### Quick Launch Workflows")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div style="background-color: #222729; border: 1px solid #384043; border-radius: 8px; padding: 18px;">
                <h4 style="margin-top: 0;">Path A: Failure Risk Classifier</h4>
                <p style="color: #AAB4B8; font-size: 0.9rem;">Predict row-level failure probability using AI4I physical variables (rotational speed, torque, temperatures, wear).</p>
                <span style="color: #38C8CC;">👉 Open 'Failure Risk' tab from the sidebar</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            """
            <div style="background-color: #222729; border: 1px solid #384043; border-radius: 8px; padding: 18px;">
                <h4 style="margin-top: 0;">Path B: C-MAPSS RUL Regressor</h4>
                <p style="color: #AAB4B8; font-size: 0.9rem;">Forecast remaining operating cycles from NASA turbofan engine run-to-failure degradation trajectories.</p>
                <span style="color: #38C8CC;">👉 Open 'RUL Forecast' tab from the sidebar</span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# PAGE 2: FAILURE RISK (AI4I CLASSIFIER)
# ==========================================
elif selected_page == "Failure Risk":
    render_header(
        title="Failure Risk Classification",
        subtitle="Row-level failure identification on AI4I operating variables",
        task_badge="Path A: AI4I 2020"
    )

    if not clf_ready:
        render_empty_state(
            title="Classifier Artifact Not Found",
            message="Model pipeline `models/ai4i_classifier_pipeline.joblib` is not on disk. Run `python src/train_classifier.py` to fit and serialize the model.",
            next_step="Execute `python src/train_classifier.py` in your terminal."
        )
    else:
        mode = st.radio("Input Mode", ["Single Operating Record", "Batch CSV Upload"], horizontal=True)

        if mode == "Single Operating Record":
            with st.form("ai4i_single_record_form"):
                st.markdown("##### Machine Operating Parameters")
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

                threshold_slider = st.slider("Advisory Risk Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
                submit_btn = st.form_submit_button("Analyze Operating Record")

            if submit_btn:
                record = {
                    "Type": machine_type,
                    "Air temperature [K]": air_temp,
                    "Process temperature [K]": process_temp,
                    "Rotational speed [rpm]": rot_speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": tool_wear
                }
                try:
                    result = predict_failure_risk(record, threshold=threshold_slider)
                    alert_eval = evaluate_classifier_alert(result["model_score"], threshold=threshold_slider)

                    # Update session state
                    st.session_state.session_records_count += 1
                    log_entry = format_alert_log_entry(
                        task="AI4I Failure Risk",
                        context="AI4I 2020",
                        metric_value=result["model_score"],
                        metric_unit="risk_score",
                        alert_level=alert_eval["alert_level"],
                        reason=alert_eval["reason"],
                        source=f"Single: {machine_type} | {rot_speed}rpm | {torque}Nm"
                    )
                    st.session_state.session_alerts.append(log_entry)

                    # Render result card
                    badge_html = render_prediction_badge(alert_eval["alert_level"])
                    st.markdown(
                        f"""
                        <div style="background-color: #222729; border: 1px solid #384043; border-radius: 8px; padding: 20px; margin-top: 15px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h3 style="margin: 0;">Prediction Outcome</h3>
                                <div>{badge_html}</div>
                            </div>
                            <div style="display: flex; gap: 24px; margin-bottom: 15px;">
                                <div><small style="color: #AAB4B8;">Model Risk Score</small><br><strong style="font-size: 1.5rem; color: #38C8CC;">{result['model_score']:.3f}</strong></div>
                                <div><small style="color: #AAB4B8;">Predicted Label</small><br><strong style="font-size: 1.5rem; color: #F3F6F7;">{'Failure (1)' if result['predicted_label'] == 1 else 'Normal (0)'}</strong></div>
                                <div><small style="color: #AAB4B8;">Configured Threshold</small><br><strong style="font-size: 1.5rem; color: #F3F6F7;">{threshold_slider:.2f}</strong></div>
                            </div>
                            <p style="color: #F3F6F7; margin-bottom: 5px;"><strong>Advisory Rationale:</strong> {alert_eval['reason']}</p>
                            <p style="color: #AAB4B8; font-size: 0.85rem; margin: 0;"><em>{result['message']}</em></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    render_error_notice([str(e)])

        else:
            uploaded_file = st.file_uploader("Upload Batch CSV conforming to AI4I schema", type=["csv"])
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.markdown(f"**Loaded Preview:** {len(batch_df)} rows")
                    st.dataframe(batch_df.head(5), use_container_width=True)

                    if st.button("Run Batch Classification"):
                        with st.spinner("Processing batch records through classifier pipeline..."):
                            predictions_df = predict_failure_risk(batch_df)
                            
                            # Log batch summary to session state
                            st.session_state.session_records_count += len(predictions_df)
                            high_count = (predictions_df["alert_level"] == "HIGH_ALERT").sum()
                            if high_count > 0:
                                log_entry = format_alert_log_entry(
                                    task="AI4I Failure Risk",
                                    context="AI4I 2020 Batch",
                                    metric_value=float(high_count),
                                    metric_unit="high_risk_records",
                                    alert_level="HIGH_ALERT",
                                    reason=f"Batch analysis identified {high_count} records exceeding threshold.",
                                    source=f"Batch Upload ({len(predictions_df)} rows)"
                                )
                                st.session_state.session_alerts.append(log_entry)

                            st.success(f"✅ Successfully processed {len(predictions_df)} records.")
                            st.dataframe(predictions_df[["Type", "Rotational speed [rpm]", "Torque [Nm]", "predicted_label", "model_score", "alert_level"]].head(20), use_container_width=True)

                            # CSV Download
                            csv_data = predictions_df.to_csv(index=False)
                            st.download_button(
                                label="Download Enriched Predictions (CSV)",
                                data=csv_data,
                                file_name="indusentry_batch_predictions.csv",
                                mime="text/csv"
                            )
                except Exception as e:
                    render_error_notice([f"Batch inference failed: {str(e)}"])


# ==========================================
# PAGE 3: SENSOR ANALYTICS
# ==========================================
elif selected_page == "Sensor Analytics":
    render_header(
        title="Operating & Sensor Analytics",
        subtitle="Physical parameter distributions and engine degradation curves",
        task_badge="Telemetry Inspection"
    )

    dataset_tab = st.radio("Analytics Target", ["AI4I Machine Operating Data", "C-MAPSS Jet Engine Degradation"], horizontal=True)

    if dataset_tab == "AI4I Machine Operating Data":
        ai4i_csv = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
        if not ai4i_csv.exists():
            render_empty_state(
                title="AI4I Raw Dataset Pending",
                message="Dataset `data/raw/ai4i2020.csv` not found. Download it via `python src/download_ai4i.py` to inspect real operating distributions.",
                next_step="Run `python src/download_ai4i.py`."
            )
        else:
            df_ai4i = pd.read_csv(ai4i_csv)
            col_x = st.selectbox("Select Operating Variable (X-axis)", ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Air temperature [K]"])
            col_y = st.selectbox("Select Operating Variable (Y-axis)", ["Torque [Nm]", "Rotational speed [rpm]", "Process temperature [K]", "Tool wear [min]"])

            fig = px.scatter(
                df_ai4i,
                x=col_x,
                y=col_y,
                color="Machine failure",
                color_discrete_map={0: "#38C8CC", 1: "#F06565"},
                labels={"Machine failure": "Failure (1/0)"},
                title=f"{col_x} vs {col_y} (Colored by Machine Failure)"
            )
            fig.update_layout(paper_bgcolor="#171A1C", plot_bgcolor="#222729", font_color="#F3F6F7")
            st.plotly_chart(fig, use_container_width=True)

    else:
        cmapss_train = PROJECT_ROOT / "data" / "raw" / "NASA_CMAPSS" / "train_FD001.txt"
        if not cmapss_train.exists():
            render_empty_state(
                title="C-MAPSS Trajectories Pending",
                message="File `data/raw/NASA_CMAPSS/train_FD001.txt` not found. Download it via `python src/download_cmapss.py` to plot real run-to-failure curves.",
                next_step="Run `python src/download_cmapss.py`."
            )
        else:
            df_cmapss = pd.read_csv(cmapss_train, sep=r"\s+", header=None, names=["unit_nr", "cycle"] + CMAPSS_SETTINGS + CMAPSS_SENSORS)
            selected_unit = st.slider("Select Engine Unit ID", min_value=1, max_value=int(df_cmapss["unit_nr"].max()), value=1)
            selected_sensor = st.selectbox("Select Monitored Sensor", ["s_2", "s_3", "s_4", "s_7", "s_11", "s_12", "s_15", "s_20", "s_21"])

            unit_data = df_cmapss[df_cmapss["unit_nr"] == selected_unit]
            fig = px.line(
                unit_data,
                x="cycle",
                y=selected_sensor,
                title=f"Engine #{selected_unit} — Sensor {selected_sensor} Trend Across Operating Cycles",
                labels={"cycle": "Operating Cycle", selected_sensor: f"Sensor Reading ({selected_sensor})"}
            )
            fig.update_traces(line_color="#38C8CC", line_width=2)
            fig.update_layout(paper_bgcolor="#171A1C", plot_bgcolor="#222729", font_color="#F3F6F7")
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 4: RUL FORECAST (NASA C-MAPSS FD001)
# ==========================================
elif selected_page == "RUL Forecast":
    render_header(
        title="Remaining Useful Life (RUL) Forecasting",
        subtitle="Trajectory-based cycle forecasting for monitored turbofan engines",
        task_badge="Path B: NASA C-MAPSS FD001"
    )

    if not rul_ready:
        render_empty_state(
            title="RUL Regressor Artifact Not Found",
            message="Model pipeline `models/cmapss_rul_pipeline.joblib` is not on disk. Run `python src/train_rul.py` to fit and serialize the regressor.",
            next_step="Execute `python src/train_rul.py` in your terminal."
        )
    else:
        st.markdown("##### Monitored Engine Trajectory Input")
        traj_source = st.radio("Trajectory Source", ["Demo Test Engine Example", "Upload Space-Separated / CSV Trajectory"], horizontal=True)

        traj_df = None
        if traj_source == "Demo Test Engine Example":
            test_file = PROJECT_ROOT / "data" / "raw" / "NASA_CMAPSS" / "test_FD001.txt"
            if test_file.exists():
                all_test = pd.read_csv(test_file, sep=r"\s+", header=None, names=["unit_nr", "cycle"] + CMAPSS_SETTINGS + CMAPSS_SENSORS)
                demo_unit = st.selectbox("Select Test Engine Unit", list(range(1, 11)))
                traj_df = all_test[all_test["unit_nr"] == demo_unit].copy()
                st.caption(f"Loaded Demo Trajectory: Engine #{demo_unit} with {len(traj_df)} recorded operating cycles.")
            else:
                st.info("Test dataset not found locally. Upload a trajectory file below or run download script.")

        else:
            uploaded_traj = st.file_uploader("Upload engine trajectory", type=["txt", "csv"])
            if uploaded_traj is not None:
                try:
                    traj_df = pd.read_csv(uploaded_traj, sep=r"\s+", header=None, names=["unit_nr", "cycle"] + CMAPSS_SETTINGS + CMAPSS_SENSORS)
                except Exception as e:
                    render_error_notice([f"Error parsing trajectory file: {str(e)}"])

        if traj_df is not None:
            threshold_cycles = st.slider("Advisory Maintenance Threshold (Cycles)", min_value=10, max_value=80, value=30, step=5)
            
            if st.button("Forecast Remaining Useful Life"):
                try:
                    rul_result = predict_remaining_useful_life(traj_df, warning_threshold_cycles=threshold_cycles)
                    alert_eval = evaluate_rul_alert(rul_result["predicted_remaining_cycles"], threshold_cycles=threshold_cycles)

                    # Update session state
                    st.session_state.session_records_count += 1
                    log_entry = format_alert_log_entry(
                        task="C-MAPSS RUL",
                        context="NASA C-MAPSS FD001",
                        metric_value=rul_result["predicted_remaining_cycles"],
                        metric_unit="operating_cycles",
                        alert_level=alert_eval["alert_level"],
                        reason=alert_eval["reason"],
                        source=f"Engine #{int(traj_df['unit_nr'].iloc[0])} ({len(traj_df)} cycles)"
                    )
                    st.session_state.session_alerts.append(log_entry)

                    # Render outcome card
                    badge_html = render_prediction_badge(alert_eval["alert_level"])
                    st.markdown(
                        f"""
                        <div style="background-color: #222729; border: 1px solid #384043; border-radius: 8px; padding: 20px; margin-top: 15px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h3 style="margin: 0;">RUL Forecast Outcome</h3>
                                <div>{badge_html}</div>
                            </div>
                            <div style="display: flex; gap: 24px; margin-bottom: 15px;">
                                <div><small style="color: #AAB4B8;">Estimated Remaining Life</small><br><strong style="font-size: 1.5rem; color: #38C8CC;">{rul_result['predicted_remaining_cycles']:.1f} Cycles</strong></div>
                                <div><small style="color: #AAB4B8;">Observed Cycles So Far</small><br><strong style="font-size: 1.5rem; color: #F3F6F7;">{int(traj_df['cycle'].max())} Cycles</strong></div>
                                <div><small style="color: #AAB4B8;">Warning Threshold</small><br><strong style="font-size: 1.5rem; color: #F3F6F7;">{threshold_cycles} Cycles</strong></div>
                            </div>
                            <p style="color: #F3F6F7; margin-bottom: 5px;"><strong>Advisory Rationale:</strong> {alert_eval['reason']}</p>
                            <p style="color: #AAB4B8; font-size: 0.85rem; margin: 0;"><em>{rul_result['message']}</em></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Degradation Trend Plot
                    fig = px.line(traj_df, x="cycle", y="s_11", title="Observed Static Pressure Degradation (s_11) over Monitored Cycles")
                    fig.update_traces(line_color="#38C8CC", line_width=2)
                    fig.update_layout(paper_bgcolor="#171A1C", plot_bgcolor="#222729", font_color="#F3F6F7")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    render_error_notice([f"RUL forecast failed: {str(e)}"])


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
        metrics_file = PROJECT_ROOT / "reports" / "metrics" / "ai4i_classification_metrics.json"
        if not metrics_file.exists():
            render_empty_state(
                title="Classification Metrics Not Generated",
                message="Report file `reports/metrics/ai4i_classification_metrics.json` is missing. Run `python src/train_classifier.py` to compute real metrics.",
                next_step="Run `python src/train_classifier.py`."
            )
        else:
            with open(metrics_file, "r", encoding="utf-8") as f:
                clf_metrics = json.load(f)

            tuned = clf_metrics.get("tuned_threshold", {})
            st.markdown(f"##### Selected Model: **{clf_metrics.get('model_name', 'RandomForest')}** (Tuned Threshold: **{clf_metrics.get('selected_threshold', 0.5):.3f}**)")
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("PR-AUC (Avg Precision)", f"{clf_metrics.get('test_pr_auc', 0.0):.4f}")
            with m2:
                st.metric("Test F1-Score", f"{tuned.get('f1', 0.0):.4f}")
            with m3:
                st.metric("Test Recall", f"{tuned.get('recall', 0.0):.4f}")
            with m4:
                st.metric("Test Precision", f"{tuned.get('precision', 0.0):.4f}")

            st.markdown("##### Confusion Matrix (Test Split)")
            cm = tuned.get("confusion_matrix", [[0, 0], [0, 0]])
            cm_df = pd.DataFrame(cm, index=["Actual Normal (0)", "Actual Failure (1)"], columns=["Pred Normal (0)", "Pred Failure (1)"])
            st.table(cm_df)

    with eval_tab2:
        rul_metrics_file = PROJECT_ROOT / "reports" / "metrics" / "cmapss_rul_metrics.json"
        if not rul_metrics_file.exists():
            render_empty_state(
                title="RUL Regression Metrics Not Generated",
                message="Report file `reports/metrics/cmapss_rul_metrics.json` is missing. Run `python src/train_rul.py` to evaluate error metrics.",
                next_step="Run `python src/train_rul.py`."
            )
        else:
            with open(rul_metrics_file, "r", encoding="utf-8") as f:
                rul_metrics = json.load(f)

            val_res = rul_metrics.get("validation", {}).get("RandomForest", {})
            test_res = rul_metrics.get("test", {})

            st.markdown("##### Performance in Operating Cycles")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Val MAE (Cycles)", f"{val_res.get('val_mae_cycles', 0.0):.2f}")
            with r2:
                st.metric("Val RMSE (Cycles)", f"{val_res.get('val_rmse_cycles', 0.0):.2f}")
            with r3:
                st.metric("Test MAE (Cycles)", f"{test_res.get('test_mae_cycles', 0.0):.2f}")
            with r4:
                st.metric("Test RMSE (Cycles)", f"{test_res.get('test_rmse_cycles', 0.0):.2f}")

            st.caption("Evaluated on official test engines using ground-truth terminal RUL labels.")


# ==========================================
# PAGE 6: ALERTS & HISTORY
# ==========================================
elif selected_page == "Alerts & History":
    render_header(
        title="Advisory Alerts & Session History",
        subtitle="Audit log of early warning events and threshold triggers",
        task_badge="Advisory Log"
    )

    if len(st.session_state.session_alerts) == 0:
        render_empty_state(
            title="No Advisory Events Logged Yet",
            message="Warnings generated by the ML models during this active session will appear in this audit log.",
            next_step="Run predictions from 'Failure Risk' or 'RUL Forecast'."
        )
    else:
        alerts_df = pd.DataFrame(st.session_state.session_alerts)

        col_filter1, col_filter2 = st.columns([0.4, 0.6])
        with col_filter1:
            task_filter = st.selectbox("Filter Task", ["All Tasks"] + list(alerts_df["task"].unique()))
        with col_filter2:
            severity_filter = st.selectbox("Filter Severity", ["All Severities", "HIGH_ALERT", "CAUTION", "NOMINAL"])

        filtered_df = alerts_df.copy()
        if task_filter != "All Tasks":
            filtered_df = filtered_df[filtered_df["task"] == task_filter]
        if severity_filter != "All Severities":
            filtered_df = filtered_df[filtered_df["alert_level"] == severity_filter]

        st.dataframe(filtered_df, use_container_width=True)

        st.download_button(
            label="Download Session Alert History (CSV)",
            data=filtered_df.to_csv(index=False),
            file_name="indusentry_session_alerts.csv",
            mime="text/csv"
        )
