"""InduSentry AI - Reusable Dashboard UI Components."""

from typing import List, Optional
import streamlit as st


def apply_theme_styling():
    """Injects minimal CSS to ensure adherence to design.md dark graphite palette."""
    st.markdown(
        """
        <style>
        /* Base background and text colors */
        .stApp {
            background-color: #171A1C;
            color: #F3F6F7;
        }
        [data-testid="stSidebar"] {
            background-color: #101315 !important;
            border-right: 1px solid #384043;
        }
        
        /* Metric card styling */
        div[data-testid="metric-container"] {
            background-color: #222729;
            border: 1px solid #384043;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        /* Accent styling for headers */
        h1, h2, h3 {
            color: #F3F6F7;
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #292F31;
            color: #F3F6F7;
            border: 1px solid #38C8CC;
            border-radius: 6px;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #38C8CC;
            color: #101315;
            border-color: #38C8CC;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header(title: str, subtitle: str, task_badge: Optional[str] = None):
    """Renders a clean top header with title, subtitle, and an optional task tag."""
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(f"## {title}")
        st.markdown(f"<p style='color: #AAB4B8; margin-top: -12px;'>{subtitle}</p>", unsafe_allow_html=True)
    with col2:
        if task_badge:
            st.markdown(
                f"<div style='text-align: right;'><span style='background-color: #292F31; color: #38C8CC; "
                f"padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; border: 1px solid #38C8CC;'>"
                f"{task_badge}</span></div>",
                unsafe_allow_html=True
            )
    st.divider()


def render_status_chip(label: str, status: str, state: str = "neutral"):
    """Displays a status chip pairing color and text (accessible multi-modal design)."""
    color_map = {
        "success": ("#43C887", "🟢"),
        "caution": ("#F0B44D", "🟡"),
        "alert": ("#F06565", "🔴"),
        "neutral": ("#AAB4B8", "⚪")
    }
    color, icon = color_map.get(state, color_map["neutral"])
    st.markdown(
        f"<span style='background-color: #222729; border: 1px solid {color}; padding: 3px 8px; border-radius: 6px; color: #F3F6F7; font-size: 0.85rem;'>"
        f"{icon} <strong>{label}:</strong> {status}</span>",
        unsafe_allow_html=True
    )


def render_empty_state(title: str, message: str, next_step: str):
    """Displays an honest empty state advising the user on required prior actions."""
    st.markdown(
        f"""
        <div style="background-color: #222729; border: 1px dashed #384043; border-radius: 8px; padding: 28px; text-align: center; margin: 15px 0;">
            <h4 style="color: #F3F6F7; margin-bottom: 8px;">{title}</h4>
            <p style="color: #AAB4B8; font-size: 0.95rem; margin-bottom: 12px;">{message}</p>
            <span style="color: #38C8CC; font-size: 0.85rem; background-color: #171A1C; padding: 5px 12px; border-radius: 4px; border: 1px solid #384043;">
                👉 Next step: {next_step}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_error_notice(errors: List[str]):
    """Renders human-readable validation errors in a red-bordered dark alert box."""
    error_items = "".join([f"<li>{err}</li>" for err in errors])
    st.markdown(
        f"""
        <div style="background-color: #222729; border-left: 4px solid #F06565; border-radius: 4px; padding: 12px 18px; margin: 10px 0;">
            <strong style="color: #F06565;">⚠️ Validation Failed</strong>
            <ul style="color: #F3F6F7; margin-top: 6px; margin-bottom: 2px; padding-left: 20px;">
                {error_items}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
