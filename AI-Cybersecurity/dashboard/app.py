"""
Module 10: Streamlit Dashboard
----------------------------------
SOC-style monitoring dashboard. Talks to the FastAPI backend over HTTP --
never imports the Gemini client directly, so no API key ever touches
the frontend process's exposed surface beyond the backend boundary.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Cybersecurity Threat Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .demo-banner {
        background-color: #2b2b2b; color: #f5c542; padding: 8px 16px;
        border-radius: 6px; font-weight: 600; margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def api_post(path: str, json_body: dict | None = None):
    try:
        resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=30)
        if resp.status_code >= 400:
            st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            return None
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


st.markdown('<div class="demo-banner">🧪 Demo / Public Dataset — not real-time production network monitoring.</div>',
            unsafe_allow_html=True)
st.title("🛡️ AI-Powered Cybersecurity Threat Detection Dashboard")

# ---------------- Sidebar navigation ----------------
page = st.sidebar.radio("Navigate", ["Overview", "Threat Distribution", "Alert Table", "Alert Details"])

model_status = api_get("/models/status")
if model_status:
    with st.sidebar.expander("Model Status"):
        st.write(model_status)

# ---------------- Overview ----------------
if page == "Overview":
    stats = api_get("/statistics")
    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        sev = stats.get("severity_counts", {})
        col1.metric("Total Events", stats.get("total_events", 0))
        col2.metric("Total Anomalies", stats.get("total_anomalies", 0))
        col3.metric("Low Risk", sev.get("LOW", 0))
        col4.metric("Medium Risk", sev.get("MEDIUM", 0))
        col5.metric("High/Critical", sev.get("HIGH", 0) + sev.get("CRITICAL", 0))

# ---------------- Threat Distribution ----------------
elif page == "Threat Distribution":
    stats = api_get("/statistics")
    if stats:
        c1, c2 = st.columns(2)
        with c1:
            sev = stats.get("severity_counts", {})
            if sev:
                fig = px.pie(names=list(sev.keys()), values=list(sev.values()),
                             title="Severity Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No alerts yet.")
        with c2:
            cats = stats.get("threat_category_counts", {})
            if cats:
                fig2 = px.bar(x=list(cats.keys()), y=list(cats.values()),
                              title="Threat Categories", labels={"x": "Category", "y": "Count"})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No classified threats yet (classifier may be unavailable).")

# ---------------- Alert Table ----------------
elif page == "Alert Table":
    st.subheader("Alerts")
    c1, c2, c3 = st.columns(3)
    severity_filter = c1.selectbox("Severity", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    status_filter = c2.selectbox("Status", ["All", "open", "reviewed", "closed"])
    category_filter = c3.text_input("Threat Category contains")

    params = {}
    if severity_filter != "All":
        params["severity"] = severity_filter
    if status_filter != "All":
        params["status"] = status_filter

    alerts = api_get("/alerts", params=params)
    if alerts:
        df = pd.DataFrame(alerts)
        if category_filter and "threat_category" in df.columns:
            df = df[df["threat_category"].fillna("").str.contains(category_filter, case=False)]
        display_cols = ["id", "timestamp", "source_ip", "destination_ip",
                         "threat_category", "risk_score", "severity", "status"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("No alerts found. Run analysis via POST /analyze first.")

# ---------------- Alert Details ----------------
elif page == "Alert Details":
    alert_id = st.number_input("Alert ID", min_value=1, step=1)
    if st.button("Load Alert"):
        alert = api_get(f"/alerts/{int(alert_id)}")
        if alert:
            st.session_state["current_alert"] = alert

    alert = st.session_state.get("current_alert")
    if alert:
        st.subheader(f"Alert #{alert['id']}")
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Source IP:** {alert.get('source_ip', 'N/A')}")
        c1.write(f"**Destination IP:** {alert.get('destination_ip', 'N/A')}")
        c1.write(f"**Port:** {alert.get('destination_port', 'N/A')}")
        c2.write(f"**Timestamp:** {alert.get('timestamp', 'N/A')}")
        c2.write(f"**Anomaly Score:** {alert.get('anomaly_score', 'N/A')}")
        c2.write(f"**Threat Category:** {alert.get('threat_category', 'N/A')}")
        c3.write(f"**Confidence:** {alert.get('confidence', 'N/A')}")
        c3.write(f"**Risk Score:** {alert.get('risk_score', 'N/A')}")
        c3.write(f"**Severity:** {alert.get('severity', 'N/A')}")

        st.divider()
        st.subheader("AI Explanation")
        if st.button("Generate AI Explanation"):
            explanation = api_post(f"/alerts/{alert['id']}/explain")
            if explanation:
                st.session_state["current_explanation"] = explanation

        explanation = st.session_state.get("current_explanation")
        if explanation:
            st.write("**Summary**")
            st.write(explanation.get("summary", "N/A"))
            st.write("**Why Suspicious**")
            for item in explanation.get("why_suspicious", []):
                st.write(f"- {item}")
            st.write("**Possible Category**")
            st.write(explanation.get("possible_category", "N/A"))
            st.write("**Severity Explanation**")
            st.write(explanation.get("severity_explanation", "N/A"))
            st.write("**Recommended Investigation**")
            for item in explanation.get("recommended_investigation", []):
                st.write(f"- {item}")
            st.caption(explanation.get("confidence_note", ""))
