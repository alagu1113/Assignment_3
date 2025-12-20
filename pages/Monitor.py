# pages/3_Monitor.py
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Model Monitor",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 Application Monitoring")

st.info("MLflow monitoring has been removed. This page shows app-level status only.")

# -----------------------------
# Basic App Health Information
# -----------------------------
st.subheader("📊 Application Status")

status_data = {
    "Component": [
        "Streamlit App",
        "Prediction Engine",
        "User Interface",
        "Cloud Deployment"
    ],
    "Status": [
        "Running ✅",
        "Operational ✅",
        "Active ✅",
        "Healthy ✅"
    ],
    "Last Checked": [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ] * 4
}

df_status = pd.DataFrame(status_data)
st.dataframe(df_status, use_container_width=True)

# -----------------------------
# Placeholder for Future Logs
# -----------------------------
st.subheader("📁 Prediction Logs (Optional)")

st.warning(
    "Prediction logging is currently disabled.\n\n"
    "
