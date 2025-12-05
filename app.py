# app.py
import streamlit as st
from dotenv import load_dotenv
import os
import mlflow

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
model_uri = os.getenv("MODEL_URI")

model = mlflow.pyfunc.load_model(model_uri)

st.set_page_config(page_title="EMI Prediction App", layout="wide")

st.title("🏦 EMI Prediction Platform")

st.markdown(
    """
    This app provides:
    - Real-time predictions (classification & regression)
    - Interactive EDA and visualization
    - Model metrics & monitoring (from MLflow)
    - Admin area for uploading datasets
    """
)

st.sidebar.title("Navigation")
st.sidebar.info("Use the pages menu (top-left) to navigate different app pages.")
