import streamlit as st
import joblib
import requests
import pandas as pd
#import mlflow
import os
from io import BytesIO

st.set_page_config(page_title="EMI Prediction App", layout="wide")

# -----------------------------
# GitHub RAW Model URLs
# -----------------------------
CLASS_MODEL_URL = "https://github.com/alagu1113/Assignment_3/blob/main/emi_classifier_small.pkl"
REG_MODEL_URL   = "https://github.com/alagu1113/Assignment_3/blob/main/emi_regression_4features.pkl"

# -----------------------------
# Function to load model from GitHub
# -----------------------------
@st.cache_resource
def load_model_from_github(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Failed to download model from GitHub: " + str(url))
        return None

    model_bytes = BytesIO(response.content)
    model = joblib.load(model_bytes)
    return model

# -----------------------------
# Load models
# -----------------------------
st.sidebar.title("Model Loader")

if st.sidebar.button("Load Models"):
    st.session_state["class_model"] = load_model_from_github(CLASS_MODEL_URL)
    st.session_state["reg_model"] = load_model_from_github(REG_MODEL_URL)

    if st.session_state["class_model"] and st.session_state["reg_model"]:
        st.sidebar.success("✓ Models loaded successfully!")
    else:
        st.sidebar.error("Error loading models.")

# -----------------------------
# Input form
# -----------------------------
st.title("🏦 EMI Prediction Web App (GitHub + Streamlit Cloud)")

st.write("Enter customer/input details below:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Monthly Income", 1000, 1000000, 50000)
    loan_amount = st.number_input("Loan Amount", 10000, 10000000, 150000)

with col2:
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    dependents = st.number_input("Dependents", 0, 10, 1)
    tenure = st.number_input("Tenure (Months)", 1, 360, 24)

# Convert to DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "income": [income],
    "loan_amount": [loan_amount],
    "credit_score": [credit_score],
    "dependents": [dependents],
    "tenure": [tenure]
})

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict EMI Eligibility & Amount"):

    if "class_model" not in st.session_state:
        st.error("❌ Load the models first using sidebar → Load Models")
    else:
        class_pred = st.session_state["class_model"].predict(input_data)[0]

        if class_pred == 1:
            st.success("🎉 Customer is Eligible for EMI Loan")

            reg_pred = st.session_state["reg_model"].predict(input_data)[0]
            st.info(f"💰 *Predicted EMI Amount:* **₹ {round(reg_pred, 2)}**")
        else:
            st.error("❌ Customer is NOT eligible for EMI")

    # Optional: MLflow tracking
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.start_run()

        mlflow.log_params({
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "dependents": dependents,
            "tenure": tenure
        })
        mlflow.log_metric("eligibility", int(class_pred))
        if class_pred == 1:
            mlflow.log_metric("emi_amount", float(reg_pred))

        mlflow.end_run()

    except:
        st.warning("⚠️ MLflow tracking disabled or not configured.")

