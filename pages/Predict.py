# pages/1_Predict.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.mlflow_utils import load_registered_model, register_prediction_log
from dotenv import load_dotenv
import os

load_dotenv()

st.title("🔮 Predict EMI Eligibility & Max EMI")

# Load models lazily
CLASS_MODEL_URI = os.getenv("CLASSIFICATION_MODEL")
REG_MODEL_URI = os.getenv("REGRESSION_MODEL")

if st.button("Load models from MLflow"):
    try:
        with st.spinner("Loading models..."):
            class_model = load_registered_model(CLASS_MODEL_URI) if CLASS_MODEL_URI else None
            reg_model = load_registered_model(REG_MODEL_URI) if REG_MODEL_URI else None
        st.success("Models loaded.")
        st.session_state['class_model'] = class_model
        st.session_state['reg_model'] = reg_model
    except Exception as e:
        st.error(f"Error loading models: {e}")

# Input form — customize fields to match your training features
st.subheader("Enter borrower features")
col1, col2, col3 = st.columns(3)
age = col1.number_input("Age", min_value=18, max_value=100, value=30)
income = col2.number_input("Monthly Income", min_value=0.0, value=30000.0, step=500.0)
loan_amount = col3.number_input("Loan Amount", min_value=0.0, value=200000.0, step=1000.0)
tenure = st.number_input("Loan Tenure (months)", min_value=1, value=60)

# Build an input row; adapt columns to your feature list
input_dict = {"age": age, "monthly_income": income, "loan_amount": loan_amount, "loan_tenure": tenure}
input_df = pd.DataFrame([input_dict])

st.write("Input preview:")
st.dataframe(input_df)

if st.button("Predict"):
    try:
        class_model = st.session_state.get('class_model') or (load_registered_model(CLASS_MODEL_URI) if CLASS_MODEL_URI else None)
        reg_model = st.session_state.get('reg_model') or (load_registered_model(REG_MODEL_URI) if REG_MODEL_URI else None)

        if class_model is None or reg_model is None:
            st.warning("Models not loaded. Click 'Load models from MLflow' or set MODEL URIs in .env.")
        else:
            # Note: your model expects exact preprocessed features. If you used pipelines then direct predict works.
            class_pred = class_model.predict(input_df)
            reg_pred = reg_model.predict(input_df)

            st.metric("EMI Eligibility (1=eligible)", class_pred[0])
            st.metric("Predicted Max Monthly EMI (₹)", float(reg_pred[0]))

            # optional: log the prediction for monitoring
            register_prediction_log(run_name="prediction_event", model_name="EMI_Prediction", input_data=input_dict, pred={"elig": int(class_pred[0]), "max_emi": float(reg_pred[0])})
            st.success("Prediction logged.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
