import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="EMI Eligibility & EMI Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🔮 EMI Eligibility & EMI Amount Prediction")
st.caption("Prediction based on 4 input features")

# -----------------------------
# Load Models Safely
# -----------------------------
@st.cache_resource
def load_models():
    if not os.path.exists("emi_classifier_4features.pkl"):
        st.error("❌ Classifier model file not found.")
        st.stop()
    if not os.path.exists("emi_regression_4features.pkl"):
        st.error("❌ Regression model file not found.")
        st.stop()

    clf_model = joblib.load("emi_classifier_4features.pkl")
    reg_model = joblib.load("emi_regression_4features.pkl")
    return clf_model, reg_model

clf, reg = load_models()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📥 Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=100, step=1)
income = st.number_input(
    "Monthly Salary (₹)", min_value=1000.0, step=1000.0, format="%.2f"
)
loan_amount = st.number_input(
    "Requested Loan Amount (₹)", min_value=1000.0, step=5000.0, format="%.2f"
)
tenure = st.number_input(
    "Requested Loan Tenure (Months)", min_value=1, max_value=360, step=1
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict"):
    try:
        input_df = pd.DataFrame([{
            "age": age,
            "monthly_salary": income,
            "requested_amount": loan_amount,
            "requested_tenure": tenure
        }])

        # Align feature order with training
        try:
            input_df = input_df[clf.feature_names_in_]
        except AttributeError:
            pass

        # Predictions
        prob = clf.predict_proba(input_df)[0][1]
        emi_pred = float(reg.predict(input_df)[0])

        # Business rule
        max_affordable_emi = income * 0.40
        eligible = prob >= 0.55 and emi_pred <= max_affordable_emi
        eligible_label = "Eligible" if eligible else "Not Eligible"

        # -----------------------------
        # Display Result
        # -----------------------------
        st.subheader("📊 Prediction Result")

        if eligible:
            st.success(f"✔ Loan Eligibility: {eligible_label}")
            st.metric("Recommended EMI (₹ / month)", f"{emi_pred:,.2f}")
        else:
            st.error(f"❌ Loan Eligibility: {eligible_label}")
            st.write(
                f"Required EMI: ₹{emi_pred:,.2f}  |  "
                f"Affordable EMI Limit: ₹{max_affordable_emi:,.2f}"
            )

    except Exception as e:
        st.error("⚠️ Prediction failed. Please check input values.")
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🚀 Streamlit Cloud | EMI Prediction App (No MLflow)")
