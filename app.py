import streamlit as st
import pandas as pd
import math

st.set_page_config(
    page_title="EMI Prediction App",
    layout="wide"
)

# -----------------------------
# App Title
# -----------------------------
st.title("EMI Eligibility & EMI Amount Prediction")
st.write("Rule-based EMI calculation (No ML models)")

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, value=50000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, value=150000)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
    tenure_months = st.number_input("Tenure (Months)", min_value=1, max_value=360, value=24)

annual_interest = st.slider(
    "Annual Interest Rate (%)",
    min_value=5.0,
    max_value=20.0,
    value=10.0,
    step=0.1
)

# -----------------------------
# EMI Formula
# -----------------------------
def calculate_emi(P, annual_rate, N):
    r = annual_rate / (12 * 100)
    emi = (P * r * (1 + r) ** N) / ((1 + r) ** N - 1)
    return emi

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict EMI Eligibility & Amount"):
    try:
        emi = calculate_emi(loan_amount, annual_interest, tenure_months)

        # Business rules
        max_affordable_emi = monthly_income * 0.40

        eligibility_conditions = [
            age <= 60,
            credit_score >= 600,
            emi <= max_affordable_emi
        ]

        eligible = all(eligibility_conditions)

        st.subheader("Prediction Result")

        if eligible:
            st.success("Customer is Eligible for EMI Loan")
            st.metric("Estimated Monthly EMI (₹)", f"{emi:,.2f}")
            st.write(f"Maximum Affordable EMI: ₹{max_affordable_emi:,.2f}")
        else:
            st.error("Customer is NOT Eligible for EMI Loan")
            st.write(f"Calculated EMI: ₹{emi:,.2f}")
            st.write(f"Affordable EMI Limit: ₹{max_affordable_emi:,.2f}")

            if age > 60:
                st.warning("Age exceeds eligibility limit")
            if credit_score < 600:
                st.warning("Low credit score")
            if emi > max_affordable_emi:
                st.warning("EMI exceeds 40% of monthly income")

    except Exception as e:
        st.error("Error calculating EMI")
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Streamlit Cloud | EMI Prediction App (No ML Models)")
