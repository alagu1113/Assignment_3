import streamlit as st
import math

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="EMI Eligibility & EMI Calculator",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 EMI Eligibility & EMI Prediction")
st.caption("Rule-based EMI calculation (No ML Model)")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📥 Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=100, step=1)
monthly_income = st.number_input(
    "Monthly Salary (₹)", min_value=1000.0, step=1000.0, format="%.2f"
)
loan_amount = st.number_input(
    "Requested Loan Amount (₹)", min_value=1000.0, step=5000.0, format="%.2f"
)
tenure_months = st.number_input(
    "Loan Tenure (Months)", min_value=1, max_value=360, step=1
)
annual_interest = st.number_input(
    "Annual Interest Rate (%)", min_value=1.0, max_value=30.0, step=0.1
)

# -----------------------------
# EMI Formula Function
# -----------------------------
def calculate_emi(P, annual_rate, N):
    r = annual_rate / (12 * 100)  # monthly interest rate
    emi = (P * r * (1 + r) ** N) / ((1 + r) ** N - 1)
    return emi

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("🔮 Predict EMI & Eligibility"):
    try:
        emi = calculate_emi(loan_amount, annual_interest, tenure_months)

        # Business rules
        max_affordable_emi = monthly_income * 0.40

        age_ok = age <= 60
        income_ok = emi <= max_affordable_emi

        eligible = age_ok and income_ok
        eligibility_label = "Eligible" if eligible else "Not Eligible"

        # -----------------------------
        # Display Results
        # -----------------------------
        st.subheader("📊 Prediction Result")

        if eligible:
            st.success(f"✔ Loan Eligibility: {eligibility_label}")
            st.metric("Monthly EMI (₹)", f"{emi:,.2f}")
            st.write(f"Maximum Affordable EMI: ₹{max_affordable_emi:,.2f}")
        else:
            st.error(f"❌ Loan Eligibility: {eligibility_label}")
            st.write(f"Calculated EMI: ₹{emi:,.2f}")
            st.write(f"Affordable EMI Limit: ₹{max_affordable_emi:,.2f}")

            if not age_ok:
                st.warning("⚠️ Age exceeds eligibility criteria (≤ 60 years).")
            if not income_ok:
                st.warning("⚠️ EMI exceeds 40% of monthly income.")

    except Exception as e:
        st.error("⚠️ Unable to calculate EMI.")
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🚀 Streamlit Cloud | Rule-Based EMI Prediction App")
