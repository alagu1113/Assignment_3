import streamlit as st
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

# ---------- MLflow Setup ----------
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # change if using remote mlflow server
mlflow.set_experiment("EMI_Prediction_Experiment_01")

# ---------- Load Models ----------
clf = joblib.load("emi_classifier_4features.pkl")
reg = joblib.load("emi_regression_4features.pkl")

st.title("🔮 EMI Eligibility & EMI Amount Prediction (4 Features) + MLflow Tracking")

# ---------- User Inputs ----------
age = st.number_input("Age", min_value=18, max_value=100, step=1)
income = st.number_input("Monthly Salary (₹)", min_value=1000.0, step=1000.0, format="%.2f")
loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=1000.0, step=5000.0, format="%.2f")
tenure = st.number_input("Requested Loan Tenure (Months)", min_value=1, max_value=360, step=1)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "monthly_salary": income,
        "requested_amount": loan_amount,
        "requested_tenure": tenure
    }])

    # Align with training feature order
    try:
        input_df = input_df[clf.feature_names_in_]
    except:
        pass

    # Predict probability & EMI
    prob = clf.predict_proba(input_df)[0][1]
    emi_pred = float(reg.predict(input_df)[0])

    # Business rule
    max_affordable_emi = income * 0.40
    eligible = prob >= 0.55 and emi_pred <= max_affordable_emi
    eligible_label = "Eligible" if eligible else "Not Eligible"

    # ---------- Display result to user ----------
    st.subheader("📌 Prediction Result")
    if eligible:
        st.success(f"✔ Loan Eligibility: {eligible_label}")
        st.write(f"💰 EMI Recommended: ₹{round(emi_pred, 2)} / month")
    else:
        st.error(f"❌ Loan Eligibility: {eligible_label}")
        st.write(
            f"Required EMI: ₹{round(emi_pred, 2)} vs Affordable EMI Limit: ₹{round(max_affordable_emi, 2)}"
        )

    # ---------- MLflow Logging ----------
    with mlflow.start_run():
        mlflow.log_params({
            "age": age,
            "monthly_salary": income,
            "requested_amount": loan_amount,
            "requested_tenure": tenure
        })

        mlflow.log_metric("probability_eligible", prob)
        mlflow.log_metric("emi_predicted", emi_pred)
        mlflow.log_metric("max_affordable_emi", max_affordable_emi)
        mlflow.log_metric("eligible_flag", 1 if eligible else 0)

        mlflow.set_tag("eligibility_status", eligible_label)
        mlflow.set_tag("prediction_source", "Streamlit UI")

    st.success("🟢 Prediction logged to MLflow successfully!")
