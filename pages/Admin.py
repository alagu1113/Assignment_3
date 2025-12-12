# pages/4_Admin.py
import streamlit as st
import pandas as pd
import os, shutil

st.title("🔧 Admin: Data & Retrain")

UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Upload cleaned dataset (CSV)", type=["csv"])
if uploaded_file:
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved to {save_path}")
    if st.button("Preview saved dataset"):
        df = pd.read_csv(save_path)
        st.dataframe(df.head())

if st.button("Use uploaded dataset as primary"):
    # copy into your Cleaned_data (overwrite)
    src = os.path.join(UPLOAD_DIR, uploaded_file.name) if uploaded_file else None
    if src and os.path.exists(src):
        dest = os.path.join("Cleaned_data", "Feature_Engineering_Outputs", "emi_dataset_feature_engineered.csv")
        shutil.copy(src, dest)
        st.success(f"Dataset copied to {dest}. Now rerun training script to update models.")
    else:
        st.warning("No uploaded file found.")
