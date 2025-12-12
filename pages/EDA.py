# pages/2_EDA.py
import streamlit as st
import pandas as pd
import altair as alt
import os

st.title("📈 Data Exploration")

DATA_PATH = os.path.join("Cleaned_data", "Feature_Engineering_Outputs", "emi_dataset_feature_engineered.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.write("Dataset shape:", df.shape)
    if st.checkbox("Show raw data"):
        st.dataframe(df.sample(200))

    # Quick numeric summary
    st.subheader("Numeric summary")
    st.dataframe(df.describe().T)

    st.subheader("Feature vs Target plots")
    target = st.selectbox("Choose target column", options=[c for c in df.columns if "emi" in c.lower()])
    col = st.selectbox("Choose numeric feature", options=[c for c in df.select_dtypes(include='number').columns if c != target])
    chart = alt.Chart(df.sample(500)).mark_circle(size=60).encode(
        x=col, y=target, tooltip=[col, target]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning(f"Data not found at {DATA_PATH}. Use the Admin page to upload.")
