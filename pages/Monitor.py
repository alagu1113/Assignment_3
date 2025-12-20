import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Monitor",
    layout="wide"
)

st.title("Model Monitoring")

st.write("MLflow monitoring has been removed.")
st.write("This page is kept to avoid navigation errors.")

data = {
    "Component": [
        "Streamlit App",
        "Prediction Page",
        "Monitoring Page"
    ],
    "Status": [
        "Running",
        "Available",
        "Active"
    ]
}

df = pd.DataFrame(data)
st.dataframe(df)
