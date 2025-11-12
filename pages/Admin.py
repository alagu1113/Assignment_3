import os
import shutil
import streamlit as st
import pandas as pd

# -----------------------------
# Admin Dashboard Title
# -----------------------------
st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("🔐 Admin Panel - EMI Prediction Project")

st.write("This page allows admin users to manage files, logs, and datasets for the EMI Prediction system.")

# -----------------------------
# Define source and destination
# -----------------------------
src = "mlruns"                # Example: MLflow experiment directory or data folder
dest = "backup/mlruns_backup" # Destination inside repo or app container

# -----------------------------
# Create destination if missing
# -----------------------------
os.makedirs(os.path.dirname(dest), exist_ok=True)

# -----------------------------
# File Copy Section
# -----------------------------
st.subheader("📁 File Management")

if st.button("Backup MLflow Runs Folder"):
    if not os.path.exists(src):
        st.error(f"❌ Source folder not found: `{src}`")
        st.info("Make sure the 'mlruns' folder is part of your GitHub repository or uploaded manually.")
    else:
        try:
            # Copy folder recursively (safe for Streamlit Cloud)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            st.success(f"✅ Backup completed successfully to `{dest}`.")
        except Exception as e:
            st.error("⚠️ Error during backup operation.")
            st.exception(e)

# -----------------------------
# Display File Summary
# -----------------------------
st.subheader("📊 Folder Overview")

if os.path.exists(src):
    files = []
    for root, dirs, filenames in os.walk(src):
        for f in filenames:
            files.append(os.path.join(root, f))

    if len(files) > 0:
        st.write(f"**Total Files Found:** {len(files)}")
        df_files = pd.DataFrame(files, columns=["File Path"])
        st.dataframe(df_files, use_container_width=True)
    else:
        st.warning("No files found inside the MLflow directory.")
else:
    st.warning("The MLflow tracking folder (mlruns) is not available in this environment.")

# -----------------------------
# Optional: Admin Notes
# -----------------------------
st.info("""
💡 **Tips for Streamlit Cloud:**
- Only files in your GitHub repo are accessible.
- Local paths like `C:/Users/...` will not work.
- Use relative paths (`mlruns`, `data/model.pkl`) for portability.
- If using MLflow locally, consider connecting to a remote tracking URI.
""")
