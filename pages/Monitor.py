# pages/3_Monitor.py
import streamlit as st
#from utils.mlflow_utils import list_experiments, list_runs
import pandas as pd

st.title("🧾 Model Monitoring (MLflow)")

exps = list_experiments()
exp_names = [e.name for e in exps]
sel = st.selectbox("Select Experiment", options=[""] + exp_names)

if sel:
    runs = list_runs(sel)
    if not runs:
        st.info("No runs found for experiment.")
    else:
        rows = []
        for r in runs:
            metrics = r.data.metrics
            params = r.data.params
            rows.append({
                "run_id": r.info.run_id,
                "start_time": r.info.start_time,
                "status": r.info.status,
                **{f"metric_{k}": v for k,v in metrics.items()},
                **{f"param_{k}": v for k,v in params.items()}
            })
        df = pd.DataFrame(rows)
        st.dataframe(df.sort_values("start_time", ascending=False).reset_index(drop=True))
        # Plot a metric if exists
        metric_keys = [c for c in df.columns if c.startswith("metric_")]
        if metric_keys:
            sel_metric = st.selectbox("Select metric to plot", metric_keys)
            st.line_chart(df.set_index("start_time")[sel_metric])

