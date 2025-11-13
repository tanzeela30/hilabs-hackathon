import streamlit as st
import pandas as pd
import numpy as np
import pickle
import runpy, os
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title='HiLabs Risk Predictor', layout='wide')

st.title('HiLabs - Risk Score Predictor and Evaluator')
st.write("""
Upload the **5 CSV files** (patient.csv, diagnosis.csv, care.csv, visit.csv, risk.csv).
The app will run the exact preprocessing pipeline, generate predictions, 
and compute accuracy against the true risk_score column.
""")

# ---------------------------------
# File Upload Section
# ---------------------------------

st.sidebar.header("Upload Input Files")

patient_file = st.sidebar.file_uploader("patient.csv", type=["csv"])
diagnosis_file = st.sidebar.file_uploader("diagnosis.csv", type=["csv"])
care_file = st.sidebar.file_uploader("care.csv", type=["csv"])
visit_file = st.sidebar.file_uploader("visit.csv", type=["csv"])
risk_file = st.sidebar.file_uploader("risk.csv", type=["csv"])

# ---------------------------------
# RUN PIPELINE
# ---------------------------------

if st.sidebar.button("Run Prediction & Evaluation"):

    # Check all files uploaded
    files_missing = []
    if not patient_file:  files_missing.append("patient.csv")
    if not diagnosis_file: files_missing.append("diagnosis.csv")
    if not care_file: files_missing.append("care.csv")
    if not visit_file: files_missing.append("visit.csv")
    if not risk_file: files_missing.append("risk.csv")

    if files_missing:
        st.error("Missing files: " + ", ".join(files_missing))
        st.stop()

    # Save files
    st.info("Saving uploaded files...")
    open("patient.csv", "wb").write(patient_file.getvalue())
    open("diagnosis.csv", "wb").write(diagnosis_file.getvalue())
    open("care.csv", "wb").write(care_file.getvalue())
    open("visit.csv", "wb").write(visit_file.getvalue())
    open("risk.csv", "wb").write(risk_file.getvalue())

    # ---------------------------------
    # Run preprocessing pipeline
    # ---------------------------------

    st.info("Running preprocessing...")

    if not os.path.exists("cleaned_preprocessing.py"):
        st.error("cleaned_preprocessing.py not found! Please upload it to your repo.")
        st.stop()

    try:
        ns = runpy.run_path("cleaned_preprocessing.py")
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # Retrieve processed dataframe
    if "data" in ns:
        data = ns["data"]
    elif "df" in ns:
        data = ns["df"]
    else:
        st.error("No dataframe named 'data' or 'df' was returned from preprocessing.")
        st.stop()

    st.success("Preprocessing completed.")
    st.write("### Processed Data Preview")
    st.dataframe(data.head())

    # ---------------------------------
    # Load Model
    # ---------------------------------

    if not os.path.exists("model_up.pkl"):
        st.error("model_up.pkl not found in root folder.")
        st.stop()

    with open("model_up.pkl", "rb") as f:
        model = pickle.load(f)

    # ---------------------------------
    # Feature Selection
    # ---------------------------------

    cols_to_keep = ns.get("cols_to_keep", None)

    if cols_to_keep:
        missing = [c for c in cols_to_keep if c not in data.columns]
        if missing:
            st.warning(f"Missing columns skipped: {missing}")
        existing = [c for c in cols_to_keep if c in data.columns]
        X = data[existing]
    else:
        X = data.select_dtypes(include=[np.number])

    # ---------------------------------
    # Predict
    # ---------------------------------

    try:
        preds = model.predict(X)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Ensure patient_id exists
    if "patient_id" not in data.columns:
        if data.index.name == "patient_id":
            data = data.reset_index()
        else:
            st.error("patient_id missing after preprocessing. Cannot match predictions.")
            st.stop()

    predictions = pd.DataFrame({
        "patient_id": data["patient_id"],
        "predicted_risk_score": preds
    })

    # ---------------------------------
    # Load TRUE risk scores
    # ---------------------------------

    risk_true = pd.read_csv("risk.csv")

    if "patient_id" not in risk_true.columns or "risk_score" not in risk_true.columns:
        st.error("risk.csv must contain 'patient_id' and 'risk_score'.")
        st.stop()

    # Merge predicted + true
    merged = predictions.merge(risk_true, on="patient_id", how="left")

    # Compute error
    merged["abs_error"] = (merged["predicted_risk_score"] - merged["risk_score"]).abs()

    # ---------------------------------
    # Evaluation Metrics
    # ---------------------------------
    from sklearn.metrics import mean_absolute_error, mean_squared_error

# MAE
    mae = mean_absolute_error(merged["risk_score"], merged["predicted_risk_score"])

# RMSE (manual calculation for compatibility)
    mse = mean_squared_error(merged["risk_score"], merged["predicted_risk_score"])
    rmse = np.sqrt(mse)

# Correlation
    corr = merged["risk_score"].corr(merged["predicted_risk_score"])

    st.subheader("📊 Model Evaluation")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**Correlation:** {corr:.4f}")

    # ---------------------------------
    # Show Results
    # ---------------------------------

    st.subheader("Prediction vs True Risk")
    st.dataframe(merged.head(50))

    # Download button
    csv_data = merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full Evaluation CSV",
        csv_data,
        "risk_score_evaluation.csv",
        mime="text/csv"
    )
