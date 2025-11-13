import streamlit as st
import pandas as pd
import numpy as np
import pickle
import runpy, os
import json

st.set_page_config(page_title='HiLabs Risk Predictor', layout='wide')

st.title('HiLabs - Batch Risk Score Predictor')
st.write("""
Upload the **5 CSV files** (patient.csv, diagnosis.csv, care.csv, risk.csv, visit.csv).
The app will run the *exact same preprocessing pipeline* from your notebook
and generate risk scores for all patients using model_up.pkl.
""")

# ----------------------------
# Sidebar File Uploads
# ----------------------------
st.sidebar.header("Upload all required files")

patient_file = st.sidebar.file_uploader("patient.csv", type=["csv"])
diagnosis_file = st.sidebar.file_uploader("diagnosis.csv", type=["csv"])
care_file = st.sidebar.file_uploader("care.csv", type=["csv"])
risk_file = st.sidebar.file_uploader("risk.csv", type=["csv"])
visit_file = st.sidebar.file_uploader("visit.csv", type=["csv"])

# ----------------------------
# Main Button
# ----------------------------
if st.sidebar.button("Run Prediction"):

    # Ensure all files uploaded
    missing = []
    if not patient_file: missing.append("patient.csv")
    if not diagnosis_file: missing.append("diagnosis.csv")
    if not care_file: missing.append("care.csv")
    if not risk_file: missing.append("risk.csv")
    if not visit_file: missing.append("visit.csv")

    if missing:
        st.error("Please upload ALL required files: " + ", ".join(missing))
        st.stop()

    # ----------------------------
    # Save uploaded files
    # ----------------------------
    st.info("Saving uploaded CSV files...")

    open("patient.csv", "wb").write(patient_file.getvalue())
    open("diagnosis.csv", "wb").write(diagnosis_file.getvalue())
    open("care.csv", "wb").write(care_file.getvalue())
    open("risk.csv", "wb").write(risk_file.getvalue())
    open("visit.csv", "wb").write(visit_file.getvalue())

    # ----------------------------
    # Load preprocessing code
    # ----------------------------
    st.info("Running preprocessing pipeline...")

    # Load the cleaned preprocessing segment generated earlier
    with open("cleaned_preprocessing.py") as f:
        preproc_code = f.read()

    # Write to file so runpy can execute it
    with open("preprocessing_exec.py", "w") as f:
        f.write(preproc_code)

    try:
        ns = runpy.run_path("preprocessing_exec.py")
    except Exception as e:
        st.error(f"❌ Error in preprocessing pipeline:\n{e}")
        st.stop()

    # ----------------------------
    # Fetch processed dataframe
    # ----------------------------
    if "data" in ns:
        data = ns["data"]
    elif "df" in ns:
        data = ns["df"]
    else:
        st.error("Preprocessing did not produce a dataframe named `data`.")
        st.stop()

    st.success("Preprocessing completed successfully.")
    st.write("### Preview of Processed Data")
    st.dataframe(data.head())

    # ----------------------------
    # Load model
    # ----------------------------
    if not os.path.exists("model_up.pkl"):
        st.error("model_up.pkl not found in the repository root!")
        st.stop()

    with open("model_up.pkl", "rb") as f:
        model = pickle.load(f)

    # Determine feature columns
    cols_to_keep = ns.get("cols_to_keep", None)

    if cols_to_keep:
      existing = [c for c in cols_to_keep if c in data.columns]
      missing = [c for c in cols_to_keep if c not in data.columns]

      

      X = data[existing]

    else:
        X = data.select_dtypes(include=[np.number])

    # ----------------------------
    # Run prediction
    # ----------------------------
    try:
        preds = model.predict(X)
    except Exception as e:
        st.error(f"Prediction failed:\n{e}")
        st.stop()

    # ----------------------------
    # Generate Output
    # ----------------------------
   # Ensure patient_id column always exists
    if "patient_id" not in data.columns:
        st.warning("patient_id missing in processed data — attempting recovery.")
        if data.index.name == "patient_id":
        data = data.reset_index()

    output = pd.DataFrame({
        "patient_id": data["patient_id"],
        "predicted_risk_score": preds
    })


    st.write("### Prediction Results")
    st.dataframe(output.head(50))

    # Download button
    csv_data = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="predicted_risk_scores.csv",
        mime="text/csv"
    )
