import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    with open("model_up.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Patient Risk Score Prediction App")
st.write("Provide patient details and get predicted risk score.")

# ----------------------------
# Create Input Fields
# ----------------------------

def user_input_features():

    age = st.number_input("Age", min_value=0, value=30)
    comorbidity_score = st.number_input("Comorbidity Score", value=0.0)
    chronic_ratio = st.number_input("Chronic Ratio", value=0.0)
    HYPERTENSION = st.checkbox("Hypertension")
    age_cat_senior = st.checkbox("Age Category: Senior")
    is_high_comorbidity = st.checkbox("High Comorbidity (Yes)")
    DIABETES = st.checkbox("Diabetes")
    patient_engagement = st.number_input("Patient Engagement Score", value=0.0)
    visit_type_ER_weighted = st.number_input("ER Visit Weighted", value=0.0)
    is_high_risk_age = st.checkbox("High Risk Age")
    visit_type_ER = st.number_input("ER Visits", value=0.0)
    visit_type_ER_ = st.number_input("ER Visits (Flag)", value=0.0)

    msrmnt_type_subtype_SCREENING_COLORECTAL_CANCER = st.checkbox("Screening: Colorectal Cancer")
    weighted_visits = st.number_input("Weighted Visits", value=0.0)
    visit_type_ratio = st.number_input("Visit Type Ratio", value=0.0)
    visit_diag_ratio = st.number_input("Visit Diagnosis Ratio", value=0.0)
    total_visits = st.number_input("Total Visits", value=0)
    visit_count = st.number_input("Visit Count", value=0)
    total_readmissions = st.number_input("Total Readmissions", value=0)
    age_cat_geriatric = st.checkbox("Age Category: Geriatric")
    total_followups = st.number_input("Total Follow-ups", value=0)
    follow_up_ratio = st.number_input("Follow-up Ratio", value=0.0)
    visit_type_INPATIENT_weighted = st.number_input("Inpatient Visit Weighted", value=0.0)
    visit_type_INPATIENT = st.number_input("Inpatient Visit", value=0.0)
    CANCER = st.checkbox("Cancer")
    diag_Other = st.number_input("Diagnosis: Other", value=0.0)
    hot_spotter_chronic_flag_t = st.checkbox("Chronic Hotspot (True)")
    diag_Neurology_Psych = st.number_input("Diagnosis: Neurology/Psych", value=0.0)

    msrmnt_type_subtype_SCREENING_COLORECTAL_CANCER_SCREENING_BREAST_CANCER = st.checkbox(
        "Screening: Colorectal + Breast"
    )
    diag_Cardiovascular = st.number_input("Diagnosis: Cardiovascular", value=0.0)
    diag_GI = st.number_input("Diagnosis: GI", value=0.0)

    msrmnt_type_subtype_SCREENING_BREAST_CANCER_SCREENING_COLORECTAL_CANCER = st.checkbox(
        "Screening: Breast + Colorectal"
    )
    diag_Skin_Soft_Tissue = st.number_input("Diagnosis: Skin/Soft Tissue", value=0.0)
    hot_spotter_chronic_flag_f = st.checkbox("Chronic Hotspot (False)")
    age_cat_adult = st.checkbox("Age Category: Adult")

    msrmnt_type_subtype_no_screening = st.checkbox("No Screening")
    age_cat_pediatric = st.checkbox("Age Category: Pediatric")

    data = [
        age, comorbidity_score, chronic_ratio, int(HYPERTENSION),
        int(age_cat_senior), int(is_high_comorbidity), int(DIABETES),
        patient_engagement, visit_type_ER_weighted, int(is_high_risk_age),
        visit_type_ER, visit_type_ER_,
        int(msrmnt_type_subtype_SCREENING_COLORECTAL_CANCER),
        weighted_visits, visit_type_ratio, visit_diag_ratio,
        total_visits, visit_count, total_readmissions,
        int(age_cat_geriatric), total_followups, follow_up_ratio,
        visit_type_INPATIENT_weighted, visit_type_INPATIENT,
        int(CANCER), diag_Other, int(hot_spotter_chronic_flag_t),
        diag_Neurology_Psych,
        int(msrmnt_type_subtype_SCREENING_COLORECTAL_CANCER_SCREENING_BREAST_CANCER),
        diag_Cardiovascular, diag_GI,
        int(msrmnt_type_subtype_SCREENING_BREAST_CANCER_SCREENING_COLORECTAL_CANCER),
        diag_Skin_Soft_Tissue, int(hot_spotter_chronic_flag_f),
        int(age_cat_adult),
        int(msrmnt_type_subtype_no_screening),
        int(age_cat_pediatric)
    ]

    return np.array(data).reshape(1, -1)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Risk Score"):
    features = user_input_features()
    prediction = model.predict(features)[0]
    st.success(f"Predicted Patient Risk Score: **{prediction}**")
