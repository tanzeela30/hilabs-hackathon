import pandas as pd
import numpy as np
patient = pd.read_csv('patient.csv')

# ------------------ CELL ------------------




diagnosis = pd.read_csv('diagnosis.csv')


# ------------------ CELL ------------------




care = pd.read_csv('care.csv')


# ------------------ CELL ------------------



risk = pd.read_csv('risk.csv')


# ------------------ CELL ------------------



visit = pd.read_csv('visit.csv')



# ------------------ CELL ------------------

df = patient.copy()

# ------------------ CELL ------------------

df['patient_id'].duplicated().sum()

# ------------------ CELL ------------------

# Ensure date columns are datetime objects
date_columns = ['visit_start_dt', 'visit_end_dt', 'follow_up_dt']
for col in date_columns:
    if col in visit.columns:
        visit[col] = pd.to_datetime(visit[col], errors='coerce')

# Find the latest date across all relevant columns
latest_date_in_visit = visit[date_columns].max().max()

print(f"The latest date in the visit data is: {latest_date_in_visit}")
# Convert 'hot_spotter_identified_at' to datetime objects
df['hot_spotter_identified_at'] = pd.to_datetime(df['hot_spotter_identified_at'], errors='coerce')

# Calculate the time difference in days since the hotspot was identified
# Use the latest_date_in_visit found previously as the reference
df['time_since_hotspot_identified'] = (latest_date_in_visit - df['hot_spotter_identified_at']).dt.days

df.drop('hot_spotter_identified_at', axis=1 ,inplace=True )

# ------------------ CELL ------------------



# ------------------ CELL ------------------

# Combine msrmnt_type and msrmnt_sub_type in the care table
care['msrmnt_type_subtype'] = care['msrmnt_type'] + '_' + care['msrmnt_sub_type']

# Group by patient_id and aggregate the combined string, handling potential NaNs
care_aggregated = care.groupby('patient_id')['msrmnt_type_subtype'].apply(lambda x: '_'.join(x.dropna().unique())).reset_index()

# Merge the aggregated care data with the df DataFrame
df = pd.merge(df, care_aggregated, on='patient_id', how='left')

# Display the updated df DataFrame

# ------------------ CELL ------------------

# Group care by patient_id and aggregate care_gap_ind
care_gap_aggregated = care.groupby('patient_id')['care_gap_ind'].apply(lambda x: '_'.join(x.dropna().unique())).reset_index()

# Merge with df
df = pd.merge(df, care_gap_aggregated, on='patient_id', how='left')

# Display the updated df DataFrame

# ------------------ CELL ------------------

# Get unique condition names from the diagnosis table
unique_conditions = diagnosis['condition_name'].unique()

# Create a new DataFrame with patient_id and a column of ones
diagnosis_binary = diagnosis[['patient_id', 'condition_name']].copy()
diagnosis_binary['has_condition'] = 1

# Pivot the table to get unique conditions as columns
diagnosis_pivot = diagnosis_binary.pivot_table(
    index='patient_id',
    columns='condition_name',
    values='has_condition',
    fill_value=0
).reset_index()

# Merge the new binary columns with the df DataFrame
df = pd.merge(df, diagnosis_pivot, on='patient_id', how='left')

# Fill NaN values (for patients not in diagnosis) with 0
for condition in unique_conditions:
    df[condition] = df[condition].fillna(0)

# ---- NEW CODE: Calculate chronic_ratio ----
df['chronic_ratio'] = df[unique_conditions].sum(axis=1) / len(unique_conditions)

# Display updated DataFrame


# ------------------ CELL ------------------

# --- Unique diagnosis values (drop NaN early)
unique_visit_diag = visit['prncpl_diag_nm'].dropna().unique()

# --- Build patient × diagnosis multi-hot (presence = 1)
visit_binary = (
    visit[['patient_id', 'prncpl_diag_nm']]
      .dropna(subset=['prncpl_diag_nm'])
      .assign(has_diag=1)
)

visit_pivot = (
    visit_binary
      .pivot_table(
          index='patient_id',
          columns='prncpl_diag_nm',
          values='has_diag',
          aggfunc='max',        # presence if ever seen
          fill_value=0
      )
      .rename_axis(None, axis=1)
)

# Optional: add a prefix to avoid collisions
visit_pivot = visit_pivot.add_prefix('dx_')

# Make it small: store as uint8 (or bool)
visit_pivot = visit_pivot.astype('uint8')

# --- Merge into your main df on patient_id
df = df.merge(visit_pivot.reset_index(), on='patient_id', how='left')

# Columns that were absent for some patients will be NaN after merge; fill them:
dx_cols = [c for c in df.columns if c.startswith('dx_')]
df[dx_cols] = df[dx_cols].fillna(0).astype('uint8')

# --- Ratios / totals based on distinct diagnoses present
df['visit_diag_ratio'] = df[dx_cols].sum(axis=1) / len(dx_cols)
df['total_visit_diagnoses'] = df[dx_cols].sum(axis=1)

# peek


# ------------------ CELL ------------------

import pandas as pd
import re

# -----------------------------
# 1) BINNING RULES
# -----------------------------
rules = [
    ('Respiratory Infection', r'infection|pneumonia|bronchitis|pharyngitis|laryngitis|tracheitis|tonsillitis|sinusitis|nasopharyngitis|flu|influenza|covid'),
    ('Lower Respiratory', r'pneumonia|bronchitis|bronchiolitis|wheezing|asthma|copd'),
    ('Upper Respiratory', r'cough|cold|throat|upper respiratory|pharyngitis|laryngitis'),
    ('ENT', r'otitis|ear|sinus|sinusitis|epistaxis|rhinitis|cerumen|tonsil|throat|pharynx|larynx'),
    ('Musculoskeletal', r'strain|sprain|fracture|contusion|myalgia|arthritis|back pain|shoulder|wrist|knee|hip|ligament|muscle|joint|tendon|osteo|sciatica'),
    ('Injury/Wound', r'injury|wound|laceration|open wound|foreign body|bite|burn|abrasion|crush|trauma|contusion|dislocation|fracture|amputation'),
    ('Skin/Soft Tissue', r'cellulitis|abscess|furuncle|erythematous|dermatitis|rash|swelling|urticaria|cyst|ulcer|bite|laceration|wound|pruritus'),
    ('GU', r'cystitis|urinary|bladder|hematuria|pyelonephritis|prostatitis|incontinence|urethritis'),
    ('GI', r'abdominal|gastro|nausea|vomiting|epigastric|colitis|diarrhea|constipation|appendicitis|gastritis|pancreatitis|hepatitis|hernia|peritonitis|cholecystitis|gallbladder|bleeding|hemorrhoids|gerd|reflux|diverticulitis'),
    ('Neurology/Psych', r'headache|migraine|dizz|giddiness|vertigo|syncope|collapse|seizure|epilepsy|paralysis|stroke|tremor|neuro|disorder|depression|anxiety|mood|insomnia|sleep'),
    ('Cardiovascular', r'chest pain|palpitations|hypertension|tachy|arrhythmia|heart failure|infarction|angina|embolism|thrombosis|atherosclerosis|hypotension|stemi|nstemi'),
    ('Obstetric/Gyne', r'pregnancy|labor|childbirth|preterm|vaginitis|menstruation|miscarriage|abortion|perineal|postpartum|uterovaginal|ovarian|endometriosis|fetal|maternal care|gestational'),
    ('Endocrine/Metabolic', r'diabetes|thyroid|metabolic|nutritional|obesity|hypoglycemia|ketoacidosis|electrolyte|hypokalemia|hyperglycemia'),
    ('Eye', r'conjunctivitis|hordeolum|chalazion|stye|blepharitis|cataract|glaucoma|corneal|iridocyclitis|retinal'),
    ('Other Infection', r'viral|bacterial|abscess|sepsis|tuberculosis|mononucleosis|infection'),
    ('Allergy/Immune', r'allergy|urticaria|anaphylaxis|angioedema|immune|contact dermatitis'),
    ('Pain', r'pain'),
    ('Other', r'fever|malaise|fatigue|unspecified|other|abnormal|screening|observation|follow-up'),
]

def bin_diagnosis(text):
    if pd.isnull(text):
        return 'Other'
    t = text.lower()
    for cat, pat in rules:
        if re.search(pat, t):
            return cat
    return 'Other'

# -----------------------------
# 2) APPLY BINNING
# -----------------------------
visit['diag_bin'] = visit['prncpl_diag_nm'].apply(bin_diagnosis)

# -----------------------------
# 3) One-hot diag bins + visit types
# -----------------------------
diag_dummies = pd.get_dummies(visit['diag_bin'], prefix='diag', dtype=int)
visit_type_dummies = pd.get_dummies(visit['visit_type'], prefix='visit_type', dtype=int)

# -----------------------------
# 4) readmission mapping
# -----------------------------
def encode_flag(value):
    if pd.isna(value):
        return 0  # or -1 if you prefer
    return int(bool(value))

visit['readmsn_ind'] = visit['readmsn_ind'].apply(encode_flag)

# -----------------------------
# 5) Combine + aggregate
# -----------------------------
df_for_agg = pd.concat([visit[['patient_id', 'readmsn_ind']], visit_type_dummies, diag_dummies], axis=1)

# dynamic aggregations
aggregations = {
    **{c: (c, 'sum') for c in visit_type_dummies.columns},
    **{c: (c, 'sum') for c in diag_dummies.columns},
    'visit_count': ('readmsn_ind', 'count'),
    'total_readmissions': ('readmsn_ind', 'sum')
}

patient_summary = df_for_agg.groupby('patient_id').agg(**aggregations).reset_index()

# convert visit type to indicator
for c in visit_type_dummies.columns:
    patient_summary[c] = (patient_summary[c] > 0).astype(int)


# ------------------ CELL ------------------

df = df.merge(patient_summary, on='patient_id', how='left')

# -----------------------------
# ✅ FIX: fill NaNs with 0
# -----------------------------
fill_cols = [c for c in patient_summary.columns if c != 'patient_id']
df[fill_cols] = df[fill_cols].fillna(0).astype(int)


# ------------------ CELL ------------------

df.head()

# ------------------ CELL ------------------

# --- ensure date columns are datetime ---
visit = visit.copy()
for col in ['visit_end_dt', 'follow_up_dt']:
    if col in visit.columns:
        visit[col] = pd.to_datetime(visit[col], errors='coerce')

# --- 1) one-hot / binary columns for visit_type (similar to diagnosis pivot) ---
unique_visit_types = visit['visit_type'].dropna().unique()

visit_type_binary = visit[['patient_id', 'visit_type']].copy()
visit_type_binary['has_visit_type'] = 1

visit_type_pivot = visit_type_binary.pivot_table(
    index='patient_id',
    columns='visit_type',
    values='has_visit_type',
    fill_value=0
).reset_index()

# If any visit_type column names collide with df, you can keep as-is or rename:
visit_type_cols = [c for c in visit_type_pivot.columns if c != 'patient_id']
# optional: add prefix
visit_type_pivot = visit_type_pivot.rename(columns={c: f"visit_type__{c}" for c in visit_type_cols})
visit_type_cols = [f"visit_type__{c}" for c in visit_type_cols]

# --- 2) total visits per patient ---
visits_count = visit.groupby('patient_id').size().reset_index(name='total_visits')

# --- 3) min / max difference in days between follow_up_dt and visit_end_dt per patient ---
if {'visit_end_dt', 'follow_up_dt'}.issubset(visit.columns):
    visit['diff_days'] = (visit['follow_up_dt'] - visit['visit_end_dt']).dt.days
else:
    visit['diff_days'] = pd.NA

diff_stats = visit.groupby('patient_id').agg(
    min_diff_days=('diff_days', 'min'),
    max_diff_days=('diff_days', 'max'),
    mean_diff_days=('diff_days', 'mean')  # optional
).reset_index()

# --- Merge everything into df ---
df = df.merge(visit_type_pivot, on='patient_id', how='left')
df = df.merge(visits_count, on='patient_id', how='left')
df = df.merge(diff_stats, on='patient_id', how='left')

# --- Fill NaNs ---
for col in visit_type_cols:
    df[col] = df[col].fillna(0)

df['total_visits'] = df['total_visits'].fillna(0).astype(int)

# --- 4) visit_type_ratio: proportion of unique visit types patient has ---
num_visit_types = len(unique_visit_types) if len(unique_visit_types) > 0 else 1
df['visit_type_ratio'] = df[visit_type_cols].sum(axis=1) / num_visit_types

# --- final: show head ---


# ------------------ CELL ------------------

df.info()

# ------------------ CELL ------------------

# --- Count readmission 't' and 'f' per patient ---
readmission_stats = (
    visit.groupby(['patient_id', 'readmsn_ind'])
    .size()
    .unstack(fill_value=0)  # creates columns 'f' and 't'
    .reset_index()
)

# Ensure both columns exist (some patients may have only 't' or only 'f')
for col in ['t', 'f']:
    if col not in readmission_stats.columns:
        readmission_stats[col] = 0

# --- Merge only counts into df ---
df = df.merge(
    readmission_stats[['patient_id', 't', 'f']],
    on='patient_id',
    how='left'
)

# Rename columns for clarity
df = df.rename(columns={
    't': 'readmission_true_count',
    'f': 'readmission_false_count'
})

# Fill missing values with 0
df[['readmission_true_count', 'readmission_false_count']] = (
    df[['readmission_true_count', 'readmission_false_count']].fillna(0)
)

# --- Display updated DataFrame ---


# ------------------ CELL ------------------

# --- Count total follow_up_date entries per patient ---
follow_up_counts = (
    visit.groupby('patient_id')['follow_up_dt']
    .count()  # counts non-null follow_up_date values
    .reset_index(name='total_followups')
)

# --- Merge this count into df --
df = df.merge(
    follow_up_counts,
    on='patient_id',
    how='left'
)

# --- Fill missing values with 0 (for patients with no follow-ups) ---
df['total_followups'] = df['total_followups'].fillna(0).astype(int)

# --- Display updated DataFrame ---


# ------------------ CELL ------------------

df['follow_up_ratio'] = df['total_followups']/df['total_visits']

# ------------------ CELL ------------------

df.columns

# ------------------ CELL ------------------

df

# ------------------ CELL ------------------

# Check for null values in each column and display columns with NaNs
nan_columns = df.columns[df.isnull().any()].tolist()
print("Columns with NaN values:")
print(nan_columns)

# ------------------ CELL ------------------

# Identify the one-hot encoded visit type columns
visit_type_cols = [col for col in df.columns if col.startswith('visit_type__')]

# Calculate the weighted visits for each visit type
for col in visit_type_cols:
    df[f'{col}_weighted'] = df[col] * df['total_visits']

# Create a single column with the sum of weighted visits for each patient
weighted_visit_cols = [f'{col}_weighted' for col in visit_type_cols]
df['weighted_visits'] = df[weighted_visit_cols].sum(axis=1)

# Display the updated DataFrame with the new weighted columns and the total weighted_visits

# ------------------ CELL ------------------

def handle_missing_values(df):
    """
    Intelligently handle missing values based on feature type and medical context
    """
    # Create flags for missing data (often meaningful in medical context)
    df['has_care_data'] = df['msrmnt_type_subtype'].notna().astype(int)
    df['has_care_gap_data'] = df['care_gap_ind'].notna().astype(int)

    # For numerical features, use median instead of 0
    numerical_cols = ['min_diff_days', 'max_diff_days', 'mean_diff_days',
                      'time_since_hotspot_identified', 'weighted_visits']
    for col in numerical_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  ✓ Filled {col} with median: {median_val:.2f}")

    # For follow_up_ratio, 0 is meaningful (no follow-up)
    if 'follow_up_ratio' in df.columns:
        df['follow_up_ratio'] = df['follow_up_ratio'].fillna(0)
        print(f"  ✓ Filled follow_up_ratio with 0")

    # Categorical - keep as is but add meaningful label
    if 'msrmnt_type_subtype' in df.columns:
        df['msrmnt_type_subtype'] = df['msrmnt_type_subtype'].fillna('no_screening')

    if 'care_gap_ind' in df.columns:
        df['care_gap_ind'] = df['care_gap_ind'].fillna('no_data')

    print("✓ Missing values handled")
    return df

def create_medical_features(df):
    """
    Create clinically meaningful features based on medical knowledge
    """
    print("Creating medical features...")

    # 1. Comorbidity score (weighted by severity)
    if all(col in df.columns for col in ['DIABETES', 'HYPERTENSION', 'CANCER']):
        df['comorbidity_score'] = (
            df['DIABETES'] * 1 +      # Moderate severity
            df['HYPERTENSION'] * 1 +  # Moderate severity
            df['CANCER'] * 3         # High severity
        )
        print("  ✓ Comorbidity score created")

    # 2. Healthcare utilization intensity
    if all(col in df.columns for col in ['visit_type_URGENTCARE', 'visit_type_ER', 'visit_type_INPATIENT']):
        df['utilization_intensity'] = (
            df['visit_type_URGENTCARE']*2 +
            df['visit_type__ER'] * 3 +       # ER visits weighted heavily
            df['visit_type__INPATIENT'] * 5  # Inpatient visits significant
        )
        print("  ✓ Utilization intensity created")

    # 3. Patient engagement score
    if all(col in df.columns for col in ['total_followups', 'follow_up_ratio']):
        df['patient_engagement'] = (
            (df['total_followups'] > 0).astype(int) +
            (df['follow_up_ratio'] > 0.5).astype(int) +
            (df['care_gap_ind'] != 'no_data').astype(int)
        )
        print("  ✓ Patient engagement score created")

    # 4. Age risk categories (standard geriatric categories)
    if 'age' in df.columns:
        df['age_risk_category'] = pd.cut(
            df['age'],
            bins=[0, 18, 45, 65, 100],
            labels=['pediatric', 'adult', 'senior', 'geriatric']
        )
        # Convert to dummy variables
        age_dummies = pd.get_dummies(df['age_risk_category'], prefix='age_cat')
        df = pd.concat([df, age_dummies], axis=1)
        df = df.drop('age_risk_category', axis=1)
        print("  ✓ Age risk categories created")

    # 5. Acute vs Chronic care ratio
    if all(col in df.columns for col in ['total_vist_actue', 'chronic_ratio']):
        df['acute_chronic_ratio'] = df['total_vist_actue'] / (df['chronic_ratio'] + 0.01)
        print("  ✓ Acute/chronic ratio created")

    # 6. Care continuity (inverse of days between visits)
    if 'mean_diff_days' in df.columns:
        df['care_continuity_score'] = np.where(
            df['mean_diff_days'] > 0,
            1 / (df['mean_diff_days'] + 1),
            0
        )
        print("  ✓ Care continuity score created")

    # 7. High-risk flags
    if 'age' in df.columns:
        df['is_high_risk_age'] = (df['age'] >= 65).astype(int)
        print("  ✓ High-risk age flag created")

    if 'comorbidity_score' in df.columns:
        df['is_high_comorbidity'] = (df['comorbidity_score'] >= 2).astype(int)
        print("  ✓ High comorbidity flag created")

    print("✓ All medical features created")
    return df


def stratify_risk(predictions, percentiles=[50, 75, 90]):
    """
    Categorize patients into actionable risk tiers
    """
    thresholds = np.percentile(predictions, percentiles)

    def assign_risk(score):
        if score < thresholds[0]:
            return 'Low Risk'
        elif score < thresholds[1]:
            return 'Moderate Risk'
        elif score < thresholds[2]:
            return 'High Risk'
        else:
            return 'Critical Risk'

    return np.array([assign_risk(s) for s in predictions])

print("✓ Utility functions defined")


# ------------------ CELL ------------------

df = handle_missing_values(df)
df = create_medical_features(df)



# ------------------ CELL ------------------



# ------------------ CELL ------------------

# Drop rows with any NaN values
df = df.dropna(axis=1)

# Check for null values in each column after dropping NaNs and display columns with NaNs
nan_columns = df.columns[df.isnull().any()].tolist()
print("Columns with NaN values after dropping rows:")
print(nan_columns)

# Display the updated DataFrame head

# ------------------ CELL ------------------

#import pandas as pd
#Merge data and risk on patient_id
#data = pd.merge(df, risk, on='patient_id', how='left')

# Display the updated data DataFrame
#display(data.head())

data = df.copy()

# ------------------ CELL ------------------



# ------------------ CELL ------------------

cols_to_keep = [
    'age', 'comorbidity_score', 'chronic_ratio', 'HYPERTENSION',
    'age_cat_senior', 'is_high_comorbidity', 'DIABETES', 'patient_engagement',
    'visit_type_ER_weighted', 'is_high_risk_age', 'visit_type_ER', 'visit_type_ER_',
    'msrmnt_type_subtype_SCREENING_COLORECTAL_CANCER', 'weighted_visits',
    'visit_type_ratio', 'visit_diag_ratio', 'total_visits', 'visit_count',
    'total_readmissions', 'age_cat_geriatric', 'total_followups', 'follow_up_ratio',
    'visit_type_INPATIENT_weighted', 'visit_type_INPATIENT', 'CANCER', 'diag_Other',
    'hot_spotter_chronic_flag_t', 'diag_Neurology/Psych',
    'msrmnt_type_subtype_SCREENING_COLORECTAL_CANCER_SCREENING_BREAST_CANCER',
    'diag_Cardiovascular', 'diag_GI',
    'msrmnt_type_subtype_SCREENING_BREAST_CANCER_SCREENING_COLORECTAL_CANCER',
    'diag_Skin/Soft Tissue', 'hot_spotter_chronic_flag_f', 'age_cat_adult',
    'msrmnt_type_subtype_no_screening', 'age_cat_pediatric'
]


# ------------------ CELL ------------------

len(cols_to_keep)

# ------------------ CELL ------------------

data = data[[c for c in cols_to_keep if c in data.columns]]


# ------------------ CELL ------------------

data.info()

# ------------------ CELL ------------------

missing_cols = [c for c in cols_to_keep if c not in data.columns]
missing_cols


# ------------------ CELL ------------------



# ------------------ CELL ------------------

data.head()

# ------------------ CELL ------------------

data.shape

# ------------------ CELL ------------------

import pickle

# ------------------ CELL ------------------

import pickle
