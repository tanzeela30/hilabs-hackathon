# README

## Project Overview

This project focuses on developing a machine learning model to predict patient-level risk scores using multiple healthcare datasets. The primary objective is to engineer meaningful clinical and utilization-based features, merge them into a unified analytical table, identify the most predictive attributes, train and evaluate different machine-learning models, and ultimately produce a serialized model artifact that can be used for inference.

Five different data sources were used:
1. Patient
2. Risk
3. Diagnosis
4. Care
5. Visit

After extensive feature engineering and merging, the Gradient Boosting model performed best and was selected for the final implementation. The trained model was saved as a `.pkl` file for future inference workflows.

## Data Architecture and Processing Approach

All datasets were provided independently. Since each dataset contained a `patient_id` field, merging was performed on this key to create a consolidated patient-level table.

Summary of data processing steps:
1. Load raw input data.
2. Perform standard cleaning (type casting, duplicate handling, missing value treatment).
3. Engineer additional variables from the original attributes.
4. Sequentially merge datasets using `patient_id`.
5. Create final modeling dataset, ensuring consistent indexing and no leakage.

## Feature Engineering and Selection

New features were engineered based on domain intuition and exploratory analysis, including:
- Comorbidity or diagnosis-based scores
- Derived care-level metrics
- Visit frequency and other utilization patterns
- Code-level rollups

Correlation statistics were used to identify variables most strongly associated with risk score. Redundant or low-value attributes were removed during preprocessing.

## Modeling Approach

Several supervised learning models were evaluated. Although multiple models were tested, Gradient Boosting performed best in balancing accuracy and generalization.

Models tested (non-exhaustive):
- Linear/regularized regression variants
- Random Forest
- Gradient Boosting

Hyperparameters such as number of estimators, learning rate, and maximum depth were tuned. The final model was then serialized as a `.pkl` file.

## Model Explainability

Interpretability was emphasized through:
1. Correlation-based feature selection
2. Model-level feature importance from Gradient Boosting

This helps provide transparency into what clinical/utilization behaviors are most influential for risk estimation.

## Setup and Execution

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Open **Prediction file.ipynb**.
3. Upload the datasets one by one when prompted (patient, risk, diagnosis, care, visit).
4. Run all cells in the notebook.
5. The final output will include:
   - `patient_id` and`predicted_risk_score`
   - Model accuracy

## Output

The primary output of the project is:
- A trained Gradient Boosting model (`model.pkl`) that predicts risk scores for each `patient_id`.

