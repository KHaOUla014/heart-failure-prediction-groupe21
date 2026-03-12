import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys

# Add the correct path to find train_model.py
sys.path.append('src')
from train_model1 import train_and_select_best_model

# ─────────────────────────────────────
# PAGE DESIGN
# ─────────────────────────────────────
st.set_page_config(page_title="Heart Failure Risk Predictor", layout="centered")
st.title("🫀 Heart Failure Risk Predictor")
st.write("Enter the patient's medical data below to predict heart failure risk.")
st.divider()

# ─────────────────────────────────────
# LOAD MODEL (function method — no pkl)
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    model, X_train, X_test, y_train, y_test = train_and_select_best_model()
    return model, X_train, X_test, y_train, y_test

try:
    st.info("⏳ Training model on startup, please wait...")
    model, X_train, X_test, y_train, y_test = load_model()
    model_loaded = True
    st.success("✅ Model ready!")
except Exception as e:
    model_loaded = False
    st.error(f"Could not train model. Error: {e}")

st.divider()

# ─────────────────────────────────────
# PATIENT DATA INPUT
# ─────────────────────────────────────
st.subheader("📋 Patient Data")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 40, 95, 60)
    ejection_fraction = st.slider("Ejection Fraction (%)", 14, 80, 38)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.2)
    serum_sodium = st.slider("Serum Sodium", 100, 150, 137)
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=20, max_value=8000, value=250)
    platelets = st.number_input("Platelets", min_value=25000.0, max_value=850000.0, value=262000.0)

with col2:
    anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    time = st.slider("Follow-up Period (days)", 4, 285, 130)

st.divider()

# ─────────────────────────────────────
# BUILD PATIENT TABLE
# ─────────────────────────────────────
patient_data = pd.DataFrame([[
    age, anaemia, creatinine_phosphokinase, diabetes,
    ejection_fraction, high_blood_pressure, platelets,
    serum_creatinine, serum_sodium, sex, smoking, time
]], columns=[
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
])

# ─────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────
if st.button("PREDICT RISK", type="primary", use_container_width=True):

    if not model_loaded:
        st.error("Model not loaded. Please check the error above.")

    else:
        # Get prediction and probability
        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0][1] * 100

        st.subheader("🩺 Prediction Result")

        if prediction == 1:
            st.error(f"HIGH RISK — {probability:.1f}% probability of heart failure")
        else:
            st.success(f"LOW RISK — {probability:.1f}% probability of heart failure")

        st.progress(int(probability))
        st.divider()

        # ─────────────────────────────────────
        # SHAP EXPLANATION
        # ─────────────────────────────────────
        st.subheader("🔍 Why this prediction?")
        st.write("The waterfall plot below shows exactly how this specific patient's medical features pushed their risk higher (red) or lower (blue).")

        try:
            # Use TreeExplainer — same as evaluate_model.py
            explainer = shap.TreeExplainer(model)
            shap_values_obj = explainer(patient_data)

            fig, ax = plt.subplots(figsize=(8, 4))

            # Handle binary classification output formats
            # same logic as evaluate_model.py
            if len(shap_values_obj.shape) == 3:
                shap.plots.waterfall(shap_values_obj[0, :, 1], show=False)
            else:
                shap.plots.waterfall(shap_values_obj[0], show=False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()

        except Exception as e:
            st.warning(f"SHAP explanation unavailable. Error: {e}")

        st.divider()

        # ─────────────────────────────────────
        # PATIENT SUMMARY
        # ─────────────────────────────────────
        st.subheader("📂 Patient Summary")
        st.dataframe(patient_data)