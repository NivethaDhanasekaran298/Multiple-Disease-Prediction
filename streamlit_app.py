import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="centered"
)

# Title
st.title("Multiple Disease Prediction System")

# Disease selection
disease = st.selectbox(
    "Select Disease",
    ("Parkinson's Disease", "Kidney Disease", "Liver Disease")
)

# Load correct model and feature names
if disease == "Parkinson's Disease":
    model = joblib.load("models/parkinsons_model.pkl")
    feature_names = joblib.load("models/parkinsons_features.pkl")

elif disease == "Kidney Disease":
    model = joblib.load("models/kidney_model.pkl")
    feature_names = joblib.load("models/kidney_features.pkl")

else:
    model = joblib.load("models/liver_model.pkl")
    feature_names = joblib.load("models/liver_features.pkl")

# Input section
st.subheader("Enter Medical Details")

inputs = []
for feature in feature_names:
    value = st.number_input(
        label=feature,
        value=0.0,
        format="%.4f"
    )
    inputs.append(value)

input_data = np.array([inputs])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ {disease} Detected (Risk: {probability:.2f}%)")
    else:
        st.success(f"✅ No {disease} Detected (Risk: {probability:.2f}%)")
