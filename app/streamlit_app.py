import streamlit as st
import joblib
import numpy as np

# Carrega o modelo
model = joblib.load("models/best_model.pkl")

st.title("Credit Card Approval Prediction")

# Inputs
car_owner = st.selectbox("Do you own a car?", ["Yes", "No"])
property_owner = st.selectbox("Do you own property?", ["Yes", "No"])
income = st.number_input("Annual Income", min_value=0)
education = st.selectbox("Education level", ["Higher", "Secondary", "Incomplete", "Lower"])

# Mapeamento
education_map = {"Higher": 0, "Secondary": 1, "Incomplete": 2, "Lower": 3}

# Pré-processamento
features = np.array([[1 if car_owner == "Yes" else 0,
                      1 if property_owner == "Yes" else 0,
                      income,
                      education_map[education]]])

st.write("Input Data:", features)

# Predição
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ Credit Approved")
    else:
        st.error("❌ Credit Rejected")
