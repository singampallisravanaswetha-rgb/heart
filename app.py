import streamlit as st
import numpy as np
import pickle

# Load trained model using pickle
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("❤️ Heart Disease Prediction")

st.write("Enter patient details:")

# Inputs
age = st.slider("Age", 20, 80, 40)
bp = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
thalach = st.slider("Max Heart Rate", 60, 220, 150)

# Predict
if st.button("Predict"):
    input_data = np.array([[age, bp, chol, thalach]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Result")
    
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.subheader("Prediction Probability")
    st.write(probability)