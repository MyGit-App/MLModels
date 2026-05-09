import streamlit as st
import pandas as pd
import joblib

st.title("Predict COVID Status")

st.subheader("Enter Symptoms")

fever = st.selectbox("Fever", [0, 1])
cough = st.selectbox("Cough", [0, 1])
breath = st.selectbox("Breathlessness", [0, 1])
travel = st.selectbox("Recent Travel", [0, 1])
family = st.selectbox("Family COVID History", [0, 1])

input_df = pd.DataFrame([{
    "fever": fever,
    "cough": cough,
    "breathlessness": breath,
    "travel_history": travel,
    "family_covid_history": family
}])

if st.button("Predict using Logistic Regression"):
    model = joblib.load("logistic_model.joblib")
    pred = model.predict(input_df)[0]
    st.success(f"COVID Prediction: {'Positive' if pred==1 else 'Negative'}")