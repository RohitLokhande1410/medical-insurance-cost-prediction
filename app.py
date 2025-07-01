# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏥 Medical Insurance Cost Predictor")

st.write("""
Enter the details below to predict annual medical insurance charges:
""")

age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ("male", "female"))
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of children", 0, 10, 0)
smoker = st.selectbox("Smoker", ("yes", "no"))
region = st.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

if st.button("Predict Insurance Charge"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Annual Insurance Charge: ${prediction:.2f}")
