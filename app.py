import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

st.title("Diabeties Detetcion App")

Pregnancies = st.number_input('Pregnancies', 0, 1220000, 10)
Glucose = st.number_input('Glucose', 0, 1220000, 10)
BloodPressure = st.number_input('BloodPressure', 0, 1220000, 10)
SkinThickness = st.number_input('SkinThickness', 0, 1220000, 10)
Insulin = st.number_input('Insulin', 0, 1220000, 10)
BMI = st.number_input('BMI', 0, 1220000, 10)
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', 0, 1220000, 10)
Age = st.number_input('Age', 0, 1220000, 10)

dict_ = {
            "Pregnancies": [Pregnancies],
            "Glucose": [Glucose],
            "BloodPressure": [BloodPressure],
            "SkinThickness":[SkinThickness],
            "Insulin": [Insulin],
            "BMI": [BMI],
            "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
            "Age": [Age]
        }

results = pd.DataFrame(dict_)

with open('./datasets/trained_model.pkl', 'rb') as file:
    data = pickle.load(file)

ok = st.button("Predict")

if ok:
    if data.predict(results)[0] == 0:
        st.write("Not Diabetic")
    else:
        st.write("Diabetic")
    
  
