import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

st.title("Diabeties Detetcion")

Pregnancies = st.number_input('Pregnancies', 0, 17, 3)
Glucose = st.number_input('Glucose', 0, 200, 120)
BloodPressure = st.number_input('BloodPressure', 0, 123, 69)
SkinThickness = st.number_input('SkinThickness', 0, 100, 20)
Insulin = st.number_input('Insulin', 0, 847, 79)
BMI = st.number_input('BMI', 0, 68 , 31)
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', 0, 3 , 1)
Age = st.number_input('Age', 21, 99, 33)

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
    
  
