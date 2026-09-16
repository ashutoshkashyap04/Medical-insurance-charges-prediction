import streamlit as st
import requests  # library used to send HTTP requests to server/API

st.title("Medical Insurance Charges Prediction")

st.write("Enter the details below to predict insurance charges.")

age = st.number_input(
    "Age",
    min_value= 1,
    max_value= 120,
    value= 30
)

bmi = st.number_input(
    "BMI",
    min_value= 0.0,
    max_value= 100.0,
    value= 25.0
)

children = st.number_input(
    "Number of Children",
    min_value= 0,
    max_value= 10,
    value= 0
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ['northeast', 'northwest', 'southeast', 'southwest']
)


if st.button("Predict"):
    
    data = {
        "age" : age,
        "bmi" : bmi,
        "children" : children,
        "is_female" : 1 if gender == 'Female' else 0,
        "is_smoker" : 1 if smoker == 'Yes' else 0,
        "region" : region
    }
    
    response = requests.post(
        "https://insurance-charges-api.onrender.com/predict",
        json= data 
    )
    
    if response.status_code == 200:
        result = response.json()
        
        st.success(
            f"Predicted Insurance Charges : ${result['Predicted_charges']:.2f}"
        )
        
    else:
        
        st.error("Prediction failed")
