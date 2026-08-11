from fastapi import FastAPI
import joblib
from pathlib import Path
import pandas as pd

from backend.schemas import InsuranceInput

# create FastAPI application 
app = FastAPI(title = "Insurance Charges Prediction API")

# load model and scaler
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "gradient_boosting_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

@app.get('/')
def home():
    return {
        "Message" : "Medical Insurance Charges Prediction"
    }
    
@app.post('/predict')
def predict(data : InsuranceInput):
    
    #convert input data into dictionary(Pydantic object -> Python dictionary)
    input_data = data.model_dump()
    
    # scale age, bmi and children
    scaled_values = scaler.transform([
        [
            input_data['age'],
            input_data['bmi'],
            input_data['children']
        ]
    ])[0]
    
    age_scaled = scaled_values[0]
    bmi_scaled = scaled_values[1]
    children_scaled = scaled_values[2]
    
    
    # One Hot encoding of region field
    region = input_data['region'].lower()
    
    region_northeast = 1 if region == 'northeast' else 0
    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0
    
    # Create features in the same order as training
    features = [[
        age_scaled,
        bmi_scaled,
        children_scaled,
        input_data['is_female'],
        input_data['is_smoker'],
        region_northeast,
        region_northwest,
        region_southeast,
        region_southwest
    ]]
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return {
        'Predicted_charges' : round(float(prediction), 2)
    }
    
    
