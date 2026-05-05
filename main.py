import sys
import os

# Keeps the virtual environment as the priority for imports
venv_path = os.path.join(os.getcwd(), ".mlops", "Lib", "site-packages")
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model saved during your MLOps training phase
model = joblib.load("diabetes_model.pkl")

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}

@app.post("/predict")
def predict(data: DiabetesInput):
    # Convert input data to the format the model expects
    input_data = np.array([[
        data.Pregnancies, 
        data.Glucose, 
        data.BloodPressure, 
        data.BMI, 
        data.Age
    ]])
    
    # Run inference
    prediction = model.predict(input_data)[0]
    
    return {"diabetic": bool(prediction)}