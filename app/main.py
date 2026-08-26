from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Retention Prediction API")


# Load trained model
model = joblib.load("models/best_model.pkl")

# Load preprocessing pipeline
preprocessor = joblib.load("models/preprocessor.pkl")


# Request schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Retention Prediction API is running"
    }


@app.post("/predict")
def predict(data: CustomerData):

    # Convert request data to DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        result = "Likely to Churn"
    else:
        result = "Likely to Stay"

    return {
        "prediction": result
    }