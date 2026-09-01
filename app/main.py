from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

# Load environment variables
load_dotenv(override=True)

app = FastAPI(title="LLM Customer Retention Assistant")


# =========================
# Hugging Face Configuration
# =========================

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing in .env file")

client = InferenceClient(
    api_key=HF_TOKEN
)

HF_MODEL = "openai/gpt-oss-120b:fastest"


# =========================
# Load ML Model
# =========================

model = joblib.load("models/best_model.pkl")

preprocessor = joblib.load("models/preprocessor.pkl")


# =========================
# Customer Data Schema
# =========================

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


# =========================
# Home API
# =========================

@app.get("/")
def home():
    return {
        "message": "LLM Customer Retention Assistant is running"
    }


# =========================
# LLM Customer Assistant
# =========================

@app.get("/ask")
def ask_llm(question: str):

    try:

        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Customer Retention Assistant. "
                        "Answer questions about customer churn, retention, "
                        "customer behavior, and business strategies clearly "
                        "and simply."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
@app.post("/predict-and-explain")
def predict_and_explain(data: CustomerData):

    # Convert customer data to DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Preprocess
    processed_data = preprocessor.transform(input_data)

    # ML prediction
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        prediction_result = "Likely to Churn"
    else:
        prediction_result = "Likely to Stay"

    # Customer information for LLM
    customer_info = data.model_dump()

    # Prompt for LLM
    prompt = f"""
You are a Customer Retention AI Assistant.

Analyze the following customer information:

{customer_info}

Machine Learning Prediction:
{prediction_result}

Provide:

1. Churn risk explanation
2. Important factors affecting the prediction
3. Customer retention recommendations
4. A short action plan for the company

Keep the answer simple, professional and practical.
"""

    # Hugging Face LLM
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:fastest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    explanation = response.choices[0].message.content

    return {
        "prediction": prediction_result,
        "ai_analysis": explanation
    }
    
@app.post("/predict-and-explain")
def predict_and_explain(data: CustomerData):

    # Convert customer data to DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Preprocess
    processed_data = preprocessor.transform(input_data)

    # ML prediction
    prediction = model.predict(processed_data)[0]

    # Churn probability
    probability = model.predict_proba(processed_data)[0][1]
    churn_probability = round(float(probability) * 100, 2)

    # Risk level
    if churn_probability >= 70:
        risk_level = "High"
    elif churn_probability >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    prediction_result = (
        "Likely to Churn"
        if prediction == 1
        else "Likely to Stay"
    )

    # Customer information
    customer_info = data.model_dump()

    # LLM prompt
    prompt = f"""
You are an AI Customer Retention Assistant.

Analyze this customer profile:

{customer_info}

Machine Learning Result:
Prediction: {prediction_result}
Churn Probability: {churn_probability}%
Risk Level: {risk_level}

IMPORTANT:
- Use only the customer information provided.
- Do not invent customer details.
- If a value is "string", treat it as unknown/missing.
- Do not claim that a factor caused the ML prediction unless the data supports it.
- Give practical and realistic retention recommendations.

Provide your response in exactly these sections:

1. Risk Explanation
2. Important Risk Factors
3. Retention Strategy
4. Recommended Actions

Keep the answer concise and professional.
"""

    # Hugging Face LLM
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:fastest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    ai_analysis = response.choices[0].message.content

    return {
        "prediction": prediction_result,
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "ai_analysis": ai_analysis
    }