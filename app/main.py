from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing in .env file")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file")


# ============================================================
# 2. FASTAPI APP
# ============================================================

app = FastAPI(
    title="LLM Customer Retention Assistant"
)


# ============================================================
# 3. HUGGING FACE CONFIGURATION
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)

HF_MODEL = "openai/gpt-oss-120b:fastest"


# ============================================================
# 4. LOAD ML MODEL
# ============================================================

model = joblib.load(
    "models/best_model.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


# ============================================================
# 5. PINECONE CONFIGURATION
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

INDEX_NAME = "customer-retention"

index = pc.Index(
    INDEX_NAME
)


# ============================================================
# 6. HUGGING FACE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 7. LANGCHAIN PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_template(
    """You are a customer retention assistant. Based on the customer information,
prediction, churn probability, risk level, and retrieved context, provide a
concise, practical retention analysis and recommendations.

Customer information:
{customer_info}

Prediction: {prediction}
Churn probability: {churn_probability}%
Risk level: {risk_level}

Retrieved context:
{context}
"""
)

# ============================================================
# 8. CUSTOMER DATA SCHEMA
# ============================================================

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


# ============================================================
# 9. HOME API
# ============================================================

@app.get("/")
def home():

    return {
        "message": "LLM Customer Retention Assistant is running"
    }


# ============================================================
# 10. BASIC LLM ASSISTANT
# ============================================================

@app.get("/ask")
def ask_llm(question: str):

    try:

        response = client.chat.completions.create(

            model=HF_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Customer Retention "
                        "Assistant. Answer questions about "
                        "customer churn, retention, customer "
                        "behavior and business strategies "
                        "clearly and simply."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            max_tokens=250
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


# ============================================================
# 11. RAG SEMANTIC SEARCH
# ============================================================

def retrieve_context(
    question: str,
    top_k: int = 3
):

    # Create embedding for user question
    query_vector = embeddings.embed_query(
        question
    )

    # Search Pinecone
    results = index.query(

        vector=query_vector,

        top_k=top_k,

        include_metadata=True
    )

    retrieved_text = []

    for match in results["matches"]:

        metadata = match.get(
            "metadata",
            {}
        )

        text = metadata.get(
            "text",
            ""
        )

        if text:
            retrieved_text.append(text)

    return retrieved_text


# ============================================================
# 12. PREDICT + RAG + LLM
# ============================================================

@app.post("/predict-and-explain")
def predict_and_explain(
    data: CustomerData
):

    try:

        # ----------------------------------------------------
        # STEP 1: CUSTOMER DATA → DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [data.model_dump()]
        )


        # ----------------------------------------------------
        # STEP 2: PREPROCESS DATA
        # ----------------------------------------------------

        processed_data = preprocessor.transform(
            input_data
        )


        # ----------------------------------------------------
        # STEP 3: ML PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            processed_data
        )[0]


        # ----------------------------------------------------
        # STEP 4: CHURN PROBABILITY
        # ----------------------------------------------------

        probability = model.predict_proba(
            processed_data
        )[0][1]

        churn_probability = round(
            float(probability) * 100,
            2
        )


        # ----------------------------------------------------
        # STEP 5: RISK LEVEL
        # ----------------------------------------------------

        if churn_probability >= 70:

            risk_level = "High"

        elif churn_probability >= 40:

            risk_level = "Medium"

        else:

            risk_level = "Low"


        # ----------------------------------------------------
        # STEP 6: PREDICTION RESULT
        # ----------------------------------------------------

        prediction_result = (

            "Likely to Churn"

            if prediction == 1

            else "Likely to Stay"
        )


        # ----------------------------------------------------
        # STEP 7: CUSTOMER INFORMATION
        # ----------------------------------------------------

        customer_info = data.model_dump()


        # ----------------------------------------------------
        # STEP 8: CREATE RAG QUESTION
        # ----------------------------------------------------

        rag_question = f"""
Customer retention strategy for this customer.

Prediction: {prediction_result}

Churn Probability: {churn_probability}%

Risk Level: {risk_level}

Customer information:
{customer_info}
"""


        # ----------------------------------------------------
        # STEP 9: PINECONE SEMANTIC SEARCH
        # ----------------------------------------------------

        retrieved_documents = retrieve_context(
            rag_question,
            top_k=3
        )


        # ----------------------------------------------------
        # STEP 10: CREATE CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            retrieved_documents
        )


        # ----------------------------------------------------
        # STEP 11: LANGCHAIN PROMPT
        # ----------------------------------------------------

        formatted_prompt = rag_prompt.invoke(

            {
                "customer_info": customer_info,

                "prediction": prediction_result,

                "churn_probability":
                    churn_probability,

                "risk_level":
                    risk_level,

                "context":
                    context
            }
        )


        # Convert LangChain prompt to string

        prompt_text = formatted_prompt.to_string()


        # ----------------------------------------------------
        # STEP 12: HUGGING FACE LLM
        # ----------------------------------------------------

        response = client.chat.completions.create(

            model=HF_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],

            max_tokens=500,

            temperature=0.2
        )


        # ----------------------------------------------------
        # STEP 13: AI RESPONSE
        # ----------------------------------------------------

        ai_analysis = (
            response
            .choices[0]
            .message
            .content
        )


        # ----------------------------------------------------
        # STEP 14: FINAL RESPONSE
        # ----------------------------------------------------

        return {

            "prediction":
                prediction_result,

            "churn_probability":
                churn_probability,

            "risk_level":
                risk_level,

            "retrieved_documents":
                len(retrieved_documents),

            "ai_analysis":
                ai_analysis
        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )