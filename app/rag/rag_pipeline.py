import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient


# --------------------------------
# 1. Load environment variables
# --------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing")


# --------------------------------
# 2. Embedding model
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 3. Pinecone
# --------------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("customer-retention")


# --------------------------------
# 4. User question
# --------------------------------

question = "What should we do for a customer who is likely to churn?"


# --------------------------------
# 5. Convert question to vector
# --------------------------------

query_vector = embeddings.embed_query(question)


# --------------------------------
# 6. Retrieve relevant chunks
# --------------------------------

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)


# --------------------------------
# 7. Build context
# --------------------------------

context_parts = []

for match in results["matches"]:
    text = match["metadata"].get("text", "")
    context_parts.append(text)

context = "\n\n".join(context_parts)


# --------------------------------
# 8. Create RAG prompt
# --------------------------------

prompt = f"""
You are a customer retention AI assistant.

Answer the user's question using the provided knowledge base.

Knowledge Base:
{context}

User Question:
{question}

Instructions:
- Use the knowledge base to answer.
- Do not invent policies or facts.
- Give practical retention recommendations.
- Keep the answer clear and concise.
"""


# --------------------------------
# 9. Hugging Face LLM
# --------------------------------

client = InferenceClient(
    api_key=HF_TOKEN
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b:fastest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=500
)


# --------------------------------
# 10. Final response
# --------------------------------

answer = response.choices[0].message.content

print("\n==============================")
print("USER QUESTION")
print("==============================")
print(question)

print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================")
print(context)

print("\n==============================")
print("AI RESPONSE")
print("==============================")
print(answer)