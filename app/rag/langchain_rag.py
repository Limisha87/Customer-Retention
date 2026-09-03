import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from huggingface_hub import InferenceClient


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing in .env file")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file")


# ============================================================
# 2. LOAD DOCUMENTS
# ============================================================

KNOWLEDGE_BASE = Path("knowledge_base")

documents = []

for file_path in KNOWLEDGE_BASE.glob("*.txt"):
    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    documents.extend(loader.load())


print(f"Documents loaded: {len(documents)}")


# ============================================================
# 3. CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# 5. CONNECT TO PINECONE
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

INDEX_NAME = "customer-retention"

index = pc.Index(INDEX_NAME)

print(f"Connected to Pinecone index: {INDEX_NAME}")


# ============================================================
# 6. CREATE RETRIEVER USING PINECONE
# ============================================================

def retrieve_documents(question, top_k=3):

    query_vector = embeddings.embed_query(question)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_documents = []

    for match in results["matches"]:

        metadata = match.get("metadata", {})

        text = metadata.get("text", "")

        if text:
            retrieved_documents.append(text)

    return retrieved_documents


# ============================================================
# 7. LANGCHAIN PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI Customer Retention Assistant.

Use the provided knowledge base context to answer the user's question.

Rules:
- Use the context as the main source of information.
- Give practical customer retention recommendations.
- Do not invent company policies or information.
- If the context does not contain enough information, clearly say that.
- Keep the answer professional and easy to understand.

Knowledge Base Context:
{context}

User Question:
{question}

Provide a useful answer.
"""
)


# ============================================================
# 8. HUGGING FACE LLM
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)

MODEL_NAME = "openai/gpt-oss-120b:fastest"


# ============================================================
# 9. USER QUESTION
# ============================================================

question = "What should we do for a customer who is likely to churn?"


print("\nUser Question:")
print(question)


# ============================================================
# 10. SEMANTIC SEARCH / RETRIEVAL
# ============================================================

retrieved_documents = retrieve_documents(
    question,
    top_k=3
)

print(
    f"\nRetrieved documents: {len(retrieved_documents)}"
)


# ============================================================
# 11. CREATE CONTEXT
# ============================================================

context = "\n\n".join(
    retrieved_documents
)

print("\n========== RETRIEVED CONTEXT ==========\n")

print(context)


# ============================================================
# 12. CREATE LANGCHAIN PROMPT
# ============================================================

formatted_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# Convert LangChain prompt to plain text
prompt_text = formatted_prompt.to_string()


print("\n========== PROMPT CREATED ==========\n")


# ============================================================
# 13. SEND PROMPT TO HUGGING FACE LLM
# ============================================================

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": prompt_text
        }
    ],
    max_tokens=500,
    temperature=0.2
)


# ============================================================
# 14. GET FINAL ANSWER
# ============================================================

answer = response.choices[0].message.content


print("\n========== FINAL RAG RESPONSE ==========\n")

print(answer)


# ============================================================
# 15. DONE
# ============================================================

print("\n========================================")
print("LangChain + RAG + Pinecone + Hugging Face")
print("Pipeline completed successfully!")
print("========================================")