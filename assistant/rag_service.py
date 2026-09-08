import os

from dotenv import load_dotenv
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing in .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env")


# Hugging Face
client = InferenceClient(
    api_key=HF_TOKEN
)

HF_MODEL = "openai/gpt-oss-120b:fastest"


# Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

INDEX_NAME = "customer-retention"

index = pc.Index(INDEX_NAME)


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# LangChain prompt
prompt = ChatPromptTemplate.from_template(
    """
You are an AI Customer Retention Assistant.

Answer the user's question using the provided context.

If the answer is not available in the context, clearly say that
the information is not available in the knowledge base.

Keep the answer simple, practical and concise.

Context:
{context}

User Question:
{question}
"""
)


def get_relevant_context(question, top_k=3):

    # Convert question into embedding
    query_vector = embeddings.embed_query(question)

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    documents = []

    for match in results["matches"]:

        metadata = match.get("metadata", {})

        text = metadata.get("text")

        if text:
            documents.append(text)

    return documents


def generate_answer(question):

    # 1. Retrieve relevant documents
    documents = get_relevant_context(
        question,
        top_k=3
    )

    # 2. Create context
    context = "\n\n".join(documents)

    # 3. LangChain prompt
    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    prompt_text = formatted_prompt.to_string()

    # 4. Hugging Face LLM
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        max_tokens=300,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "retrieved_documents": len(documents)
    }