import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")


# -----------------------------
# 1. Embedding model
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 2. Connect to Pinecone
# -----------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("customer-retention")


# -----------------------------
# 3. User query
# -----------------------------

query = "What should we do for a customer who is likely to churn?"


# -----------------------------
# 4. Convert query to embedding
# -----------------------------

query_vector = embeddings.embed_query(query)


# -----------------------------
# 5. Semantic search
# -----------------------------

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)


# -----------------------------
# 6. Display results
# -----------------------------

print("\nQuery:")
print(query)

print("\nRelevant results:\n")

for i, match in enumerate(results["matches"], start=1):

    print(f"--- Result {i} ---")
    print(f"Score: {match['score']}")
    print(f"Source: {match['metadata'].get('source')}")
    print(f"Text:\n{match['metadata'].get('text')}")
    print()