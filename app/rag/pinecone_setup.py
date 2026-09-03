import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")


pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "customer-retention"

existing_indexes = [index["name"] for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created index: {INDEX_NAME}")
else:
    print(f"Index already exists: {INDEX_NAME}")


print("Pinecone setup completed!")