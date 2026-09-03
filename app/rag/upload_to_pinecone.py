import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")


# -----------------------------
# 1. Load documents
# -----------------------------

knowledge_base = Path("knowledge_base")

documents = []

for file_path in knowledge_base.glob("*.txt"):
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents.extend(loader.load())

print(f"Documents loaded: {len(documents)}")


# -----------------------------
# 2. Chunk documents
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# -----------------------------
# 4. Connect Pinecone
# -----------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("customer-retention")


# -----------------------------
# 5. Convert chunks to vectors
# -----------------------------

vectors = []

for i, chunk in enumerate(chunks):

    vector = embeddings.embed_query(chunk.page_content)

    vectors.append({
        "id": f"chunk-{i}",
        "values": vector,
        "metadata": {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", "unknown")
        }
    })


# -----------------------------
# 6. Upload vectors
# -----------------------------

index.upsert(vectors=vectors)

print(f"Uploaded {len(vectors)} vectors to Pinecone")

print("RAG knowledge base successfully uploaded!")