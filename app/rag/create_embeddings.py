from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


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
# 4. Test embedding
# -----------------------------

text = chunks[0].page_content

vector = embeddings.embed_query(text)

print(f"Original text:\n{text}")
print(f"\nVector dimensions: {len(vector)}")
print(f"First 10 values: {vector[:10]}")