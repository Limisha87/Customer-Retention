from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Knowledge base folder
KNOWLEDGE_BASE = Path("knowledge_base")


# Load all TXT documents
documents = []

for file_path in KNOWLEDGE_BASE.glob("*.txt"):
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents.extend(loader.load())


print(f"Documents loaded: {len(documents)}")


# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


print(f"Total chunks created: {len(chunks)}")


# Display first few chunks
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
    