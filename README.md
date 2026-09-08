# AI Customer Retention Assistant

An AI-powered customer retention assistant built using Python and Django.

This project uses **RAG (Retrieval-Augmented Generation)** to retrieve relevant customer retention information from Pinecone and generate useful responses using a Hugging Face LLM.

## Tech Stack

- Python
- Django
- RAG
- LangChain
- Hugging Face
- Pinecone

## How It Works

1. User sends a question through the Django API.
2. The question is converted into an embedding.
3. Pinecone searches for relevant information.
4. Relevant context is retrieved.
5. LangChain prepares the prompt.
6. Hugging Face LLM generates the final answer.
7. Django returns the response as JSON.

## API

### Ask Question

**POST**
```text
/api/ask/
