# AI Customer Support RAG Assistant (Streamlit + Docker)

## Overview

This project is an end-to-end **Retrieval-Augmented Generation (RAG) system** that functions as an AI-powered customer support assistant. It enables users to ask natural language questions and receive context-aware answers grounded in a custom knowledge base (pricing, FAQs and API documentation).

The system combines **document retrieval, vector search and large language models** to generate accurate and relevant responses.

---

## Key Features

- End-to-end RAG pipeline for question answering
- Document ingestion from multiple formats (.txt, .pdf, .md)
- Text chunking with overlap for improved retrieval quality
- Vector embeddings using OpenAI embedding models
- Semantic search using Chroma vector database
- Context-aware response generation using GPT models
- Prompt engineering to enforce grounded responses
- Streamlit web interface for interactive chatbot experience
- Dockerized application for consistent deployment

---

## Architecture

1. **Document Loading**
   - Loads raw documents from a local folder
   - Supports multiple file formats

2. **Text Processing**
   - Splits documents into overlapping chunks
   - Optimised for semantic retrieval

3. **Embedding Generation**
   - Converts text chunks into vector embeddings using OpenAI

4. **Vector Storage**
   - Stores embeddings in ChromaDB for similarity search

5. **Retrieval Layer**
   - Retrieves top-k relevant chunks based on user query

6. **LLM Generation**
   - Combines retrieved context with prompt engineering
   - Generates grounded responses using GPT models

7. **Frontend**
   - Streamlit UI for real-time interaction

---

## Tech Stack

- Python
- LangChain
- OpenAI API (Embeddings + LLM)
- ChromaDB (Vector Database)
- Streamlit (Frontend)
- Docker (Containerisation)

---

## Example Use Cases

- Customer support automation
- Internal knowledge base assistant
- FAQ chatbot for SaaS products
- Documentation Q&A system

---

## Example Queries

- "What are your pricing plans?"
- "How do I reset my password?"
- "Do you offer refunds?"
- "What does the enterprise plan include?"

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/rag-customer-support-assistant.git
cd rag-customer-support-assistant
```
### 2. Create Environment Variables
Create a .env file in the root directory:
```bash
OPENAI_API_KEY=your_api_key_here
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Streamlit App
```bash
streamlit run RAG_streamlit.py
```
### Docker Setup
#### Build Image
```bash
docker build -t rag-app .
```
#### Run Container
```bash
docker run --env-file .env -p 8501:8501 rag-app
```



