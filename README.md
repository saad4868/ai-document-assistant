# RAG Document Q&A System 

This is a Retrieval-Augmented Generation (RAG) project using LangChain, ChromaDB, and HuggingFace.

## Features
- PDF & TXT ingestion
- Recursive text splitting
- Vector embeddings
- ChromaDB storage
- Question answering using LLM

## How to run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Add documents
Put your files inside /docs folder

### 3. Run ingestion
python ingest.py

### 4. Run chatbot
python main.py
