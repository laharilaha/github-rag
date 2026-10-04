# GitHub RAG

A local Retrieval-Augmented Generation (RAG) application that allows users to chat with GitHub repositories.

The application uses GitIngest to extract repository content, LlamaIndex to build the RAG pipeline, HuggingFace embeddings for semantic search, and Ollama to generate responses locally.

## Features

- Load a GitHub repository using its URL
- Extract repository content automatically
- Create a vector index for semantic retrieval
- Ask questions about the repository
- Generate answers using a local LLM
- No OpenAI API key required for local usage

## Tech Stack

- Python
- Streamlit
- GitIngest
- LlamaIndex
- HuggingFace Embeddings
- Ollama
- Qwen 2.5 Coder 7B

## Local Setup

### 1. Create a virtual environment

```bash
python -m venv .venv