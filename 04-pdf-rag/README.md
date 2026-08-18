# PDF RAG

This project extends the basic RAG system to support PDF documents.

## What it does

1. Finds PDF files in the `documents` folder.
2. Extracts text page by page using `pypdf`.
3. Cleans the extracted text.
4. Splits each page into overlapping chunks.
5. Creates embeddings using EmbeddingGemma through Ollama.
6. Stores the text, metadata, and vectors in `pdf_index.json`.
7. Embeds user questions and retrieves relevant chunks using cosine similarity.
8. Sends the retrieved text to a local chat model for grounded answers with citations.

## Chunking configuration

- Chunk size: 180 words
- Chunk overlap: 30 words
- Chunks remain associated with their original PDF page for citation.

## Models

- Embedding model: `embeddinggemma`
- Chat model: `ministral-3:3b`

## Setup

Install the required Python packages:

```bash
python3 -m pip install ollama numpy pypdf