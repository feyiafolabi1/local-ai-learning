# Vector Database RAG with ChromaDB

This chapter replaces the manual JSON and NumPy vector-search system with a persistent ChromaDB vector database.

## What this project does

1. Finds PDF files in the `documents` folder.
2. Extracts text page by page using `pypdf`.
3. Cleans the extracted text.
4. Splits each page into overlapping chunks.
5. Generates embeddings using EmbeddingGemma through Ollama.
6. Stores chunk text, metadata, IDs, and embeddings in ChromaDB.
7. Embeds user questions.
8. Queries Chroma for the nearest stored vectors.
9. Sends the retrieved source text to Ministral for grounded answers.

## Architecture

```text
PDF documents
→ text extraction
→ overlapping chunks
→ EmbeddingGemma
→ ChromaDB collection

User question
→ EmbeddingGemma
→ Chroma nearest-neighbor search
→ retrieved chunk text
→ Ministral
→ grounded answer with citations