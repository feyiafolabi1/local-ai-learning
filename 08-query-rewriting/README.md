# Query Rewriting for RAG

This chapter explores how rewriting a user query can improve retrieval quality in a RAG system.

The goal is not to answer the question during rewriting. The goal is to produce a cleaner search query that better represents the user's intent before embedding and retrieval.

## Architecture

```text
User question
→ rewrite model
→ rewritten search query
→ EmbeddingGemma
→ ChromaDB
→ retrieved chunks