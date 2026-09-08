# Reranking

This chapter demonstrates how reranking can improve retrieval quality after an initial vector search.

## Architecture

```text
Question
→ EmbeddingGemma
→ ChromaDB retrieves top 10 candidate chunks
→ Cross-encoder reranker scores each question/chunk pair
→ Top 4 reranked chunks
→ LLM


Retriever/RAG:
Don't miss the relevant stuff. 10 chunks. 
→ high recall

Reranker:
From what we found, put the right stuff first. Rerank 10 chunks from retriever and only send top 4 to LLM. This transformer takes both question and chunk at the same time to compare. 
→ high precision

LLM:
Use the final evidence to answer.



                    284 chunks
                         ↓
                Chroma / embeddings
              FAST, somewhat imprecise
                         ↓
                     Top 10
                         ↓
                   Cross-encoder
               SLOWER, more precise
                         ↓
                      Top 4
                         ↓
                       LLM