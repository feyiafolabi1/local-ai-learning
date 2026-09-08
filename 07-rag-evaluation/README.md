# RAG Evaluation

This chapter turns manual RAG testing into a repeatable evaluation workflow.

Instead of asking a few questions and visually checking whether the answers look correct, the system uses a fixed evaluation dataset and calculates retrieval metrics automatically.

## Goal

Compare two retrieval systems on the same set of questions:

```text
System A
→ ChromaDB vector retrieval only

System B
→ ChromaDB vector retrieval
→ cross-encoder reranking


## Results

| Metric | Chroma Only | Chroma + Reranker |
|---|---:|---:|
| Top-1 Accuracy | 60% | 80% |
| Top-4 Hit Rate | 80% | 80% |
| Average Rank | 2.60 | 2.20 |

### Per-Question Results

| Question | Chroma Rank | Reranked Rank |
|---|---:|---:|
| What is the purpose of the Tensor Data Mover? | 6 | 1 |
| What is scaled dot-product attention? | 1 | 1 |
| Why does the Transformer use multiple attention heads? | 4 | 7 | (Interesting that reranking made this worse.)
| What is Kimi Delta Attention? | 1 | 1 |
| How many GPUs are in the AMD Helios scale-up domain? | 1 | 1 |



Reranking improved Top-1 accuracy from 60% to 80% and reduced the average rank from 2.60 to 2.20.

The Tensor Data Mover question improved significantly, moving from rank 6 to rank 1.

However, the multi-head attention question moved from rank 4 to rank 7, showing that rerankers do not improve every query.