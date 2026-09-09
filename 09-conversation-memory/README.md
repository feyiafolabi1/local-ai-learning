# Conversation Memory and Context-Aware Retrieval

This chapter extends the RAG system so it can understand conversational follow-up questions.

The core idea is that many user questions are not fully understandable on their own.

For example:

```text
User:
What is Kimi Delta Attention?

Assistant:
[answer]

User:
Why did they use it instead of regular attention?


So we store the users first question + answer from the LLM in a list and then when the next question is asked we show the stored list to the LLM and then it does a rewrite of the new question/prompt with that context. So the rewritten query will look something like:

Rewritten prompt: Why did they use Kimi Delta Attention instead of regular attention. 

This rewritten query is what then gets embedded and compared to the pdf chunks (RAG) and then just follow the flow chart below. We limited this script to only show the last 6 question and answers to the LLM instead of the whole list for the rewrite part but you can adjust as needed.


                  Conversation history
                          +
                    latest question
                          ↓
               context-aware rewriting
                          ↓
                standalone search query
                          ↓
                   EmbeddingGemma
                          ↓
                       Chroma
                          ↓
                 top relevant chunks
                          ↓
                         LLM
                          ↓
                 grounded response
                          ↓
            store user question + answer
                          ↓
                available next turn