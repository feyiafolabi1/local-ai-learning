from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ollama


PROJECT_DIR = Path.home() / "local_rag"
INDEX_FILE = PROJECT_DIR / "index.json"

CHAT_MODEL = "ministral-3:3b"
TOP_K = 3

SYSTEM_PROMPT = """
You are a grounded question-answering assistant.

Answer using only the retrieved source chunks.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or specifications.
3. If the retrieved chunks do not contain enough information, say exactly:
   "The retrieved sources do not provide enough information to answer that."
4. For supported questions, always write at least one complete explanatory
   sentence before any citation.
5. Cite supporting chunks using this format:
   [filename, chunk N]
6. A citation by itself is not an answer.
7. Keep the answer concise.
8. Do not add mechanisms, terminology, or explanations that are not
   explicitly stated in the retrieved chunks.
9. Paraphrase the retrieved text closely rather than expanding it.
""".strip()


def load_index() -> dict:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Index not found: {INDEX_FILE}\n"
            "Run ingest.py first."
        )

    return json.loads(
        INDEX_FILE.read_text(encoding="utf-8")
    )


def cosine_similarity(
    query_vector: np.ndarray,
    document_vector: np.ndarray,
) -> float:
    query_norm = np.linalg.norm(query_vector)
    document_norm = np.linalg.norm(document_vector)

    if query_norm == 0 or document_norm == 0:
        return 0.0

    return float(
        np.dot(query_vector, document_vector)
        / (query_norm * document_norm)
    )


def retrieve(
    question: str,
    index_data: dict,
    top_k: int = TOP_K,
) -> list[dict]:
    embedding_model = index_data["embedding_model"]

    response = ollama.embed(
        model=embedding_model,
        input=question,
    )

    question_vector = np.array(
        response["embeddings"][0],
        dtype=np.float32,
    )

    scored_chunks = []

    for chunk in index_data["chunks"]:
        chunk_vector = np.array(
            chunk["embedding"],
            dtype=np.float32,
        )

        score = cosine_similarity(
            question_vector,
            chunk_vector,
        )

        scored_chunks.append(
            {
                "filename": chunk["filename"],
                "chunk_number": chunk["chunk_number"],
                "text": chunk["text"],
                "similarity": score,
            }
        )

    scored_chunks.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return scored_chunks[:top_k]


def build_context(chunks: list[dict]) -> str:
    sections = []

    for chunk in chunks:
        sections.append(
            f"[{chunk['filename']}, chunk {chunk['chunk_number']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(sections)


def answer_question(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    context = build_context(retrieved_chunks)

    prompt = f"""
RETRIEVED SOURCES
-----------------
{context}

QUESTION
--------
{question}
""".strip()

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 500,
        },
    )

    return response["message"]["content"] or ""


def print_retrieved_chunks(chunks: list[dict]) -> None:
    print("\nRetrieved chunks:")

    for rank, chunk in enumerate(chunks, start=1):
        print(
            f"\n{rank}. "
            f"{chunk['filename']}, "
            f"chunk {chunk['chunk_number']}"
        )
        print(
            f"   Similarity: {chunk['similarity']:.4f}"
        )
        print(f"   {chunk['text']}")


def main() -> None:
    index_data = load_index()

    print(f"RAG assistant using {CHAT_MODEL}")
    print(
        f"Embedding model: "
        f"{index_data['embedding_model']}"
    )
    print("Type 'quit' to exit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        retrieved_chunks = retrieve(
            question,
            index_data,
        )

        print_retrieved_chunks(retrieved_chunks)

        answer = answer_question(
            question,
            retrieved_chunks,
        )

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()