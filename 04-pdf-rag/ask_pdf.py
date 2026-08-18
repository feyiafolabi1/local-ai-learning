from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ollama


PROJECT_DIR = Path.home() / "local-ai-learning" / "04-pdf-rag"
INDEX_FILE = PROJECT_DIR / "pdf_index.json"

CHAT_MODEL = "ministral-3:3b"
TOP_K = 4

SYSTEM_PROMPT = """
You are a grounded PDF question-answering assistant.

Answer using only the retrieved PDF chunks.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts, specifications, or explanations.
3. Refuse only when none of the retrieved chunks directly contains
   information that answers the question.
4. If at least one retrieved chunk directly answers the question,
   answer using that chunk even if the answer is brief.
5. If refusing, say exactly:
   "The retrieved PDF sources do not provide enough information to answer that."
6. For supported questions, write at least one complete explanatory sentence.
7. Cite supporting sources using this format:
   [filename, page N, chunk N]
8. A citation by itself is not an answer.
9. Keep the answer concise.
10. Do not add mechanisms or terminology not explicitly stated in the chunks.
11. Paraphrase closely rather than expanding beyond the retrieved evidence.
""".strip()


def load_index() -> dict:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Index not found: {INDEX_FILE}\n"
            "Run ingest_pdfs.py first."
        )

    return json.loads(
        INDEX_FILE.read_text(encoding="utf-8")
    )


def cosine_similarity(
    query_vector: np.ndarray,
    chunk_vector: np.ndarray,
) -> float:
    query_norm = np.linalg.norm(query_vector)
    chunk_norm = np.linalg.norm(chunk_vector)

    if query_norm == 0 or chunk_norm == 0:
        return 0.0

    return float(
        np.dot(query_vector, chunk_vector)
        / (query_norm * chunk_norm)
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

        similarity = cosine_similarity(
            question_vector,
            chunk_vector,
        )

        scored_chunks.append(
            {
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "chunk_number": chunk["chunk_number"],
                "text": chunk["text"],
                "similarity": similarity,
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
            f"[{chunk['filename']}, "
            f"page {chunk['page_number']}, "
            f"chunk {chunk['chunk_number']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(sections)


def generate_answer(
    question: str,
    chunks: list[dict],
) -> str:
    context = build_context(chunks)

    prompt = f"""
RETRIEVED PDF SOURCES
---------------------
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
            "num_predict": 600,
        },
    )

    return (response["message"]["content"] or "").strip()


def print_retrieved_chunks(chunks: list[dict]) -> None:
    print("\nRetrieved PDF chunks:")

    for rank, chunk in enumerate(chunks, start=1):
        print(
            f"\n{rank}. "
            f"{chunk['filename']}, "
            f"page {chunk['page_number']}, "
            f"chunk {chunk['chunk_number']}"
        )

        print(
            f"   Similarity: "
            f"{chunk['similarity']:.4f}"
        )

        print(f"   {chunk['text']}")


def main() -> None:
    index_data = load_index()

    print(f"PDF RAG assistant using {CHAT_MODEL}")
    print(
        f"Embedding model: "
        f"{index_data['embedding_model']}"
    )
    print(
        f"Indexed chunks: "
        f"{len(index_data['chunks'])}"
    )
    print("Type 'quit' to exit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        chunks = retrieve(
            question,
            index_data,
        )

        print_retrieved_chunks(chunks)

        answer = generate_answer(
            question,
            chunks,
        )

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()