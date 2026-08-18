from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ollama


PROJECT_DIR = Path.home() / "local_rag"
INDEX_FILE = PROJECT_DIR / "index.json"
EVAL_FILE = PROJECT_DIR / "eval_questions.json"

TOP_K = 3


def load_json(path: Path) -> dict | list:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return json.loads(
        path.read_text(encoding="utf-8")
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

        similarity = cosine_similarity(
            question_vector,
            chunk_vector,
        )

        scored_chunks.append(
            {
                "filename": chunk["filename"],
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


def chunk_matches(
    chunk: dict,
    expected_file: str,
    expected_chunk: int,
) -> bool:
    return (
        chunk["filename"] == expected_file
        and chunk["chunk_number"] == expected_chunk
    )


def main() -> None:
    index_data = load_json(INDEX_FILE)
    eval_questions = load_json(EVAL_FILE)

    supported_questions = [
        item
        for item in eval_questions
        if item["should_answer"]
    ]

    top1_hits = 0
    top3_hits = 0

    print("=" * 72)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 72)

    for number, item in enumerate(
        supported_questions,
        start=1,
    ):
        question = item["question"]
        expected_file = item["expected_file"]
        expected_chunk = item["expected_chunk"]

        retrieved = retrieve(
            question,
            index_data,
            top_k=TOP_K,
        )

        top1_correct = chunk_matches(
            retrieved[0],
            expected_file,
            expected_chunk,
        )

        top3_correct = any(
            chunk_matches(
                chunk,
                expected_file,
                expected_chunk,
            )
            for chunk in retrieved
        )

        if top1_correct:
            top1_hits += 1

        if top3_correct:
            top3_hits += 1

        print(f"\n{number}. {question}")
        print(
            f"Expected: "
            f"{expected_file}, chunk {expected_chunk}"
        )

        print("\nRetrieved:")

        for rank, chunk in enumerate(
            retrieved,
            start=1,
        ):
            marker = ""

            if chunk_matches(
                chunk,
                expected_file,
                expected_chunk,
            ):
                marker = "  <-- EXPECTED"

            print(
                f"  {rank}. "
                f"{chunk['filename']}, "
                f"chunk {chunk['chunk_number']} "
                f"(similarity {chunk['similarity']:.4f})"
                f"{marker}"
            )

        print(
            f"\nTop-1 correct: "
            f"{'YES' if top1_correct else 'NO'}"
        )

        print(
            f"Top-3 correct: "
            f"{'YES' if top3_correct else 'NO'}"
        )

    total = len(supported_questions)

    top1_accuracy = (
        top1_hits / total * 100
        if total else 0
    )

    top3_accuracy = (
        top3_hits / total * 100
        if total else 0
    )

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        f"Supported questions: {total}"
    )

    print(
        f"Top-1 retrieval accuracy: "
        f"{top1_hits}/{total} "
        f"({top1_accuracy:.1f}%)"
    )

    print(
        f"Top-3 retrieval accuracy: "
        f"{top3_hits}/{total} "
        f"({top3_accuracy:.1f}%)"
    )


if __name__ == "__main__":
    main()