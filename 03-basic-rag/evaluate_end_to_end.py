from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ollama


PROJECT_DIR = Path.home() / "local_rag"
INDEX_FILE = PROJECT_DIR / "index.json"
EVAL_FILE = PROJECT_DIR / "eval_questions.json"

CHAT_MODEL = "ministral-3:3b"
TOP_K = 3

REFUSAL_TEXT = (
    "The retrieved sources do not provide enough information to answer that."
)

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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def retrieve(question: str, index_data: dict) -> list[dict]:
    response = ollama.embed(
        model=index_data["embedding_model"],
        input=question,
    )

    question_vector = np.array(
        response["embeddings"][0],
        dtype=np.float32,
    )

    scored = []

    for chunk in index_data["chunks"]:
        chunk_vector = np.array(
            chunk["embedding"],
            dtype=np.float32,
        )

        score = cosine_similarity(
            question_vector,
            chunk_vector,
        )

        scored.append(
            {
                "filename": chunk["filename"],
                "chunk_number": chunk["chunk_number"],
                "text": chunk["text"],
                "similarity": score,
            }
        )

    scored.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return scored[:TOP_K]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{chunk['filename']}, chunk {chunk['chunk_number']}]\n"
        f"{chunk['text']}"
        for chunk in chunks
    )


def generate_answer(
    question: str,
    chunks: list[dict],
) -> str:
    context = build_context(chunks)

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

    return (response["message"]["content"] or "").strip()


def contains_expected_citation(
    answer: str,
    expected_file: str | None,
    expected_chunk: int | None,
) -> bool:
    if expected_file is None or expected_chunk is None:
        return False

    expected = f"[{expected_file}, chunk {expected_chunk}]"

    return expected.lower() in answer.lower()


def main() -> None:
    index_data = load_json(INDEX_FILE)
    eval_questions = load_json(EVAL_FILE)

    supported_total = 0
    supported_correct_behavior = 0

    unsupported_total = 0
    correct_refusals = 0

    citation_total = 0
    correct_citations = 0

    print("=" * 72)
    print("END-TO-END RAG EVALUATION")
    print("=" * 72)

    for number, item in enumerate(eval_questions, start=1):
        question = item["question"]

        chunks = retrieve(
            question,
            index_data,
        )

        answer = generate_answer(
            question,
            chunks,
        )

        should_answer = item["should_answer"]

        refused = answer.strip() == REFUSAL_TEXT

        print(f"\n{number}. {question}")
        print(f"Expected behavior: {'ANSWER' if should_answer else 'REFUSE'}")

        print("\nRetrieved:")
        for rank, chunk in enumerate(chunks, start=1):
            print(
                f"  {rank}. "
                f"{chunk['filename']}, "
                f"chunk {chunk['chunk_number']} "
                f"({chunk['similarity']:.4f})"
            )

        print("\nAnswer:")
        print(answer)

        if should_answer:
            supported_total += 1

            if not refused:
                supported_correct_behavior += 1

            citation_total += 1

            citation_ok = contains_expected_citation(
                answer,
                item["expected_file"],
                item["expected_chunk"],
            )

            if citation_ok:
                correct_citations += 1

            print(
                f"\nAnswered instead of refusing: "
                f"{'YES' if not refused else 'NO'}"
            )

            print(
                f"Expected citation present: "
                f"{'YES' if citation_ok else 'NO'}"
            )

        else:
            unsupported_total += 1

            if refused:
                correct_refusals += 1

            print(
                f"\nCorrect refusal: "
                f"{'YES' if refused else 'NO'}"
            )

    supported_rate = (
        supported_correct_behavior / supported_total * 100
        if supported_total else 0
    )

    refusal_rate = (
        correct_refusals / unsupported_total * 100
        if unsupported_total else 0
    )

    citation_rate = (
        correct_citations / citation_total * 100
        if citation_total else 0
    )

    print("\n" + "=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)

    print(
        f"Supported questions answered: "
        f"{supported_correct_behavior}/{supported_total} "
        f"({supported_rate:.1f}%)"
    )

    print(
        f"Unsupported questions refused: "
        f"{correct_refusals}/{unsupported_total} "
        f"({refusal_rate:.1f}%)"
    )

    print(
        f"Expected citations present: "
        f"{correct_citations}/{citation_total} "
        f"({citation_rate:.1f}%)"
    )


if __name__ == "__main__":
    main()