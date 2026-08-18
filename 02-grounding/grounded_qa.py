from pathlib import Path

import ollama


MODEL = "llama3.2:3b"
SOURCE_FILE = Path.home() / "hbm_source.txt"

SYSTEM_PROMPT = """
You are a grounded question-answering assistant.

Answer using only the supplied source text.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts, specifications, or explanations.
3. If the source does not contain enough information, say exactly:
   "The source does not provide enough information to answer that."
4. When the source supports the answer, include a short Evidence section
   containing the exact source sentence or sentences that support it.
5. When the source does not contain enough information, give only the
   refusal sentence and do not include an Evidence section.
6. Never use an unrelated sentence as evidence that information is absent.
7. Keep the answer concise.
""".strip()


def load_source() -> str:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}"
        )

    return SOURCE_FILE.read_text(encoding="utf-8").strip()


def ask_question(source: str, question: str) -> str:
    prompt = f"""
SOURCE TEXT
-----------
{source}

QUESTION
--------
{question}
""".strip()

    response = ollama.chat(
        model=MODEL,
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


def main() -> None:
    source = load_source()

    print(f"Grounded Q&A using {MODEL}")
    print(f"Source: {SOURCE_FILE}")
    print("Type 'quit' to exit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        answer = ask_question(source, question)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()