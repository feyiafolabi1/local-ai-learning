from __future__ import annotations

import json
from pathlib import Path

import ollama


PROJECT_DIR = Path.home() / "local_rag"
DOCUMENTS_DIR = PROJECT_DIR / "documents"
INDEX_FILE = PROJECT_DIR / "index.json"

EMBEDDING_MODEL = "embeddinggemma"


def split_into_chunks(text: str) -> list[str]:
    """
    Split a document by blank lines.

    Each paragraph becomes one chunk.
    """
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


def load_documents() -> list[dict]:
    records = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        chunks = split_into_chunks(text)

        for chunk_number, chunk_text in enumerate(chunks, start=1):
            records.append(
                {
                    "filename": file_path.name,
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                }
            )

    return records


def create_embeddings(records: list[dict]) -> list[dict]:
    texts = [record["text"] for record in records]

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = response["embeddings"]

    if len(embeddings) != len(records):
        raise RuntimeError(
            "The number of embeddings did not match the number of chunks."
        )

    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding

    return records


def main() -> None:
    records = load_documents()

    if not records:
        raise RuntimeError(
            f"No .txt documents found in {DOCUMENTS_DIR}"
        )

    print(f"Found {len(records)} chunks.")
    print(f"Embedding with {EMBEDDING_MODEL}...")

    records = create_embeddings(records)

    index_data = {
        "embedding_model": EMBEDDING_MODEL,
        "chunks": records,
    }

    INDEX_FILE.write_text(
        json.dumps(index_data, indent=2),
        encoding="utf-8",
    )

    print(f"Saved vector index to: {INDEX_FILE}")
    print(f"Indexed chunks: {len(records)}")
    print(f"Vector dimensions: {len(records[0]['embedding'])}")


if __name__ == "__main__":
    main()