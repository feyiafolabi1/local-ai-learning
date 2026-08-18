from __future__ import annotations

import json
import re
from pathlib import Path

import ollama
from pypdf import PdfReader


PROJECT_DIR = Path.home() / "local-ai-learning" / "04-pdf-rag"
DOCUMENTS_DIR = PROJECT_DIR / "documents"
INDEX_FILE = PROJECT_DIR / "pdf_index.json"

EMBEDDING_MODEL = "embeddinggemma"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 30


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")

    # Join words split across lines with a hyphen.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def extract_pdf_records(pdf_path: Path) -> list[dict]:
    reader = PdfReader(pdf_path)
    records = []

    print(
        f"\nReading {pdf_path.name} "
        f"({len(reader.pages)} pages)"
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            print(
                f"  Page {page_number}: no extractable text"
            )
            continue

        page_chunks = split_into_chunks(cleaned_text)

        print(
            f"  Page {page_number}: "
            f"{len(page_chunks)} chunk(s)"
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):
            records.append(
                {
                    "filename": pdf_path.name,
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                }
            )

    return records


def load_pdf_documents() -> list[dict]:
    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    all_records = []

    for pdf_path in pdf_files:
        try:
            all_records.extend(
                extract_pdf_records(pdf_path)
            )
        except Exception as error:
            print(
                f"Could not process {pdf_path.name}: {error}"
            )

    return all_records


def create_embeddings(
    records: list[dict],
) -> list[dict]:
    texts = [
        record["text"]
        for record in records
    ]

    print(
        f"\nCreating {len(texts)} embeddings "
        f"with {EMBEDDING_MODEL}..."
    )

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = response["embeddings"]

    if len(embeddings) != len(records):
        raise RuntimeError(
            "Embedding count did not match chunk count."
        )

    for record, embedding in zip(
        records,
        embeddings,
    ):
        record["embedding"] = embedding

    return records


def main() -> None:
    records = load_pdf_documents()

    if not records:
        raise RuntimeError(
            "The PDFs produced no extractable text chunks. "
            "They may be scanned or image-only PDFs."
        )

    records = create_embeddings(records)

    index_data = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size_words": CHUNK_SIZE,
        "chunk_overlap_words": CHUNK_OVERLAP,
        "chunks": records,
    }

    INDEX_FILE.write_text(
        json.dumps(
            index_data,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nPDF ingestion complete.")
    print(f"Indexed chunks: {len(records)}")
    print(
        f"Vector dimensions: "
        f"{len(records[0]['embedding'])}"
    )
    print(f"Saved index: {INDEX_FILE}")


if __name__ == "__main__":
    main()