from __future__ import annotations

import re
import shutil
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader


PROJECT_DIR = Path.home() / "local-ai-learning" / "05-vector-database"
DOCUMENTS_DIR = PROJECT_DIR / "documents"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "ai_papers"
EMBEDDING_MODEL = "embeddinggemma"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 30
EMBED_BATCH_SIZE = 32


def clean_text(text: str) -> str:
    """
    Clean text extracted from a PDF.

    - Remove null characters.
    - Join words broken across lines with a hyphen.
    - Collapse repeated whitespace.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """
    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def extract_pdf_records(pdf_path: Path) -> list[dict]:
    """
    Extract and chunk one PDF page by page.

    Each returned record contains:
    - unique ID
    - original chunk text
    - metadata
    """
    reader = PdfReader(pdf_path)
    records = []

    print(f"\nProcessing: {pdf_path.name}")
    print(f"Pages: {len(reader.pages)}")

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_text(raw_text)

        page_chunks = chunk_text(cleaned_text)

        print(
            f"  Page {page_number}: "
            f"{len(page_chunks)} chunk(s)"
        )

        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1,
        ):
            record_id = (
                f"{pdf_path.stem}"
                f"-page-{page_number}"
                f"-chunk-{chunk_number}"
            )

            records.append(
                {
                    "id": record_id,
                    "document": chunk,
                    "metadata": {
                        "filename": pdf_path.name,
                        "page_number": page_number,
                        "chunk_number": chunk_number,
                    },
                }
            )

    return records


def embed_records(records: list[dict]) -> list[list[float]]:
    """
    Generate embeddings in batches using EmbeddingGemma.
    """
    all_embeddings = []

    for start in range(0, len(records), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE
        batch = records[start:end]

        batch_texts = [
            record["document"]
            for record in batch
        ]

        print(
            f"Embedding records "
            f"{start + 1}–{start + len(batch)} "
            f"of {len(records)}"
        )

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=batch_texts,
        )

        all_embeddings.extend(response["embeddings"])

    return all_embeddings


def recreate_database() -> chromadb.Collection:
    """
    Delete the old local Chroma database and create
    a fresh collection.

    This keeps this first version simple and predictable.
    """
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size_words": CHUNK_SIZE,
            "chunk_overlap_words": CHUNK_OVERLAP,
        },
    )

    return collection


def add_records_to_chroma(
    collection: chromadb.Collection,
    records: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Add records to Chroma in manageable batches.
    """
    for start in range(0, len(records), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE

        batch_records = records[start:end]
        batch_embeddings = embeddings[start:end]

        collection.add(
            ids=[
                record["id"]
                for record in batch_records
            ],
            documents=[
                record["document"]
                for record in batch_records
            ],
            metadatas=[
                record["metadata"]
                for record in batch_records
            ],
            embeddings=batch_embeddings,
        )

        print(
            f"Stored records "
            f"{start + 1}–{start + len(batch_records)} "
            f"of {len(records)}"
        )


def main() -> None:
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in: {DOCUMENTS_DIR}"
        )

    all_records = []

    for pdf_path in pdf_paths:
        all_records.extend(
            extract_pdf_records(pdf_path)
        )

    if not all_records:
        raise ValueError(
            "The PDFs did not produce any text chunks."
        )

    print(f"\nTotal chunks: {len(all_records)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    embeddings = embed_records(all_records)

    if len(embeddings) != len(all_records):
        raise RuntimeError(
            "The number of embeddings does not match "
            "the number of records."
        )

    print(
        f"Embedding dimensions: "
        f"{len(embeddings[0])}"
    )

    collection = recreate_database()

    add_records_to_chroma(
        collection=collection,
        records=all_records,
        embeddings=embeddings,
    )

    print("\nChroma ingestion complete.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Records stored: {collection.count()}")
    print(f"Database location: {CHROMA_DIR}")


if __name__ == "__main__":
    main()