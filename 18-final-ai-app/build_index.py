from pathlib import Path
import time

import chromadb
import ollama

from ingest_repo import (
    REPO_ROOT,
    ingest_repo,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_DIR = (
    Path.home()
    / "local-ai-learning"
    / "18-final-ai-app"
)

CHROMA_DIR = (
    PROJECT_DIR
    / "chroma_db"
)

COLLECTION_NAME = "local_ai_learning"

EMBEDDING_MODEL = "embeddinggemma"

EMBED_BATCH_SIZE = 16

MIN_CHUNK_CHARACTERS = 80


# --------------------------------------------------
# INDEX FILTERING
# --------------------------------------------------

def should_index_chunk(chunk) -> bool:
    """
    Remove chunks that are unlikely to be useful
    retrieval targets.

    We keep:
    - Markdown sections
    - functions/classes
    - meaningful assignments
    - benchmark/task definitions
    - substantial module-level code

    We skip:
    - import-only chunks
    - tiny module-level expressions
    - tiny trivial assignments
    """

    text = chunk.text.strip()

    if not text:
        return False

    # Imports almost never answer a user question.
    if chunk.section_name == "imports":
        return False

    # Tiny chunks tend to create noisy retrieval hits.
    if len(text) < MIN_CHUNK_CHARACTERS:
        return False

    # Examples:
    # print("PyTorch version:", ...)
    if (
        chunk.section_name.startswith(
            "module-level Expr"
        )
        and len(text) < 250
    ):
        return False

    # Keep everything else.
    return True


# --------------------------------------------------
# DOCUMENT FORMATTING
# --------------------------------------------------

def make_document_title(chunk) -> str:
    """
    Build a useful document title for EmbeddingGemma.

    Example:

    16-inference-serving | README.md | ## PagedAttention
    """

    return (
        f"{chunk.chapter} | "
        f"{chunk.file_name} | "
        f"{chunk.section_name}"
    )


def format_document_for_embedding(
    chunk,
) -> str:
    """
    EmbeddingGemma retrieval-document format:

    title: <title> | text: <content>
    """

    title = make_document_title(
        chunk
    )

    return (
        f"title: {title} | "
        f"text: {chunk.text}"
    )


# --------------------------------------------------
# EMBEDDING
# --------------------------------------------------

def embed_documents(
    chunks,
) -> list[list[float]]:
    """
    Generate document embeddings using the
    EmbeddingGemma retrieval-document prompt.
    """

    formatted_documents = [
        format_document_for_embedding(
            chunk
        )
        for chunk in chunks
    ]

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=formatted_documents,
    )

    return response["embeddings"]


# --------------------------------------------------
# CHROMA DATABASE
# --------------------------------------------------

def create_fresh_collection(
    client: chromadb.PersistentClient,
):
    """
    Delete the old collection and rebuild from
    scratch so old embeddings cannot remain.
    """

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Deleted existing collection: "
            f"{COLLECTION_NAME}"
        )

    except Exception:
        pass

    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )


# --------------------------------------------------
# INDEX BUILDING
# --------------------------------------------------

def build_index():

    print()
    print("=" * 70)
    print("BUILDING LOCAL AI LEARNING VECTOR INDEX")
    print("=" * 70)

    print()
    print(
        f"Repository:\n"
        f"{REPO_ROOT}"
    )

    print()
    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Vector database: "
        f"{CHROMA_DIR}"
    )

    # ----------------------------------------------
    # STEP 1: INGEST
    # ----------------------------------------------

    print()
    print("Step 1/4: Ingesting repository...")

    all_chunks = ingest_repo(
        REPO_ROOT
    )

    print(
        f"Structured chunks created: "
        f"{len(all_chunks)}"
    )

    # ----------------------------------------------
    # STEP 2: FILTER
    # ----------------------------------------------

    print()
    print(
        "Step 2/4: Filtering "
        "low-information chunks..."
    )

    chunks = [
        chunk
        for chunk in all_chunks
        if should_index_chunk(chunk)
    ]

    removed = (
        len(all_chunks)
        - len(chunks)
    )

    print(
        f"Chunks kept:             "
        f"{len(chunks)}"
    )

    print(
        f"Chunks removed:          "
        f"{removed}"
    )

    if not chunks:
        raise RuntimeError(
            "No chunks remained after filtering."
        )

    # ----------------------------------------------
    # STEP 3: CREATE DATABASE
    # ----------------------------------------------

    print()
    print(
        "Step 3/4: Creating fresh "
        "ChromaDB collection..."
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = create_fresh_collection(
        client
    )

    # ----------------------------------------------
    # STEP 4: EMBED + STORE
    # ----------------------------------------------

    print()
    print(
        "Step 4/4: Generating retrieval-aware "
        "embeddings and storing chunks..."
    )

    start_time = time.perf_counter()

    total_chunks = len(chunks)

    for batch_start in range(
        0,
        total_chunks,
        EMBED_BATCH_SIZE,
    ):

        batch_end = min(
            batch_start + EMBED_BATCH_SIZE,
            total_chunks,
        )

        batch = chunks[
            batch_start:
            batch_end
        ]

        embeddings = embed_documents(
            batch
        )

        ids = []
        documents = []
        metadatas = []

        for offset, chunk in enumerate(batch):

            global_index = (
                batch_start + offset
            )

            ids.append(
                f"chunk-{global_index:05d}"
            )

            # Store the RAW chunk text in Chroma.
            # Only the embedding input gets the title/text
            # wrapper.
            documents.append(
                chunk.text
            )

            metadatas.append(
                {
                    "source_path":
                        chunk.source_path,

                    "chapter":
                        chunk.chapter,

                    "file_name":
                        chunk.file_name,

                    "file_type":
                        chunk.file_type,

                    "chunk_index":
                        chunk.chunk_index,

                    "section_name":
                        chunk.section_name,

                    "embedding_title":
                        make_document_title(
                            chunk
                        ),
                }
            )

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"Indexed "
            f"{batch_end}/{total_chunks} "
            f"chunks"
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    stored_count = collection.count()

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Original chunks:       "
        f"{len(all_chunks)}"
    )

    print(
        f"Filtered chunks:       "
        f"{total_chunks}"
    )

    print(
        f"Vectors stored:        "
        f"{stored_count}"
    )

    print(
        f"Embedding model:       "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Embedding batch size:  "
        f"{EMBED_BATCH_SIZE}"
    )

    print(
        f"Build time:            "
        f"{elapsed:.2f} seconds"
    )

    if elapsed > 0:

        rate = (
            total_chunks
            / elapsed
        )

        print(
            f"Embedding rate:        "
            f"{rate:.2f} chunks/sec"
        )

    print(
        f"Database location:     "
        f"{CHROMA_DIR}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    build_index()


if __name__ == "__main__":
    main()