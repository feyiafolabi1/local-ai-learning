from pathlib import Path

import chromadb
import ollama


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

TOP_K = 10


# --------------------------------------------------
# QUERY EMBEDDING
# --------------------------------------------------

def format_query_for_embedding(
    query: str,
) -> str:
    """
    EmbeddingGemma retrieval-query format.
    """

    return (
        f"task: search result | "
        f"query: {query}"
    )


def embed_query(
    query: str,
) -> list[float]:

    formatted_query = (
        format_query_for_embedding(
            query
        )
    )

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=formatted_query,
    )

    return response["embeddings"][0]


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def load_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def search(
    collection,
    query: str,
    top_k: int = TOP_K,
):

    query_embedding = embed_query(
        query
    )

    return collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

def print_results(
    query: str,
    results,
):

    documents = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )

    distances = (
        results["distances"][0]
    )

    print()
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for rank, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):

        similarity = (
            1 - distance
        )

        print()
        print("-" * 80)

        print(
            f"Rank:       {rank}"
        )

        print(
            f"Similarity: {similarity:.4f}"
        )

        print(
            f"Chapter:    "
            f"{metadata['chapter']}"
        )

        print(
            f"File:       "
            f"{metadata['source_path']}"
        )

        print(
            f"Section:    "
            f"{metadata['section_name']}"
        )

        print()

        preview = document[:700]

        print(preview)

        if len(document) > 700:
            print("...")


# --------------------------------------------------
# TEST QUERIES
# --------------------------------------------------

TEST_QUERIES = [
    (
        "How did I learn that "
        "PagedAttention works?"
    ),

    (
        "What did I learn about "
        "KV cache during inference?"
    ),

    (
        "How did I benchmark "
        "multimodal concurrency?"
    ),

    (
        "What was my conclusion "
        "about arithmetic intensity?"
    ),
]


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print()
    print(
        f"Loading vector database from:\n"
        f"{CHROMA_DIR}"
    )

    collection = load_collection()

    print(
        f"\nVectors available: "
        f"{collection.count()}"
    )

    for query in TEST_QUERIES:

        results = search(
            collection,
            query,
            top_k=TOP_K,
        )

        print_results(
            query,
            results,
        )


if __name__ == "__main__":
    main()