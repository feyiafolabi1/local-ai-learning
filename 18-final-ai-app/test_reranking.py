from pathlib import Path
import time

import chromadb
import ollama
from sentence_transformers import CrossEncoder


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

RERANKER_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"
)

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 4


# --------------------------------------------------
# QUERY EMBEDDING
# --------------------------------------------------

def format_query_for_embedding(
    query: str,
) -> str:

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
# VECTOR RETRIEVAL
# --------------------------------------------------

def retrieve_candidates(
    collection,
    query: str,
):

    start = time.perf_counter()

    query_embedding = embed_query(
        query
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=RETRIEVAL_TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    end = time.perf_counter()

    candidates = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for original_rank, (
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

        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "vector_similarity": similarity,
                "vector_rank": original_rank,
            }
        )

    retrieval_latency = (
        end - start
    )

    return (
        candidates,
        retrieval_latency,
    )


# --------------------------------------------------
# QUERY INTENT
# --------------------------------------------------

def query_wants_code_or_exact_artifact(
    query: str,
) -> bool:
    """
    Detect queries where source code, exact prompts,
    scripts, or implementation details are likely
    more useful than explanatory notes.
    """

    query_lower = query.lower()

    code_keywords = [
        "exact prompt",
        "benchmark prompt",
        "script",
        "source code",
        "code",
        "function",
        "implementation",
        "implemented",
        "how did i implement",
        "how did i code",
        "what file",
    ]

    return any(
        keyword in query_lower
        for keyword in code_keywords
    )


# --------------------------------------------------
# SOURCE UTILITY
# --------------------------------------------------

def calculate_source_bonus(
    query: str,
    candidate,
) -> float:
    """
    Add a small usefulness adjustment after the
    cross-encoder relevance score.

    Conceptual questions:
    - prefer README explanations
    - slightly penalize benchmark/test prompts

    Code/artifact questions:
    - prefer Python implementation chunks
    """

    metadata = candidate["metadata"]

    file_name = (
        metadata["file_name"]
        .lower()
    )

    file_type = (
        metadata["file_type"]
        .lower()
    )

    section_name = (
        metadata["section_name"]
        .lower()
    )

    wants_code = (
        query_wants_code_or_exact_artifact(
            query
        )
    )

    bonus = 0.0

    # ----------------------------------------------
    # CODE / IMPLEMENTATION QUERY
    # ----------------------------------------------

    if wants_code:

        if file_type == "python":
            bonus += 0.75

        if section_name.startswith(
            "function "
        ):
            bonus += 0.25

        return bonus

    # ----------------------------------------------
    # CONCEPTUAL / EXPLANATORY QUERY
    # ----------------------------------------------

    if file_name == "readme.md":
        bonus += 0.75

    # Benchmark prompts and fine-tuning examples can
    # be semantically relevant but are often not the
    # best source for answering conceptual questions.
    if "assignment tasks item" in section_name:
        bonus -= 0.50

    if "assignment examples item" in section_name:
        bonus -= 0.50

    return bonus


# --------------------------------------------------
# RERANKING
# --------------------------------------------------

def rerank_candidates(
    reranker,
    query: str,
    candidates,
):

    pairs = [
        (
            query,
            candidate["document"],
        )
        for candidate in candidates
    ]

    start = time.perf_counter()

    scores = reranker.predict(
        pairs
    )

    end = time.perf_counter()

    for candidate, score in zip(
        candidates,
        scores,
    ):

        reranker_score = float(
            score
        )

        source_bonus = (
            calculate_source_bonus(
                query,
                candidate,
            )
        )

        final_score = (
            reranker_score
            + source_bonus
        )

        candidate[
            "reranker_score"
        ] = reranker_score

        candidate[
            "source_bonus"
        ] = source_bonus

        candidate[
            "final_score"
        ] = final_score

    reranked = sorted(
        candidates,
        key=lambda candidate:
            candidate["final_score"],
        reverse=True,
    )

    for reranked_rank, candidate in enumerate(
        reranked,
        start=1,
    ):
        candidate["reranked_rank"] = (
            reranked_rank
        )

    reranking_latency = (
        end - start
    )

    return (
        reranked,
        reranking_latency,
    )


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

def print_comparison(
    query: str,
    reranked,
    retrieval_latency: float,
    reranking_latency: float,
):

    print()
    print("=" * 90)
    print(f"QUERY: {query}")
    print("=" * 90)

    print()
    print(
        f"Vector retrieval latency: "
        f"{retrieval_latency * 1000:.2f} ms"
    )

    print(
        f"Reranking latency:        "
        f"{reranking_latency * 1000:.2f} ms"
    )

    print(
        f"Candidates retrieved:     "
        f"{len(reranked)}"
    )

    print(
        f"Candidates kept:          "
        f"{RERANK_TOP_K}"
    )

    print()
    print("FINAL RERANKED RESULTS")
    print("-" * 90)

    for candidate in reranked[
        :RERANK_TOP_K
    ]:

        metadata = candidate[
            "metadata"
        ]

        vector_rank = candidate[
            "vector_rank"
        ]

        reranked_rank = candidate[
            "reranked_rank"
        ]

        movement = (
            vector_rank
            - reranked_rank
        )

        if movement > 0:
            movement_text = (
                f"↑ {movement}"
            )

        elif movement < 0:
            movement_text = (
                f"↓ {abs(movement)}"
            )

        else:
            movement_text = "—"

        print()
        print("-" * 90)

        print(
            f"Final rank:         "
            f"{reranked_rank}"
        )

        print(
            f"Vector rank:        "
            f"{vector_rank}"
        )

        print(
            f"Movement:           "
            f"{movement_text}"
        )

        print(
            f"Vector similarity:  "
            f"{candidate['vector_similarity']:.4f}"
        )

        print(
            f"Reranker score:     "
            f"{candidate['reranker_score']:.4f}"
        )

        print(
            f"Source bonus:       "
            f"{candidate['source_bonus']:+.2f}"
        )

        print(
            f"Final score:        "
            f"{candidate['final_score']:.4f}"
        )

        print(
            f"Chapter:            "
            f"{metadata['chapter']}"
        )

        print(
            f"File:               "
            f"{metadata['source_path']}"
        )

        print(
            f"Section:            "
            f"{metadata['section_name']}"
        )

        print()

        preview = (
            candidate["document"][:700]
        )

        print(preview)

        if len(
            candidate["document"]
        ) > 700:
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

    # This deliberately asks for an exact artifact.
    # Python/task chunks should become more useful.
    (
        "What exact benchmark prompt "
        "did I use for KV cache?"
    ),
]


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("LOADING RAG COMPONENTS")
    print("=" * 70)

    print()
    print(
        f"Vector database:\n"
        f"{CHROMA_DIR}"
    )

    collection = load_collection()

    print(
        f"\nVectors available: "
        f"{collection.count()}"
    )

    print()
    print(
        f"Loading reranker:\n"
        f"{RERANKER_MODEL}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print()
    print("Reranker loaded.")

    for query in TEST_QUERIES:

        candidates, retrieval_latency = (
            retrieve_candidates(
                collection,
                query,
            )
        )

        (
            reranked,
            reranking_latency,
        ) = rerank_candidates(
            reranker,
            query,
            candidates,
        )

        print_comparison(
            query,
            reranked,
            retrieval_latency,
            reranking_latency,
        )


if __name__ == "__main__":
    main()