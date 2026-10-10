from pathlib import Path
import time

import chromadb
import ollama
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# CONFIG — same models, collection, and ranking rules
# --------------------------------------------------

PROJECT_DIR = Path.home() / "local-ai-learning" / "18-final-ai-app"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "local_ai_learning"

EMBEDDING_MODEL = "embeddinggemma"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_TOP_K = 10
FINAL_TOP_K = 4

MIN_FINAL_SCORE = 0.0


# --------------------------------------------------
# QUERY EMBEDDING
# --------------------------------------------------

def format_query_for_embedding(query: str) -> str:

    return f"task: search result | query: {query}"


def embed_query(query: str) -> list[float]:

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=format_query_for_embedding(query),
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
# QUERY INTENT
# --------------------------------------------------

def query_wants_code_or_exact_artifact(
    query: str,
) -> bool:

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
# SOURCE BONUS — ORIGINAL RANKING RULES
# --------------------------------------------------

def calculate_source_bonus(
    query: str,
    candidate,
) -> float:

    metadata = candidate["metadata"]

    file_name = metadata["file_name"].lower()

    file_type = metadata["file_type"].lower()

    section_name = metadata["section_name"].lower()

    wants_code = query_wants_code_or_exact_artifact(
        query
    )

    bonus = 0.0

    if wants_code:

        if file_type == "python":
            bonus += 0.75

        if section_name.startswith("function "):
            bonus += 0.25

        return bonus

    if file_name == "readme.md":
        bonus += 0.75

    if "assignment tasks item" in section_name:
        bonus -= 0.50

    if "assignment examples item" in section_name:
        bonus -= 0.50

    if section_name.strip() in {
        "## files",
        "# files",
        "### files",
    }:
        bonus -= 0.75

    return bonus


# --------------------------------------------------
# VECTOR RETRIEVAL — NEW DETAILED TIMING
# --------------------------------------------------

def retrieve_candidates(
    collection,
    query: str,
    *,
    include_stage_metrics=False,
):

    retrieval_start = time.perf_counter()

    # ----------------------------------------------
    # STAGE 1 — QUERY EMBEDDING
    # ----------------------------------------------

    embedding_start = time.perf_counter()

    query_embedding = embed_query(query)

    embedding_ms = (
        time.perf_counter() - embedding_start
    ) * 1000

    # ----------------------------------------------
    # STAGE 2 — CHROMA VECTOR SEARCH
    # ----------------------------------------------

    chroma_start = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVAL_TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    chroma_ms = (
        time.perf_counter() - chroma_start
    ) * 1000

    # Preserve the original retrieval metric:
    # embedding + Chroma search.

    retrieval_latency = (
        time.perf_counter() - retrieval_start
    )

    # ----------------------------------------------
    # STAGE 3 — BUILD CANDIDATES
    # ----------------------------------------------

    build_start = time.perf_counter()

    candidates = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for vector_rank, (
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

        candidates.append({
            "document": document,
            "metadata": metadata,
            "vector_similarity": 1 - distance,
            "vector_rank": vector_rank,
        })

    build_ms = (
        time.perf_counter() - build_start
    ) * 1000

    # Optional extra metrics for RAGEngine.search().
    #
    # Existing callers can still receive the
    # original two-value return.

    if include_stage_metrics:

        return candidates, retrieval_latency, {
            "query_embedding_latency_ms": embedding_ms,
            "chroma_query_latency_ms": chroma_ms,
            "candidate_build_latency_ms": build_ms,
        }

    return candidates, retrieval_latency


# --------------------------------------------------
# RERANKING — ORIGINAL SCORING LOGIC
# --------------------------------------------------

def rerank_candidates(
    reranker,
    query: str,
    candidates,
):

    pairs = [
        (query, candidate["document"])
        for candidate in candidates
    ]

    # ----------------------------------------------
    # CROSS-ENCODER INFERENCE
    # ----------------------------------------------

    start = time.perf_counter()

    scores = reranker.predict(pairs)

    reranking_latency = (
        time.perf_counter() - start
    )

    # ----------------------------------------------
    # SOURCE BONUSES
    # ----------------------------------------------

    for candidate, score in zip(
        candidates,
        scores,
    ):

        reranker_score = float(score)

        source_bonus = calculate_source_bonus(
            query,
            candidate,
        )

        final_score = (
            reranker_score + source_bonus
        )

        candidate["reranker_score"] = (
            reranker_score
        )

        candidate["source_bonus"] = (
            source_bonus
        )

        candidate["final_score"] = (
            final_score
        )

    # ----------------------------------------------
    # SORT
    # ----------------------------------------------

    reranked = sorted(
        candidates,
        key=lambda item: item["final_score"],
        reverse=True,
    )

    for rank, candidate in enumerate(
        reranked,
        start=1,
    ):

        candidate["reranked_rank"] = rank

    return reranked, reranking_latency


# --------------------------------------------------
# RELEVANCE FILTER — ORIGINAL RULES
# --------------------------------------------------

def filter_relevant_candidates(
    reranked,
    top_k: int = FINAL_TOP_K,
):
    """
    Always keep the highest-ranked candidate.

    Additional candidates must satisfy:
        final_score >= MIN_FINAL_SCORE

    Never return more than top_k candidates.
    """

    if not reranked:
        return []

    selected = [reranked[0]]

    for candidate in reranked[1:]:

        if len(selected) >= top_k:
            break

        if candidate["final_score"] >= MIN_FINAL_SCORE:
            selected.append(candidate)

    for final_rank, candidate in enumerate(
        selected,
        start=1,
    ):

        candidate["final_rank"] = final_rank

    return selected


# --------------------------------------------------
# RAG ENGINE — DETAILED OBSERVABILITY
# --------------------------------------------------

class RAGEngine:

    def __init__(self):

        print("Loading Chroma collection...")

        self.collection = load_collection()

        print(
            f"Loaded {self.collection.count()} vectors."
        )

        print("Loading reranker...")

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print("RAG engine ready.")

    # ----------------------------------------------
    # SEARCH
    # ----------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
    ):

        search_start = time.perf_counter()

        # ------------------------------------------
        # VECTOR RETRIEVAL
        # ------------------------------------------

        (
            candidates,
            retrieval_latency,
            stages,
        ) = retrieve_candidates(
            self.collection,
            query,
            include_stage_metrics=True,
        )

        # ------------------------------------------
        # RERANKING
        # ------------------------------------------

        rerank_start = time.perf_counter()

        reranked, reranking_latency = rerank_candidates(
            self.reranker,
            query,
            candidates,
        )

        rerank_total_ms = (
            time.perf_counter() - rerank_start
        ) * 1000

        # ------------------------------------------
        # RELEVANCE FILTERING
        # ------------------------------------------

        filter_start = time.perf_counter()

        selected = filter_relevant_candidates(
            reranked,
            top_k=top_k,
        )

        filter_ms = (
            time.perf_counter() - filter_start
        ) * 1000

        # ------------------------------------------
        # COMPLETE SEARCH LATENCY
        # ------------------------------------------

        search_total_ms = (
            time.perf_counter() - search_start
        ) * 1000

        # ------------------------------------------
        # METRICS
        # ------------------------------------------

        metrics = {

            # Original metrics — preserved

            "retrieval_latency_ms":
                retrieval_latency * 1000,

            "reranking_latency_ms":
                reranking_latency * 1000,

            "candidates_retrieved":
                len(candidates),

            "candidates_after_reranking":
                len(reranked),

            "chunks_selected":
                len(selected),

            "chunks_filtered_out":
                len(reranked) - len(selected),

            "minimum_final_score":
                MIN_FINAL_SCORE,

            # New observability metrics

            "query_embedding_latency_ms":
                stages["query_embedding_latency_ms"],

            "chroma_query_latency_ms":
                stages["chroma_query_latency_ms"],

            "candidate_build_latency_ms":
                stages["candidate_build_latency_ms"],

            # Includes pair construction,
            # source bonuses, sorting, etc.

            "reranking_overhead_latency_ms": max(
                0.0,
                rerank_total_ms
                - reranking_latency * 1000,
            ),

            "relevance_filter_latency_ms":
                filter_ms,

            "search_total_latency_ms":
                search_total_ms,
        }

        return selected, metrics


# --------------------------------------------------
# CONTEXT FORMATTING — ORIGINAL SOURCE STRUCTURE
# --------------------------------------------------

def format_context(results) -> str:

    sections = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        section = (
            f"[SOURCE {index}]\n"
            f"Chapter: {metadata['chapter']}\n"
            f"File: {metadata['source_path']}\n"
            f"Section: {metadata['section_name']}\n"
            f"Final score: {result['final_score']:.4f}\n\n"
            f"{result['document']}"
        )

        sections.append(section)

    return "\n\n".join(sections)


# --------------------------------------------------
# MANUAL TEST
# --------------------------------------------------

def main():

    rag = RAGEngine()

    query = (
        "What did my GPU experiments teach me "
        "about arithmetic intensity?"
    )

    results, metrics = rag.search(query)

    print()
    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    print()
    print("=" * 80)
    print("SELECTED SOURCES")
    print("=" * 80)

    for result in results:

        metadata = result["metadata"]

        print()
        print("-" * 80)

        print(
            f"Final rank: {result['final_rank']}"
        )

        print(
            f"Vector rank: {result['vector_rank']}"
        )

        print(
            f"Reranked rank: {result['reranked_rank']}"
        )

        print(
            f"Chapter: {metadata['chapter']}"
        )

        print(
            f"File: {metadata['source_path']}"
        )

        print(
            f"Section: {metadata['section_name']}"
        )

        print(
            f"Reranker score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Source bonus: "
            f"{result['source_bonus']:+.2f}"
        )

        print(
            f"Final score: "
            f"{result['final_score']:.4f}"
        )

    print()
    print("=" * 80)
    print("RAG METRICS")
    print("=" * 80)

    for name, value in metrics.items():

        if name.endswith("_latency_ms"):

            print(
                f"{name}: {value:.2f} ms"
            )

        else:

            print(
                f"{name}: {value}"
            )

    print()
    print("=" * 80)
    print("FORMATTED CONTEXT")
    print("=" * 80)

    print(
        format_context(results)
    )


if __name__ == "__main__":
    main()