from pathlib import Path

import chromadb
import ollama
from sentence_transformers import CrossEncoder


CHAPTER_5_DIR = Path.home() / "local-ai-learning" / "05-vector-database"
CHROMA_DIR = CHAPTER_5_DIR / "chroma_db"

COLLECTION_NAME = "ai_papers"
EMBEDDING_MODEL = "embeddinggemma"

RETRIEVE_K = 10
RERANK_K = 4

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def retrieve_candidates(question, collection):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question,
    )

    question_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=RETRIEVE_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    candidates = []

    for i, record_id in enumerate(results["ids"][0]):
        candidates.append(
            {
                "id": record_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return candidates


def rerank(question, candidates, model):
    pairs = [
        [question, candidate["document"]]
        for candidate in candidates
    ]

    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return candidates


def print_results(title, candidates, limit):
    print(f"\n{title}")

    for rank, candidate in enumerate(
        candidates[:limit],
        start=1,
    ):
        metadata = candidate["metadata"]

        print(
            f"\n{rank}. "
            f"{metadata['filename']}, "
            f"page {metadata['page_number']}, "
            f"chunk {metadata['chunk_number']}"
        )

        print(
            f"   Vector distance: "
            f"{candidate['distance']:.4f}"
        )

        if "rerank_score" in candidate:
            print(
                f"   Rerank score: "
                f"{candidate['rerank_score']:.4f}"
            )

        print(
            f"   {candidate['document'][:350]}..."
        )


def main():
    print("Loading Chroma collection...")
    collection = load_collection()

    print("Loading reranker...")
    reranker = CrossEncoder(RERANK_MODEL)

    question = input("\nQuestion: ").strip()

    candidates = retrieve_candidates(
        question,
        collection,
    )

    print_results(
        "Chroma top 10:",
        candidates,
        RETRIEVE_K,
    )

    reranked = rerank(
        question,
        candidates,
        reranker,
    )

    print_results(
        "Reranked top 4:",
        reranked,
        RERANK_K,
    )


if __name__ == "__main__":
    main()