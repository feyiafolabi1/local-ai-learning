from pathlib import Path
import json

import chromadb
import ollama
from sentence_transformers import CrossEncoder


PROJECT_DIR = Path.home() / "local-ai-learning"

EVAL_FILE = (
    PROJECT_DIR
    / "07-rag-evaluation"
    / "eval_questions.json"
)

CHROMA_DIR = (
    PROJECT_DIR
    / "05-vector-database"
    / "chroma_db"
)

COLLECTION_NAME = "ai_papers"
EMBEDDING_MODEL = "embeddinggemma"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

RETRIEVE_K = 10
FINAL_K = 4


def load_eval_questions():
    return json.loads(
        EVAL_FILE.read_text(encoding="utf-8")
    )


def load_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def retrieve(question, collection):
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

    reranked = []

    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked


def find_expected_rank(
    candidates,
    expected_filename,
    acceptable_pages,
):
    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        metadata = candidate["metadata"]

        if (
            metadata["filename"] == expected_filename
            and metadata["page_number"] in acceptable_pages
        ):
            return rank

    return None


def evaluate_system(
    name,
    eval_questions,
    collection,
    reranker=None,
):
    total = len(eval_questions)

    top1_hits = 0
    top4_hits = 0
    ranks = []

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for item in eval_questions:
        question = item["question"]
        expected_filename = item["expected_filename"]
        acceptable_pages = item["acceptable_pages"]

        candidates = retrieve(
            question,
            collection,
        )

        if reranker is not None:
            candidates = rerank(
                question,
                candidates,
                reranker,
            )

        rank = find_expected_rank(
            candidates,
            expected_filename,
            acceptable_pages,
        )

        print(f"\nQuestion: {question}")

        print(
            f"Expected: "
            f"{expected_filename}, "
            f"acceptable pages {acceptable_pages}"
        )

        if rank is None:
            print(
                f"Result: NOT FOUND "
                f"in top {RETRIEVE_K}"
            )
        else:
            print(f"Result rank: {rank}")
            ranks.append(rank)

            if rank == 1:
                top1_hits += 1

            if rank <= FINAL_K:
                top4_hits += 1

    top1_accuracy = top1_hits / total
    top4_hit_rate = top4_hits / total

    if ranks:
        average_rank = sum(ranks) / len(ranks)
    else:
        average_rank = None

    print("\n" + "-" * 70)

    print(f"Questions: {total}")

    print(
        f"Top-1 accuracy: "
        f"{top1_accuracy:.1%}"
    )

    print(
        f"Top-{FINAL_K} hit rate: "
        f"{top4_hit_rate:.1%}"
    )

    if average_rank is not None:
        print(
            f"Average rank: "
            f"{average_rank:.2f}"
        )
    else:
        print("Average rank: N/A")

    return {
        "top1_accuracy": top1_accuracy,
        "top4_hit_rate": top4_hit_rate,
        "average_rank": average_rank,
    }


def main():
    eval_questions = load_eval_questions()
    collection = load_collection()

    print(
        f"Loaded {len(eval_questions)} "
        f"evaluation questions."
    )

    chroma_metrics = evaluate_system(
        name="SYSTEM A: Chroma only",
        eval_questions=eval_questions,
        collection=collection,
    )

    print("\nLoading reranker...")

    reranker = CrossEncoder(
        RERANK_MODEL
    )

    rerank_metrics = evaluate_system(
        name="SYSTEM B: Chroma + reranker",
        eval_questions=eval_questions,
        collection=collection,
        reranker=reranker,
    )

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"Top-1 accuracy: "
        f"{chroma_metrics['top1_accuracy']:.1%}"
        f" → "
        f"{rerank_metrics['top1_accuracy']:.1%}"
    )

    print(
        f"Top-{FINAL_K} hit rate: "
        f"{chroma_metrics['top4_hit_rate']:.1%}"
        f" → "
        f"{rerank_metrics['top4_hit_rate']:.1%}"
    )

    print(
        f"Average rank: "
        f"{chroma_metrics['average_rank']}"
        f" → "
        f"{rerank_metrics['average_rank']}"
    )


if __name__ == "__main__":
    main()