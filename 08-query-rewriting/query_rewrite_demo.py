from pathlib import Path

import chromadb
import ollama


PROJECT_DIR = Path.home() / "local-ai-learning"

CHROMA_DIR = (
    PROJECT_DIR
    / "05-vector-database"
    / "chroma_db"
)

COLLECTION_NAME = "ai_papers"

EMBEDDING_MODEL = "embeddinggemma"
REWRITE_MODEL = "ministral-3:3b"

TOP_K = 5


def load_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def rewrite_query(question):
    prompt = f"""
Rewrite the user's question into a concise search query
for retrieving relevant passages from technical AI papers.

Rules:
- Preserve the exact intent of the user's question.
- Do not answer the question.
- Do not introduce technologies, mechanisms, products,
  or technical terms that are not present in the question.
- Do not guess what vague references such as "it",
  "they", or "that thing" refer to.
- If the question is vague, preserve that ambiguity
  rather than inventing missing information.
- Return only the rewritten search query.

User question:
{question}
""".strip()

    response = ollama.chat(
        model=REWRITE_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 100,
        },
    )

    return response["message"]["content"].strip()


def retrieve(query, collection):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query,
    )

    query_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    records = []

    for i, record_id in enumerate(results["ids"][0]):
        records.append(
            {
                "id": record_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return records


def calculate_metrics(records):
    distances = [
        record["distance"]
        for record in records
    ]

    best_distance = min(distances)

    average_distance = (
        sum(distances)
        / len(distances)
    )

    top_document = records[0]["metadata"]["filename"]

    same_document_count = sum(
        1
        for record in records
        if record["metadata"]["filename"] == top_document
    )

    document_concentration = (
        same_document_count
        / len(records)
    )

    return {
        "best_distance": best_distance,
        "average_distance": average_distance,
        "document_concentration": document_concentration,
    }


def print_results(title, records):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for rank, record in enumerate(
        records,
        start=1,
    ):
        metadata = record["metadata"]

        print(
            f"\n{rank}. "
            f"{metadata['filename']}, "
            f"page {metadata['page_number']}, "
            f"chunk {metadata['chunk_number']}"
        )

        print(
            f"   Distance: "
            f"{record['distance']:.4f}"
        )

        print(
            f"   {record['document'][:350]}..."
        )


def print_comparison(
    original_metrics,
    rewritten_metrics,
):
    print("\n" + "=" * 70)
    print("NUMERIC COMPARISON")
    print("=" * 70)

    print("\nOriginal query:")
    print(
        f"Best distance: "
        f"{original_metrics['best_distance']:.4f}"
    )
    print(
        f"Average top-{TOP_K} distance: "
        f"{original_metrics['average_distance']:.4f}"
    )
    print(
        f"Top-document concentration: "
        f"{original_metrics['document_concentration']:.0%}"
    )

    print("\nRewritten query:")
    print(
        f"Best distance: "
        f"{rewritten_metrics['best_distance']:.4f}"
    )
    print(
        f"Average top-{TOP_K} distance: "
        f"{rewritten_metrics['average_distance']:.4f}"
    )
    print(
        f"Top-document concentration: "
        f"{rewritten_metrics['document_concentration']:.0%}"
    )

    best_change = (
        original_metrics["best_distance"]
        - rewritten_metrics["best_distance"]
    )

    average_change = (
        original_metrics["average_distance"]
        - rewritten_metrics["average_distance"]
    )

    print("\nChange after rewriting:")

    print(
        f"Best-distance improvement: "
        f"{best_change:+.4f}"
    )

    print(
        f"Average-distance improvement: "
        f"{average_change:+.4f}"
    )

    if average_change > 0:
        print(
            "\nResult: rewritten query produced "
            "closer top-k vector matches."
        )
    elif average_change < 0:
        print(
            "\nResult: original query produced "
            "closer top-k vector matches."
        )
    else:
        print(
            "\nResult: average vector distance "
            "was unchanged."
        )

    print(
        "\nNote: lower vector distance does not "
        "guarantee better retrieval relevance."
    )


def main():
    collection = load_collection()

    question = input(
        "\nEnter a vague or conversational question:\n> "
    ).strip()

    rewritten_query = rewrite_query(question)

    print("\nOriginal question:")
    print(question)

    print("\nRewritten search query:")
    print(rewritten_query)

    original_results = retrieve(
        question,
        collection,
    )

    rewritten_results = retrieve(
        rewritten_query,
        collection,
    )

    print_results(
        "ORIGINAL QUERY RESULTS",
        original_results,
    )

    print_results(
        "REWRITTEN QUERY RESULTS",
        rewritten_results,
    )

    original_metrics = calculate_metrics(
        original_results
    )

    rewritten_metrics = calculate_metrics(
        rewritten_results
    )

    print_comparison(
        original_metrics,
        rewritten_metrics,
    )


if __name__ == "__main__":
    main()