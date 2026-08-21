from __future__ import annotations

from pathlib import Path

import chromadb
import ollama


PROJECT_DIR = Path.home() / "local-ai-learning" / "05-vector-database"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "ai_papers"
EMBEDDING_MODEL = "embeddinggemma"
CHAT_MODEL = "ministral-3:3b"
TOP_K = 4

SYSTEM_PROMPT = """
You are a grounded PDF question-answering assistant.

Answer using only the retrieved PDF chunks.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or explanations.
3. If the retrieved sources do not contain enough information, say exactly:
   "The retrieved PDF sources do not provide enough information to answer that."
4. Cite supporting sources using this format:
   [filename, page N, chunk N]
5. Keep the answer concise.
""".strip()


def load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def retrieve(
    question: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
) -> list[dict]:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question,
    )

    question_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    for index, record_id in enumerate(results["ids"][0]):
        retrieved.append(
            {
                "id": record_id,
                "document": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index],
            }
        )

    return retrieved


def build_context(records: list[dict]) -> str:
    sections = []

    for record in records:
        metadata = record["metadata"]

        sections.append(
            f"[{metadata['filename']}, "
            f"page {metadata['page_number']}, "
            f"chunk {metadata['chunk_number']}]\n"
            f"{record['document']}"
        )

    return "\n\n".join(sections)


def generate_answer(
    question: str,
    records: list[dict],
) -> str:
    context = build_context(records)

    prompt = f"""
RETRIEVED PDF SOURCES
---------------------
{context}

QUESTION
--------
{question}
""".strip()

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 600,
        },
    )

    return response["message"]["content"].strip()


def print_retrieved_records(records: list[dict]) -> None:
    print("\nRetrieved Chroma records:")

    for rank, record in enumerate(records, start=1):
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

        print(f"   {record['document']}")


def main() -> None:
    collection = load_collection()

    print(f"Chroma RAG assistant using {CHAT_MODEL}")
    print(f"Collection: {collection.name}")
    print(f"Records stored: {collection.count()}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Type 'quit' to exit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        records = retrieve(
            question=question,
            collection=collection,
        )

        print_retrieved_records(records)

        answer = generate_answer(
            question=question,
            records=records,
        )

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()