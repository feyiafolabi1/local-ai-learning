from pathlib import Path

import chromadb


PROJECT_DIR = Path.home() / "local-ai-learning" / "05-vector-database"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "ai_papers"


def main() -> None:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(f"Collection: {collection.name}")
    print(f"Record count: {collection.count()}")

    results = collection.get(
        limit=5,
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ],
    )

    for index, record_id in enumerate(results["ids"], start=1):
        document = results["documents"][index - 1]
        metadata = results["metadatas"][index - 1]
        embedding = results["embeddings"][index - 1]

        print("\n" + "=" * 80)
        print(f"Record {index}")
        print(f"ID: {record_id}")
        print(f"Metadata: {metadata}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 10 embedding values: {embedding[:10]}")
        print(f"Document:\n{document}")


if __name__ == "__main__":
    main()