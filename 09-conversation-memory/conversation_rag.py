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
CHAT_MODEL = "ministral-3:3b"

TOP_K = 4


def load_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def rewrite_with_context(
    conversation_history,
    latest_question,
):
    history_text = ""
    recent_history = conversation_history[-6:]

    for message in recent_history:
        history_text += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )

    prompt = f"""
Rewrite the latest user question into a standalone
search query for retrieving passages from technical AI papers.

Use the conversation history only to resolve references
such as "it", "they", "that", "those", or omitted subjects.

Rules:
- Preserve the user's intent.
- Do not answer the question.
- Do not invent facts.
- Do not add unrelated technical concepts.
- Resolve vague references only when the conversation
  history clearly identifies them.
- Return only the standalone rewritten query.

Conversation history:
{history_text}

Latest user question:
{latest_question}
""".strip()

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 120,
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

    for i in range(len(results["ids"][0])):
        records.append(
            {
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return records


def build_context(records):
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


def answer_question(
    latest_question,
    records,
):
    context = build_context(records)

    prompt = f"""
Use only the retrieved sources below to answer the question.

RETRIEVED SOURCES
-----------------
{context}

QUESTION
--------
{latest_question}

Rules:
- Do not use outside knowledge.
- If the sources do not contain enough information,
  say so.
- Cite sources using:
  [filename, page N, chunk N]
""".strip()

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 500,
        },
    )

    return response["message"]["content"].strip()


def print_results(records):
    print("\nRetrieved chunks:")

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


def main():
    collection = load_collection()

    conversation_history = []

    print("Conversation-aware RAG")
    print("Type 'quit' to exit.")

    while True:
        user_question = input("\nYou: ").strip()

        if user_question.lower() in {
            "quit",
            "exit",
        }:
            break

        standalone_query = rewrite_with_context(
            conversation_history,
            user_question,
        )

        print("\nStandalone retrieval query:")
        print(standalone_query)

        records = retrieve(
            standalone_query,
            collection,
        )

        print_results(records)

        answer = answer_question(
            user_question,
            records,
        )

        print("\nAssistant:")
        print(answer)

        conversation_history.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        conversation_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


if __name__ == "__main__":
    main()