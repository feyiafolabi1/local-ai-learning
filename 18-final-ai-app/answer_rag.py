import time

import ollama

from rag import (
    RAGEngine,
    format_context,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

GENERATION_MODEL = "ministral-3:3b"

RAG_TOP_K = 4


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an AI engineering learning assistant.

You answer questions using only the supplied context from the user's
local-ai-learning repository.

Your job is to faithfully summarize what the user actually learned,
measured, implemented, or concluded.

GROUNDING RULES

1. Every factual claim must be supported by the supplied repository
   context.

2. Do not add outside technical knowledge unless the user explicitly
   asks for it.

3. Do not infer a cause, relationship, or explanation that is not
   explicitly supported by the supplied context.

4. If the repository reports a surprising or counterintuitive result,
   preserve that result exactly rather than trying to make it fit a
   more familiar explanation.

5. Do not use phrases such as:
   - "likely showed"
   - "probably means"
   - "must have been"
   - "this suggests"
   unless the supplied source itself supports that interpretation.

6. Clearly distinguish between:
   - an observed experimental result
   - a conclusion written in the repository
   - a possible explanation

7. Only include a possible explanation if the supplied repository
   context explicitly supports it.

8. If the context is insufficient to answer part of the question,
   say that the repository context does not establish that point.

9. Preserve important benchmark numbers when they are relevant.

10. Cite retrieved material using:
    [Source 1], [Source 2], etc.

11. Do not claim that a source says something unless it actually does.

12. Prefer a concise synthesis over copying source text.

Before producing the final answer, silently check each factual claim:

"Can this claim be directly supported by one of the supplied sources?"

If the answer is no, remove or qualify the claim.
"""


# --------------------------------------------------
# PROMPT BUILDING
# --------------------------------------------------

def build_user_prompt(
    query: str,
    context: str,
) -> str:

    return f"""
REPOSITORY CONTEXT

{context}


USER QUESTION

{query}


Answer using only the repository context above.

Important:
- Report observed results exactly as written.
- Do not add unsupported explanations.
- Do not convert correlations or observations into causes.
- If a result is counterintuitive, preserve it rather than correcting it.
- Cite factual claims using [Source 1], [Source 2], etc.
""".strip()


# --------------------------------------------------
# GENERATION
# --------------------------------------------------

def generate_answer(
    query: str,
    context: str,
):

    prompt = build_user_prompt(
        query,
        context,
    )

    start = time.perf_counter()

    response = ollama.chat(
        model=GENERATION_MODEL,
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
            "temperature": 0.1,
        },
    )

    generation_latency = (
        time.perf_counter()
        - start
    )

    answer = response[
        "message"
    ][
        "content"
    ]

    metrics = {
        "generation_latency_ms":
            generation_latency
            * 1000,

        "prompt_tokens":
            getattr(
                response,
                "prompt_eval_count",
                None,
            ),

        "output_tokens":
            getattr(
                response,
                "eval_count",
                None,
            ),

        "prompt_eval_duration_ns":
            getattr(
                response,
                "prompt_eval_duration",
                None,
            ),

        "eval_duration_ns":
            getattr(
                response,
                "eval_duration",
                None,
            ),
    }

    return (
        answer,
        metrics,
    )


# --------------------------------------------------
# METRIC HELPERS
# --------------------------------------------------

def calculate_generation_metrics(
    generation_metrics,
):

    output_tokens = generation_metrics[
        "output_tokens"
    ]

    eval_duration_ns = generation_metrics[
        "eval_duration_ns"
    ]

    tokens_per_second = None

    if (
        output_tokens is not None
        and eval_duration_ns is not None
        and eval_duration_ns > 0
    ):

        eval_seconds = (
            eval_duration_ns
            / 1_000_000_000
        )

        tokens_per_second = (
            output_tokens
            / eval_seconds
        )

    return {
        "tokens_per_second":
            tokens_per_second,
    }


# --------------------------------------------------
# SOURCE DISPLAY
# --------------------------------------------------

def print_sources(
    results,
):

    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result[
            "metadata"
        ]

        print()

        print(
            f"[Source {index}]"
        )

        print(
            f"Chapter: "
            f"{metadata['chapter']}"
        )

        print(
            f"File: "
            f"{metadata['source_path']}"
        )

        print(
            f"Section: "
            f"{metadata['section_name']}"
        )

        print(
            f"Vector rank: "
            f"{result['vector_rank']}"
        )

        print(
            f"Reranked rank: "
            f"{result['reranked_rank']}"
        )

        print(
            f"Final rank: "
            f"{result['final_rank']}"
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


# --------------------------------------------------
# METRIC DISPLAY
# --------------------------------------------------

def print_metrics(
    rag_metrics,
    generation_metrics,
    derived_metrics,
):

    print()
    print("=" * 80)
    print("OBSERVABILITY")
    print("=" * 80)

    print()

    print(
        f"Retrieval latency: "
        f"{rag_metrics['retrieval_latency_ms']:.2f} ms"
    )

    print(
        f"Reranking latency: "
        f"{rag_metrics['reranking_latency_ms']:.2f} ms"
    )

    print(
        f"Generation latency: "
        f"{generation_metrics['generation_latency_ms']:.2f} ms"
    )

    total_pipeline_latency = (
        rag_metrics[
            "retrieval_latency_ms"
        ]
        +
        rag_metrics[
            "reranking_latency_ms"
        ]
        +
        generation_metrics[
            "generation_latency_ms"
        ]
    )

    print(
        f"Total pipeline latency: "
        f"{total_pipeline_latency:.2f} ms"
    )

    print()

    print(
        f"Candidates retrieved: "
        f"{rag_metrics['candidates_retrieved']}"
    )

    print(
        f"Chunks selected: "
        f"{rag_metrics['chunks_selected']}"
    )

    if "chunks_filtered_out" in rag_metrics:

        print(
            f"Chunks filtered out: "
            f"{rag_metrics['chunks_filtered_out']}"
        )

    print()

    prompt_tokens = generation_metrics[
        "prompt_tokens"
    ]

    output_tokens = generation_metrics[
        "output_tokens"
    ]

    if prompt_tokens is not None:

        print(
            f"Prompt tokens: "
            f"{prompt_tokens}"
        )

    if output_tokens is not None:

        print(
            f"Output tokens: "
            f"{output_tokens}"
        )

    tokens_per_second = derived_metrics[
        "tokens_per_second"
    ]

    if tokens_per_second is not None:

        print(
            f"Generation throughput: "
            f"{tokens_per_second:.2f} tok/s"
        )


# --------------------------------------------------
# COMPLETE RAG PIPELINE
# --------------------------------------------------

def answer_question(
    rag: RAGEngine,
    query: str,
):

    # ----------------------------------------------
    # RETRIEVE
    # ----------------------------------------------

    results, rag_metrics = rag.search(
        query,
        top_k=RAG_TOP_K,
    )

    # ----------------------------------------------
    # BUILD CONTEXT
    # ----------------------------------------------

    context = format_context(
        results
    )

    # ----------------------------------------------
    # GENERATE
    # ----------------------------------------------

    (
        answer,
        generation_metrics,
    ) = generate_answer(
        query,
        context,
    )

    derived_metrics = (
        calculate_generation_metrics(
            generation_metrics
        )
    )

    return {
        "query":
            query,

        "answer":
            answer,

        "sources":
            results,

        "rag_metrics":
            rag_metrics,

        "generation_metrics":
            generation_metrics,

        "derived_metrics":
            derived_metrics,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print()
    print("=" * 80)
    print("LOCAL AI ENGINEERING COPILOT — RAG TEST")
    print("=" * 80)

    print()
    print(
        f"Generation model: "
        f"{GENERATION_MODEL}"
    )

    print()

    rag = RAGEngine()

    print()
    print("=" * 80)
    print("READY")
    print("=" * 80)

    while True:

        print()

        query = input(
            "Ask about your local-ai-learning repo\n"
            "(or type 'quit'): "
        ).strip()

        if not query:
            continue

        if query.lower() in {
            "quit",
            "exit",
            "q",
        }:
            break

        print()
        print("Running RAG pipeline...")

        result = answer_question(
            rag,
            query,
        )

        print()
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print()
        print(
            result["answer"]
        )

        print_sources(
            result["sources"]
        )

        print_metrics(
            result["rag_metrics"],
            result[
                "generation_metrics"
            ],
            result[
                "derived_metrics"
            ],
        )


if __name__ == "__main__":
    main()