import time
from statistics import median

import ollama

from router_test_cases import (
    ROUTES,
    TEST_CASES,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

ROUTER_MODEL = "ministral-3:3b"


# --------------------------------------------------
# ROUTER PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a routing classifier for an AI engineering copilot.

You must classify each user request into exactly one route.

Valid routes:

CHAT
RAG
MULTIMODAL
TOOL
BENCHMARK_ANALYSIS


CORE PRINCIPLE

Route based on what information or capability is REQUIRED
to answer the user's request.

Do not route based only on the technical topic being mentioned.


ROUTE DEFINITIONS

CHAT
- General AI, ML, GPU, systems, or engineering explanation.
- The question can be answered from general knowledge.
- It does not require the user's repository, personal experiment history,
  an image, or a deterministic tool.

Examples:

"What is grouped-query attention?"
→ CHAT

"Why are GPUs good for matrix multiplication?"
→ CHAT

"What does arithmetic intensity mean?"
→ CHAT


RAG
- The request requires information from the user's
  local-ai-learning repository or their previous project work.
- Look for explicit references to what the user learned, measured,
  implemented, wrote, used, concluded, or did in a chapter or experiment.

Strong RAG signals include:

"what did I learn"
"what did I use"
"what was my result"
"my benchmark"
"my implementation"
"in Chapter 15"
"in my repo"
"what exact prompt did I use"
"what file did I create"

Examples:

"What did I learn about KV cache during inference?"
→ RAG

"What exact benchmark prompt did I use for KV cache?"
→ RAG

"What throughput did I get in Chapter 17?"
→ RAG


MULTIMODAL
- The answer requires seeing or understanding visual input.
- Includes images, screenshots, diagrams, charts, graphs,
  profiler screenshots, and terminal screenshots.

If the user asks about what a visible graph or screenshot shows,
MULTIMODAL takes precedence over benchmark analysis.

Examples:

"What does this benchmark graph show?"
→ MULTIMODAL

"Read this terminal screenshot and tell me why it failed."
→ MULTIMODAL

"Explain this GPU profiler screenshot."
→ MULTIMODAL


TOOL
- The request primarily requires a deterministic calculation,
  conversion, lookup from a system tool, or similar operation.

Examples:

"Calculate the percentage increase from 106 to 574."
→ TOOL

"Convert 8192 MB to GB."
→ TOOL

"Show me my current system memory information."
→ TOOL


BENCHMARK_ANALYSIS
- The request asks to interpret supplied or referenced performance
  measurements, benchmark behavior, scaling, or bottlenecks.
- The important distinction is that actual benchmark results or
  performance behavior are being analyzed.

Common signals include:

TTFT
TPOT
throughput
latency
concurrency
scaling
memory-bound
compute-bound
benchmark results
performance numbers

Examples:

"My TTFT increased by 19x while TPOT barely changed.
What does that tell me about the bottleneck?"
→ BENCHMARK_ANALYSIS

"Why can latency increase while throughput also improves?"
→ BENCHMARK_ANALYSIS

"My 4096 matrix multiplication was 2.77x faster on the GPU.
Why?"
→ BENCHMARK_ANALYSIS


IMPORTANT DISTINCTIONS

GENERAL TOPIC vs RAG

"What is grouped-query attention?"
→ CHAT

"What did I learn about grouped-query attention?"
→ RAG


GENERAL TOPIC vs BENCHMARK_ANALYSIS

"Why are GPUs good for matrix multiplication?"
→ CHAT

"My GPU matrix multiplication was 2.77x faster than CPU.
Why did that happen?"
→ BENCHMARK_ANALYSIS


RAG vs BENCHMARK_ANALYSIS

"What throughput did I get in Chapter 17?"
→ RAG

"Why did throughput increase as concurrency increased?"
→ BENCHMARK_ANALYSIS


MULTIMODAL vs BENCHMARK_ANALYSIS

"What does this benchmark graph show?"
→ MULTIMODAL

"My throughput rose from 106 to 574 tokens/sec as concurrency increased.
Why?"
→ BENCHMARK_ANALYSIS


ROUTING PRECEDENCE

Use these rules:

1. If visual input must be seen or understood:
   → MULTIMODAL

2. Else if actual benchmark measurements or performance behavior
   must be interpreted:
   → BENCHMARK_ANALYSIS

3. Else if the user's own repository/project history is required:
   → RAG

4. Else if a deterministic calculation or tool is required:
   → TOOL

5. Else:
   → CHAT


OUTPUT RULE

Return exactly one route name and nothing else.

Valid outputs:

CHAT
RAG
MULTIMODAL
TOOL
BENCHMARK_ANALYSIS
"""


# --------------------------------------------------
# ROUTING
# --------------------------------------------------

def route_query(
    query: str,
):

    start = time.perf_counter()

    response = ollama.chat(
        model=ROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        options={
            "temperature": 0.0,
        },
    )

    latency = (
        time.perf_counter()
        - start
    )

    raw_output = (
        response["message"]["content"]
        .strip()
        .upper()
    )

    return (
        raw_output,
        latency,
    )


# --------------------------------------------------
# OUTPUT CLEANUP
# --------------------------------------------------

def normalize_route(
    raw_output: str,
):

    cleaned = (
        raw_output
        .strip()
        .upper()
    )

    if cleaned in ROUTES:
        return cleaned

    for route in ROUTES:

        if route in cleaned:
            return route

    return "INVALID"


# --------------------------------------------------
# BENCHMARK
# --------------------------------------------------

def run_benchmark():

    print()
    print("=" * 80)
    print("LOCAL LLM ROUTER BENCHMARK")
    print("=" * 80)

    print()
    print(
        f"Router model: "
        f"{ROUTER_MODEL}"
    )

    print(
        f"Test cases: "
        f"{len(TEST_CASES)}"
    )

    correct = 0
    invalid = 0

    latencies = []
    failures = []

    print()

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        query = case["query"]

        expected = (
            case["expected_route"]
        )

        raw_output, latency = (
            route_query(
                query
            )
        )

        predicted = normalize_route(
            raw_output
        )

        is_correct = (
            predicted == expected
        )

        if is_correct:
            correct += 1

        else:
            failures.append(
                {
                    "query": query,
                    "expected": expected,
                    "predicted": predicted,
                    "raw_output": raw_output,
                }
            )

        if predicted == "INVALID":
            invalid += 1

        latencies.append(
            latency
        )

        status = (
            "PASS"
            if is_correct
            else "FAIL"
        )

        print(
            f"[{index:02d}] "
            f"{status} | "
            f"Expected: {expected:<20} "
            f"Predicted: {predicted:<20} "
            f"Latency: {latency * 1000:.2f} ms"
        )

        print(
            f"     {query}"
        )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    accuracy = (
        correct
        / len(TEST_CASES)
        * 100
    )

    latency_ms = [
        latency * 1000
        for latency in latencies
    ]

    average_latency_ms = (
        sum(latency_ms)
        / len(latency_ms)
    )

    median_latency_ms = median(
        latency_ms
    )

    minimum_latency_ms = min(
        latency_ms
    )

    maximum_latency_ms = max(
        latency_ms
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Correct: "
        f"{correct}/{len(TEST_CASES)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        f"Invalid outputs: "
        f"{invalid}"
    )

    print()

    print(
        f"Average latency: "
        f"{average_latency_ms:.2f} ms"
    )

    print(
        f"Median latency: "
        f"{median_latency_ms:.2f} ms"
    )

    print(
        f"Fastest request: "
        f"{minimum_latency_ms:.2f} ms"
    )

    print(
        f"Slowest request: "
        f"{maximum_latency_ms:.2f} ms"
    )

    # --------------------------------------------------
    # FAILURES
    # --------------------------------------------------

    if failures:

        print()
        print("=" * 80)
        print("FAILURES")
        print("=" * 80)

        for failure in failures:

            print()

            print(
                f"Query: "
                f"{failure['query']}"
            )

            print(
                f"Expected: "
                f"{failure['expected']}"
            )

            print(
                f"Predicted: "
                f"{failure['predicted']}"
            )

            print(
                f"Raw output: "
                f"{failure['raw_output']}"
            )

    return {
        "model":
            ROUTER_MODEL,

        "accuracy":
            accuracy,

        "correct":
            correct,

        "total":
            len(TEST_CASES),

        "invalid_outputs":
            invalid,

        "average_latency_ms":
            average_latency_ms,

        "median_latency_ms":
            median_latency_ms,

        "minimum_latency_ms":
            minimum_latency_ms,

        "maximum_latency_ms":
            maximum_latency_ms,

        "failures":
            failures,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    run_benchmark()


if __name__ == "__main__":
    main()