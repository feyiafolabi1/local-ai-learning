import os
import time
from statistics import mean, median
from pathlib import Path

from dotenv import load_dotenv

from typesafe_sdk import (
    Choice,
    TypeSafeClient,
    TypeSafeAPIError,
)

from router_test_cases import (
    ROUTES,
    TEST_CASES,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

MODEL = "jev-1.13.0"

PRICE_PER_MILLION_INPUT_TOKENS = 0.042


# --------------------------------------------------
# ROUTING CRITERIA
# --------------------------------------------------

ROUTE_CRITERIA = {

    "CHAT": (
        "General AI, ML, GPU, or engineering questions "
        "answerable from general knowledge. "
        "A technical topic alone does not imply "
        "repository retrieval or benchmark analysis."
    ),

    "RAG": (
        "Requires information from the user's "
        "local-ai-learning repository or previous "
        "project work. Includes what they learned, "
        "measured, implemented, wrote, concluded, "
        "or which exact prompt or file they used."
    ),

    "MULTIMODAL": (
        "Requires interpreting a visual input such "
        "as an image, screenshot, graph, diagram, "
        "or profiler view. Select this route when "
        "the user asks what a visual shows, even "
        "if the image has not yet been attached."
    ),

    "TOOL": (
        "Primarily requires a deterministic "
        "calculation, percentage, unit conversion, "
        "or system information tool."
    ),

    "BENCHMARK_ANALYSIS": (
        "Requires interpreting performance "
        "measurements, benchmark results, scaling, "
        "latency, TTFT, TPOT, throughput, or "
        "bottlenecks. General GPU explanations "
        "without measurements belong to CHAT."
    ),
}


# --------------------------------------------------
# ROUTING INSTRUCTIONS
# --------------------------------------------------

ROUTING_INSTRUCTIONS = """
Select the single primary handler required to answer
the user's request.

Route based on the information or capability required,
not simply the technical topic.

Important distinctions:

- General technical explanations use CHAT.

- Questions about the user's own repository,
  experiments, or project history use RAG.

- Questions requiring visual interpretation use
  MULTIMODAL, including benchmark graphs.

- Deterministic calculations and system operations
  use TOOL.

- Interpretation of performance measurements
  or bottlenecks uses BENCHMARK_ANALYSIS.

Examples:

"What is grouped-query attention?"
CHAT

"What did I learn about grouped-query attention?"
RAG

"Why are GPUs good for matrix multiplication?"
CHAT

"What does this benchmark graph show?"
MULTIMODAL

"My TTFT increased 19x while TPOT barely changed.
What does this tell me?"
BENCHMARK_ANALYSIS

"Calculate the percentage increase from 106 to 574."
TOOL
"""


# --------------------------------------------------
# PERCENTILE HELPER
# --------------------------------------------------

def percentile(values, percentage):

    if not values:
        return None

    sorted_values = sorted(values)

    position = (
        (len(sorted_values) - 1)
        * percentage / 100
    )

    lower = int(position)

    upper = min(
        lower + 1,
        len(sorted_values) - 1,
    )

    fraction = position - lower

    return (
        sorted_values[lower]
        + fraction * (
            sorted_values[upper]
            - sorted_values[lower]
        )
    )


# --------------------------------------------------
# SINGLE JEV REQUEST
# --------------------------------------------------

def route_query(client, query):

    start = time.perf_counter()

    response = client.system_one(
        state={
            "user_request": query,
        },
        questions={
            "route": Choice(
                instructions=ROUTING_INSTRUCTIONS,
                criteria=ROUTE_CRITERIA,
            ),
        },
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    decision = response.answers["route"]

    predicted = decision.choice

    confidence = decision.confidence

    probabilities = decision.probabilities

    input_tokens = response.usage.input_tokens

    estimated_cost = (
        input_tokens
        / 1_000_000
        * PRICE_PER_MILLION_INPUT_TOKENS
    )

    return {
        "predicted": predicted,
        "confidence": confidence,
        "probabilities": probabilities,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "estimated_cost": estimated_cost,
    }


# --------------------------------------------------
# BENCHMARK
# --------------------------------------------------

def run_benchmark():

    if not os.getenv("TYPESAFE_API_KEY"):
        raise RuntimeError(
            "TYPESAFE_API_KEY missing from .env"
        )

    print()
    print("=" * 80)
    print("JEV ROUTER BENCHMARK")
    print("=" * 80)

    print()
    print(f"Model: {MODEL}")
    print(f"Test cases: {len(TEST_CASES)}")

    correct = 0
    invalid = 0
    errors = 0

    latencies = []
    confidences = []
    total_input_tokens = 0
    total_cost = 0.0

    failures = []

    print()

    # Reuse ONE client for the entire benchmark.
    with TypeSafeClient(model=MODEL) as client:

        for index, case in enumerate(
            TEST_CASES,
            start=1,
        ):

            query = case["query"]

            expected = case["expected_route"]

            try:

                result = route_query(
                    client,
                    query,
                )

            except TypeSafeAPIError as error:

                errors += 1

                failures.append({
                    "query": query,
                    "expected": expected,
                    "predicted": "API_ERROR",
                })

                print(
                    f"[{index:02d}] ERROR | "
                    f"HTTP {getattr(error, 'status', 'unknown')}"
                )

                continue

            except Exception as error:

                errors += 1

                failures.append({
                    "query": query,
                    "expected": expected,
                    "predicted": "ERROR",
                })

                print(
                    f"[{index:02d}] ERROR | "
                    f"{type(error).__name__}"
                )

                continue

            predicted = result["predicted"]

            is_valid = predicted in ROUTES

            is_correct = (
                is_valid
                and predicted == expected
            )

            if is_correct:
                correct += 1

            if not is_valid:
                invalid += 1

            if not is_correct:

                failures.append({
                    "query": query,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                })

            latencies.append(
                result["latency_ms"]
            )

            confidences.append(
                result["confidence"]
            )

            total_input_tokens += (
                result["input_tokens"]
            )

            total_cost += (
                result["estimated_cost"]
            )

            status = (
                "PASS"
                if is_correct
                else "FAIL"
            )

            print(
                f"[{index:02d}] {status} | "
                f"Expected: {expected:<20} "
                f"Predicted: {str(predicted):<20} "
                f"Latency: {result['latency_ms']:.2f} ms | "
                f"Confidence: {result['confidence']:.3f}"
            )

            print(f"     {query}")

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    total = len(TEST_CASES)

    accuracy = (
        correct / total * 100
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print()
    print(f"Correct: {correct}/{total}")

    print(f"Accuracy: {accuracy:.2f}%")

    print(f"Invalid outputs: {invalid}")

    print(f"API errors: {errors}")

    if latencies:

        print()
        print("LATENCY")

        print(
            f"Average latency: "
            f"{mean(latencies):.2f} ms"
        )

        print(
            f"Median latency: "
            f"{median(latencies):.2f} ms"
        )

        print(
            f"P95 latency: "
            f"{percentile(latencies, 95):.2f} ms"
        )

        print(
            f"Fastest request: "
            f"{min(latencies):.2f} ms"
        )

        print(
            f"Slowest request: "
            f"{max(latencies):.2f} ms"
        )

    if confidences:

        print()
        print("CONFIDENCE")

        print(
            f"Mean confidence: "
            f"{mean(confidences):.4f}"
        )

        print(
            f"Median confidence: "
            f"{median(confidences):.4f}"
        )

    print()
    print("USAGE AND COST")

    print(
        f"Total input tokens: "
        f"{total_input_tokens}"
    )

    print(
        f"Estimated total cost: "
        f"${total_cost:.8f}"
    )

    if latencies:

        print(
            f"Estimated cost per successful API request: "
            f"${total_cost / len(latencies):.8f}"
        )

    # --------------------------------------------------
    # FAILURE DETAILS
    # --------------------------------------------------

    if failures:

        print()
        print("=" * 80)
        print("FAILURES")
        print("=" * 80)

        for failure in failures:

            print()
            print(f"Query: {failure['query']}")

            print(
                f"Expected: {failure['expected']}"
            )

            print(
                f"Predicted: {failure['predicted']}"
            )

            if "confidence" in failure:

                print(
                    f"Confidence: "
                    f"{failure['confidence']:.4f}"
                )

                print("Probabilities:")

                for route, probability in sorted(
                    failure["probabilities"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):

                    print(
                        f"  {route:<22} "
                        f"{probability:.4f}"
                    )

    return {
        "model": MODEL,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "invalid_outputs": invalid,
        "errors": errors,
        "average_latency_ms": (
            mean(latencies)
            if latencies
            else None
        ),
        "median_latency_ms": (
            median(latencies)
            if latencies
            else None
        ),
        "p95_latency_ms": (
            percentile(latencies, 95)
            if latencies
            else None
        ),
        "total_input_tokens": total_input_tokens,
        "estimated_total_cost": total_cost,
        "failures": failures,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    run_benchmark()


if __name__ == "__main__":
    main()