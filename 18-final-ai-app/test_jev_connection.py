import os
import time
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import (
    Choice,
    TypeSafeClient,
    TypeSafeAPIError,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

MODEL = "jev-1.13.0"

QUERY = (
    "What did I learn about KV cache "
    "during inference?"
)

EXPECTED_ROUTE = "RAG"

PRICE_PER_MILLION_INPUT_TOKENS = 0.042


# --------------------------------------------------
# ROUTING OPTIONS
# --------------------------------------------------

ROUTE_CRITERIA = {
    "CHAT": (
        "General AI or engineering explanation "
        "that can be answered from general knowledge. "
        "No personal project lookup or benchmark "
        "interpretation is required."
    ),

    "RAG": (
        "Requires facts from the user's "
        "local-ai-learning repository, including "
        "what they learned, implemented, measured, "
        "wrote, or concluded."
    ),

    "MULTIMODAL": (
        "Requires understanding a screenshot, "
        "image, graph, diagram, or other visual input."
    ),

    "TOOL": (
        "Primarily requires a deterministic "
        "calculation, conversion, or system tool."
    ),

    "BENCHMARK_ANALYSIS": (
        "Requires interpreting performance "
        "measurements, scaling, bottlenecks, "
        "latency, TTFT, TPOT, or throughput. "
        "Not merely a general GPU explanation."
    ),
}


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not os.getenv("TYPESAFE_API_KEY"):
        raise RuntimeError(
            "TYPESAFE_API_KEY was not found in .env"
        )

    print()
    print("=" * 70)
    print("JEV CONNECTION TEST")
    print("=" * 70)

    print()
    print(f"Model: {MODEL}")
    print(f"Query: {QUERY}")
    print(f"Expected route: {EXPECTED_ROUTE}")

    print()
    print("Sending request to TypeSafe...")

    start = time.perf_counter()

    try:

        with TypeSafeClient(model=MODEL) as client:

            response = client.system_one(
                state={
                    "user_request": QUERY,
                    "has_image": False,
                },
                questions={
                    "route": Choice(
                        instructions=(
                            "Choose the single primary handler "
                            "required to answer this user request. "
                            "Route according to the capability "
                            "needed, not just the technical topic. "
                            "General explanations use CHAT. "
                            "Personal project history uses RAG. "
                            "Visual interpretation uses MULTIMODAL. "
                            "Deterministic calculations use TOOL. "
                            "Performance measurement interpretation "
                            "uses BENCHMARK_ANALYSIS."
                        ),
                        criteria=ROUTE_CRITERIA,
                    )
                },
            )

    except TypeSafeAPIError as error:

        print()
        print("TypeSafe API request failed.")
        print(f"HTTP status: {error.status}")
        print(f"Request ID: {error.request_id}")
        return

    except Exception as error:

        print()
        print("Request failed.")
        print(f"Error type: {type(error).__name__}")
        return

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    # ----------------------------------------------
    # READ DECISION
    # ----------------------------------------------

    decision = response.answers["route"]

    predicted_route = decision.choice
    confidence = decision.confidence
    probabilities = decision.probabilities

    # ----------------------------------------------
    # TOKEN USAGE / COST
    # ----------------------------------------------

    input_tokens = response.usage.input_tokens

    cost = None

    if input_tokens is not None:

        cost = (
            input_tokens
            / 1_000_000
            * PRICE_PER_MILLION_INPUT_TOKENS
        )

    # ----------------------------------------------
    # DISPLAY
    # ----------------------------------------------

    print()
    print("=" * 70)
    print("JEV RESPONSE")
    print("=" * 70)

    print()
    print(f"Model used:       {response.model}")
    print(f"Selected route:   {predicted_route}")
    print(f"Expected route:   {EXPECTED_ROUTE}")
    print(f"Confidence:       {confidence:.4f}")

    print()
    print("ROUTE PROBABILITIES")

    for route, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"{route:<22} {probability:.4f}"
        )

    print()
    print("=" * 70)
    print("PERFORMANCE")
    print("=" * 70)

    print()
    print(f"API latency: {latency_ms:.2f} ms")
    print(f"Input tokens: {input_tokens}")

    if cost is not None:
        print(f"Estimated API cost: ${cost:.8f}")

    print()
    print("=" * 70)

    if predicted_route == EXPECTED_ROUTE:

        print("RESULT: PASS")

    else:

        print("RESULT: FAIL")


if __name__ == "__main__":
    main()