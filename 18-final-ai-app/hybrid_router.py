import os
import sys
import time
from pathlib import Path
from statistics import mean, median

import ollama

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from router_test_cases import (
    ROUTES,
    TEST_CASES,
)

from router import (
    ROUTER_MODEL,
    SYSTEM_PROMPT,
    normalize_route,
)

from jev_router import (
    MODEL as JEV_MODEL,
    route_query as jev_route_query,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

CONFIDENCE_THRESHOLD = 0.70

DIAGNOSTIC_QUERY = (
    "Why can latency increase while "
    "overall throughput also improves?"
)

DIAGNOSTIC_EXPECTED = "BENCHMARK_ANALYSIS"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def ns_to_ms(value):
    """
    Ollama reports internal durations in nanoseconds.

    Convert nanoseconds to milliseconds.
    """

    if value is None:
        return None

    return value / 1_000_000


def format_ms(value):

    if value is None:
        return "N/A"

    return f"{value:.2f} ms"


def percentile(values, percentage):

    if not values:
        return None

    values = sorted(values)

    position = (
        (len(values) - 1)
        * percentage / 100
    )

    lower = int(position)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = position - lower

    return (
        values[lower]
        + fraction * (
            values[upper] - values[lower]
        )
    )


# --------------------------------------------------
# LOCAL MINISTRAL ROUTER
# --------------------------------------------------

def run_local_fallback(query):
    """
    Run the same Ministral classification as router.py,
    but also capture Ollama's internal timing metrics.

    We are NOT changing the model or routing prompt.
    """

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

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    raw_output = (
        response["message"]["content"]
        .strip()
        .upper()
    )

    predicted = normalize_route(
        raw_output
    )

    # Ollama's server-side timing fields.
    ollama_total_ms = ns_to_ms(
        getattr(
            response,
            "total_duration",
            None,
        )
    )

    load_ms = ns_to_ms(
        getattr(
            response,
            "load_duration",
            None,
        )
    )

    prompt_eval_ms = ns_to_ms(
        getattr(
            response,
            "prompt_eval_duration",
            None,
        )
    )

    eval_ms = ns_to_ms(
        getattr(
            response,
            "eval_duration",
            None,
        )
    )

    # Difference between our Python wall-clock timing
    # and Ollama's reported total duration.
    #
    # This can include client/server communication
    # and other overhead. It is not necessarily
    # network transfer alone.
    other_ms = None

    if ollama_total_ms is not None:

        other_ms = (
            latency_ms - ollama_total_ms
        )

    return {
        "predicted": predicted,
        "raw_output": raw_output,
        "latency_ms": latency_ms,

        "ollama_total_ms": ollama_total_ms,
        "load_ms": load_ms,
        "prompt_eval_ms": prompt_eval_ms,
        "eval_ms": eval_ms,
        "other_ms": other_ms,

        "prompt_tokens": getattr(
            response,
            "prompt_eval_count",
            None,
        ),

        "output_tokens": getattr(
            response,
            "eval_count",
            None,
        ),
    }


# --------------------------------------------------
# LOCAL TIMING DISPLAY
# --------------------------------------------------

def print_local_timing(result):

    print()
    print("MINISTRAL TIMING BREAKDOWN")
    print("-" * 55)

    print(
        f"Python wall-clock:    "
        f"{format_ms(result['latency_ms'])}"
    )

    print(
        f"Ollama total:         "
        f"{format_ms(result['ollama_total_ms'])}"
    )

    print(
        f"Model loading:        "
        f"{format_ms(result['load_ms'])}"
    )

    print(
        f"Prompt processing:    "
        f"{format_ms(result['prompt_eval_ms'])}"
    )

    print(
        f"Token generation:     "
        f"{format_ms(result['eval_ms'])}"
    )

    print(
        f"Other client overhead:"
        f" {format_ms(result['other_ms'])}"
    )

    print()
    print(
        f"Prompt tokens:        "
        f"{result['prompt_tokens']}"
    )

    print(
        f"Output tokens:        "
        f"{result['output_tokens']}"
    )

    print(
        f"Predicted route:      "
        f"{result['predicted']}"
    )


# --------------------------------------------------
# HYBRID ROUTING
# --------------------------------------------------

def hybrid_route(
    client,
    query,
    force_fallback=False,
):

    total_start = time.perf_counter()

    jev_result = None
    jev_error = None
    local_result = None

    jev_latency_ms = 0.0

    confidence = None
    fallback_reason = None

    input_tokens = 0
    estimated_cost = 0.0

    # ----------------------------------------------
    # STEP 1: JEV
    # ----------------------------------------------

    jev_start = time.perf_counter()

    try:

        jev_result = jev_route_query(
            client,
            query,
        )

        jev_latency_ms = (
            jev_result["latency_ms"]
        )

        confidence = (
            jev_result["confidence"]
        )

        input_tokens = (
            jev_result["input_tokens"]
        )

        estimated_cost = (
            jev_result["estimated_cost"]
        )

    except Exception as error:

        jev_latency_ms = (
            time.perf_counter() - jev_start
        ) * 1000

        jev_error = type(error).__name__

    # ----------------------------------------------
    # STEP 2: CONFIDENCE CHECK
    # ----------------------------------------------

    if jev_error is not None:

        fallback_reason = (
            f"JEV_API_ERROR: {jev_error}"
        )

    elif jev_result["predicted"] not in ROUTES:

        fallback_reason = (
            "INVALID_JEV_ROUTE"
        )

    elif (
        confidence is None
        or confidence < CONFIDENCE_THRESHOLD
    ):

        fallback_reason = (
            "LOW_CONFIDENCE"
        )

    # Diagnostic mode can deliberately invoke
    # Ministral to measure it even if Jev is
    # sufficiently confident.
    if (
        force_fallback
        and fallback_reason is None
    ):

        fallback_reason = (
            "DIAGNOSTIC_FORCED_FALLBACK"
        )

    # ----------------------------------------------
    # STEP 3: FINAL DECISION
    # ----------------------------------------------

    if fallback_reason is None:

        final_route = jev_result["predicted"]

        router_used = "JEV"

    else:

        local_result = run_local_fallback(
            query
        )

        final_route = (
            local_result["predicted"]
        )

        router_used = (
            "MINISTRAL_FALLBACK"
        )

    # ----------------------------------------------
    # STEP 4: TOTAL LATENCY
    # ----------------------------------------------

    total_latency_ms = (
        time.perf_counter() - total_start
    ) * 1000

    return {
        "predicted": final_route,

        "router_used": router_used,

        "used_fallback":
            local_result is not None,

        "fallback_reason": fallback_reason,

        "jev_prediction": (
            jev_result["predicted"]
            if jev_result is not None
            else None
        ),

        "jev_confidence": confidence,

        "jev_latency_ms": jev_latency_ms,

        "local_latency_ms": (
            local_result["latency_ms"]
            if local_result is not None
            else 0.0
        ),

        "local_result": local_result,

        "total_latency_ms": total_latency_ms,

        "input_tokens": input_tokens,

        "estimated_cost": estimated_cost,

        "jev_error": jev_error,
    }


# --------------------------------------------------
# TARGETED DIAGNOSTIC
# --------------------------------------------------

def run_diagnostic():
    """
    Test the problematic fallback request.

    1. Call Jev.
    2. Run Ministral once.
    3. Run Ministral again immediately.
    4. Compare the two local calls.

    The first local call is only genuinely cold
    if the model was not already loaded.
    """

    if not os.getenv("TYPESAFE_API_KEY"):

        raise RuntimeError(
            "TYPESAFE_API_KEY missing from .env"
        )

    print()
    print("=" * 75)
    print("HYBRID ROUTER — FALLBACK DIAGNOSTIC")
    print("=" * 75)

    print()
    print(f"Jev model: {JEV_MODEL}")
    print(f"Local model: {ROUTER_MODEL}")

    print()
    print(f"Query: {DIAGNOSTIC_QUERY}")
    print(f"Expected: {DIAGNOSTIC_EXPECTED}")

    print()
    print("Running Jev + first Ministral call...")

    with TypeSafeClient(
        model=JEV_MODEL
    ) as client:

        first = hybrid_route(
            client,
            DIAGNOSTIC_QUERY,
            force_fallback=True,
        )

    print()
    print("=" * 75)
    print("FIRST REQUEST")
    print("=" * 75)

    print()
    print(
        f"Jev prediction: "
        f"{first['jev_prediction']}"
    )

    print(
        f"Jev confidence: "
        f"{first['jev_confidence']}"
    )

    print(
        f"Fallback reason: "
        f"{first['fallback_reason']}"
    )

    print()
    print(
        f"Jev latency:   "
        f"{format_ms(first['jev_latency_ms'])}"
    )

    print(
        f"Local latency: "
        f"{format_ms(first['local_latency_ms'])}"
    )

    print(
        f"Total latency: "
        f"{format_ms(first['total_latency_ms'])}"
    )

    print_local_timing(
        first["local_result"]
    )

    # ----------------------------------------------
    # SECOND LOCAL CALL
    # ----------------------------------------------

    print()
    print("=" * 75)
    print("SECOND MINISTRAL CALL — IMMEDIATE REPEAT")
    print("=" * 75)

    print()
    print("Running the same local query again...")

    second = run_local_fallback(
        DIAGNOSTIC_QUERY
    )

    print_local_timing(
        second
    )

    # ----------------------------------------------
    # COMPARISON
    # ----------------------------------------------

    first_local_ms = (
        first["local_latency_ms"]
    )

    second_local_ms = (
        second["latency_ms"]
    )

    print()
    print("=" * 75)
    print("COMPARISON")
    print("=" * 75)

    print()

    print(
        f"First Ministral:  "
        f"{first_local_ms:.2f} ms"
    )

    print(
        f"Second Ministral: "
        f"{second_local_ms:.2f} ms"
    )

    if second_local_ms > 0:

        ratio = (
            first_local_ms / second_local_ms
        )

        print(
            f"First/second ratio: "
            f"{ratio:.2f}x"
        )

    print()
    print(
        "Check the Ollama model-loading time "
        "to determine whether a cold start "
        "explains the difference."
    )


# --------------------------------------------------
# FULL HYBRID BENCHMARK
# --------------------------------------------------

def run_benchmark():

    if not os.getenv("TYPESAFE_API_KEY"):

        raise RuntimeError(
            "TYPESAFE_API_KEY missing from .env"
        )

    print()
    print("=" * 80)
    print("HYBRID ROUTER BENCHMARK")
    print("=" * 80)

    print()
    print(f"Jev model: {JEV_MODEL}")
    print(f"Local model: {ROUTER_MODEL}")

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Test cases: {len(TEST_CASES)}"
    )

    correct = 0
    invalid = 0
    fallback_count = 0
    fallback_correct = 0
    jev_errors = 0

    total_input_tokens = 0
    total_cost = 0.0

    latencies = []
    jev_latencies = []

    failures = []
    fallback_details = []

    with TypeSafeClient(
        model=JEV_MODEL
    ) as client:

        for index, case in enumerate(
            TEST_CASES,
            start=1,
        ):

            query = case["query"]
            expected = case["expected_route"]

            result = hybrid_route(
                client,
                query,
            )

            predicted = result["predicted"]

            is_correct = (
                predicted == expected
            )

            if is_correct:
                correct += 1

            if predicted not in ROUTES:
                invalid += 1

            if result["jev_error"] is not None:
                jev_errors += 1

            if result["used_fallback"]:

                fallback_count += 1

                if is_correct:
                    fallback_correct += 1

                fallback_details.append({
                    "query": query,
                    "expected": expected,
                    "jev_prediction":
                        result["jev_prediction"],
                    "final_prediction": predicted,
                    "confidence":
                        result["jev_confidence"],
                    "reason":
                        result["fallback_reason"],
                    "jev_latency_ms":
                        result["jev_latency_ms"],
                    "local_result":
                        result["local_result"],
                    "total_latency_ms":
                        result["total_latency_ms"],
                })

            if not is_correct:

                failures.append({
                    "query": query,
                    "expected": expected,
                    "predicted": predicted,
                })

            latencies.append(
                result["total_latency_ms"]
            )

            jev_latencies.append(
                result["jev_latency_ms"]
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

            print()
            print(
                f"[{index:02d}] {status} | "
                f"Expected: {expected:<20} "
                f"Predicted: {predicted:<20}"
            )

            print(
                f"Router: {result['router_used']}"
            )

            print(
                f"Total latency: "
                f"{result['total_latency_ms']:.2f} ms"
            )

            if result["used_fallback"]:

                print(
                    f"Fallback reason: "
                    f"{result['fallback_reason']}"
                )

                print(
                    f"Jev latency: "
                    f"{result['jev_latency_ms']:.2f} ms"
                )

                print(
                    f"Ministral latency: "
                    f"{result['local_latency_ms']:.2f} ms"
                )

    # ----------------------------------------------
    # SUMMARY
    # ----------------------------------------------

    total = len(TEST_CASES)

    accuracy = correct / total * 100

    fallback_rate = (
        fallback_count / total * 100
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print()
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Invalid outputs: {invalid}")
    print(f"Jev API errors: {jev_errors}")

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

    print()
    print(
        f"Average Jev latency: "
        f"{mean(jev_latencies):.2f} ms"
    )

    print()
    print("FALLBACK")

    print(
        f"Fallback requests: "
        f"{fallback_count}/{total}"
    )

    print(
        f"Fallback rate: "
        f"{fallback_rate:.2f}%"
    )

    print(
        f"Correct fallback decisions: "
        f"{fallback_correct}/{fallback_count}"
        if fallback_count
        else "Correct fallback decisions: N/A"
    )

    print()
    print("COST")

    print(
        f"Jev input tokens: "
        f"{total_input_tokens}"
    )

    print(
        f"Estimated Jev API cost: "
        f"${total_cost:.8f}"
    )

    # ----------------------------------------------
    # FALLBACK DETAILS
    # ----------------------------------------------

    if fallback_details:

        print()
        print("=" * 80)
        print("FALLBACK DETAILS")
        print("=" * 80)

        for item in fallback_details:

            print()
            print(f"Query: {item['query']}")

            print(
                f"Expected: {item['expected']}"
            )

            print(
                f"Jev prediction: "
                f"{item['jev_prediction']}"
            )

            print(
                f"Final prediction: "
                f"{item['final_prediction']}"
            )

            print(
                f"Confidence: {item['confidence']}"
            )

            print(
                f"Reason: {item['reason']}"
            )

            print(
                f"Jev latency: "
                f"{item['jev_latency_ms']:.2f} ms"
            )

            print(
                f"Total latency: "
                f"{item['total_latency_ms']:.2f} ms"
            )

            print_local_timing(
                item["local_result"]
            )

    # ----------------------------------------------
    # FAILURES
    # ----------------------------------------------

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


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if (
        len(sys.argv) > 1
        and sys.argv[1] == "--diagnose"
    ):

        run_diagnostic()

    else:

        run_benchmark()


if __name__ == "__main__":
    main()