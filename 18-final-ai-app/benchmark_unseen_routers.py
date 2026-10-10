import json
import os
import time

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from router_unseen_test_cases import (
    ROUTES,
    TEST_CASES,
    validate_test_cases,
)

from router import (
    ROUTER_MODEL,
    route_query as local_route_query,
    normalize_route,
)

from jev_router import (
    MODEL as JEV_MODEL,
    route_query as jev_route_query,
)

from hybrid_router import (
    CONFIDENCE_THRESHOLD,
    hybrid_route,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

OUTPUT_FILE = PROJECT_DIR / "router_unseen_results.json"


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

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
# MINISTRAL
# --------------------------------------------------

def run_ministral(query):

    raw_output, latency_seconds = local_route_query(
        query
    )

    return {
        "predicted": normalize_route(raw_output),
        "latency_ms": latency_seconds * 1000,
        "confidence": None,
        "used_fallback": False,
        "api_error": False,
        "cost_usd": 0.0,
    }


# --------------------------------------------------
# JEV
# --------------------------------------------------

def run_jev(client, query):

    result = jev_route_query(
        client,
        query,
    )

    return {
        "predicted": result["predicted"],
        "latency_ms": result["latency_ms"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "used_fallback": False,
        "api_error": False,
        "input_tokens": result["input_tokens"],
        "cost_usd": result["estimated_cost"],
    }


# --------------------------------------------------
# HYBRID
# --------------------------------------------------

def run_hybrid(client, query):

    result = hybrid_route(
        client,
        query,
    )

    return {
        "predicted": result["predicted"],
        "latency_ms": result["total_latency_ms"],
        "confidence": result["jev_confidence"],
        "used_fallback": result["used_fallback"],
        "fallback_reason": result["fallback_reason"],
        "jev_prediction": result["jev_prediction"],
        "jev_latency_ms": result["jev_latency_ms"],
        "local_latency_ms": result["local_latency_ms"],
        "api_error": result["jev_error"] is not None,
        "input_tokens": result["input_tokens"],
        "cost_usd": result["estimated_cost"],
    }


# --------------------------------------------------
# EVALUATE ONE ROUTER
# --------------------------------------------------

def evaluate_router(name, run_request):

    print()
    print("=" * 85)
    print(f"EVALUATING: {name}")
    print("=" * 85)

    records = []

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        query = case["query"]
        expected = case["expected_route"]

        start = time.perf_counter()

        try:

            result = run_request(query)

        except Exception as error:

            result = {
                "predicted": "ERROR",
                "latency_ms": (
                    time.perf_counter() - start
                ) * 1000,
                "confidence": None,
                "used_fallback": False,
                "api_error": True,
                "cost_usd": 0.0,
                "error_type": type(error).__name__,
            }

        predicted = result["predicted"]

        correct = predicted == expected

        result.update({
            "query": query,
            "expected": expected,
            "correct": correct,
            "valid": predicted in ROUTES,
        })

        records.append(result)

        status = "PASS" if correct else "FAIL"

        print(
            f"[{index:02d}] {status} | "
            f"{expected:<20} → "
            f"{predicted:<20} | "
            f"{result['latency_ms']:.2f} ms"
        )

        if result["used_fallback"]:

            print(
                f"     FALLBACK | "
                f"Jev: {result['jev_prediction']} | "
                f"Confidence: {result['confidence']} | "
                f"Reason: {result['fallback_reason']}"
            )

        if not correct:

            print(f"     Question: {query}")

    return records


# --------------------------------------------------
# SUMMARIZE ONE ROUTER
# --------------------------------------------------

def summarize_router(records):

    total = len(records)

    correct = sum(
        record["correct"]
        for record in records
    )

    invalid = sum(
        not record["valid"]
        for record in records
    )

    api_errors = sum(
        record["api_error"]
        for record in records
    )

    latencies = [
        record["latency_ms"]
        for record in records
    ]

    fallbacks = [
        record
        for record in records
        if record["used_fallback"]
    ]

    cost = sum(
        record["cost_usd"]
        for record in records
    )

    per_route = {}

    for route in ROUTES:

        matching = [
            record
            for record in records
            if record["expected"] == route
        ]

        route_correct = sum(
            record["correct"]
            for record in matching
        )

        per_route[route] = {
            "correct": route_correct,
            "total": len(matching),
            "accuracy_percent": (
                100 * route_correct / len(matching)
                if matching else 0
            ),
        }

    return {
        "correct": correct,
        "total": total,
        "accuracy_percent": 100 * correct / total,
        "invalid_outputs": invalid,
        "api_errors": api_errors,

        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),

        "fallback_count": len(fallbacks),
        "fallback_rate_percent": (
            100 * len(fallbacks) / total
        ),

        "cost_usd": cost,
        "per_route": per_route,
    }


# --------------------------------------------------
# PRINT FINAL COMPARISON
# --------------------------------------------------

def print_comparison(summaries):

    print()
    print("=" * 95)
    print("FINAL ROUTER COMPARISON — 40 UNSEEN QUESTIONS")
    print("=" * 95)

    print()

    header = (
        f"{'Router':<15}"
        f"{'Accuracy':<13}"
        f"{'Median':<14}"
        f"{'P95':<14}"
        f"{'Mean':<14}"
        f"{'Fallbacks':<12}"
    )

    print(header)
    print("-" * 95)

    for name, summary in summaries.items():

        print(
            f"{name:<15}"
            f"{summary['accuracy_percent']:.1f}%"
            f" ({summary['correct']}/40)"
            f"{'':<3}"
            f"{summary['median_latency_ms']:<14.2f}"
            f"{summary['p95_latency_ms']:<14.2f}"
            f"{summary['mean_latency_ms']:<14.2f}"
            f"{summary['fallback_count']:<12}"
        )

    print()
    print("=" * 95)
    print("PER-ROUTE ACCURACY")
    print("=" * 95)

    print()

    print(
        f"{'Expected route':<24}"
        f"{'Ministral':<15}"
        f"{'Jev':<15}"
        f"{'Hybrid':<15}"
    )

    print("-" * 75)

    for route in ROUTES:

        values = []

        for name in [
            "MINISTRAL",
            "JEV",
            "HYBRID",
        ]:

            stats = summaries[name]["per_route"][route]

            values.append(
                f"{stats['correct']}/{stats['total']}"
            )

        print(
            f"{route:<24}"
            f"{values[0]:<15}"
            f"{values[1]:<15}"
            f"{values[2]:<15}"
        )

    print()
    print("=" * 95)
    print("COST")
    print("=" * 95)

    for name, summary in summaries.items():

        print(
            f"{name:<15}"
            f"${summary['cost_usd']:.8f}"
        )

    print()
    print(
        "Note: Local inference hardware/energy costs "
        "are not included."
    )


# --------------------------------------------------
# FAILURE ANALYSIS
# --------------------------------------------------

def print_failures(all_records):

    print()
    print("=" * 85)
    print("FAILURE ANALYSIS")
    print("=" * 85)

    for name, records in all_records.items():

        failures = [
            record
            for record in records
            if not record["correct"]
        ]

        print()
        print(
            f"{name}: {len(failures)} failures"
        )

        for failure in failures:

            print()
            print(f"  Query: {failure['query']}")
            print(f"  Expected: {failure['expected']}")
            print(f"  Predicted: {failure['predicted']}")

            if failure.get("confidence") is not None:

                print(
                    f"  Confidence: "
                    f"{failure['confidence']:.4f}"
                )

            probabilities = failure.get("probabilities")

            if probabilities:

                print("  Probabilities:")

                for route, probability in sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):

                    print(
                        f"    {route:<20} "
                        f"{probability:.4f}"
                    )


# --------------------------------------------------
# HYBRID FALLBACK ANALYSIS
# --------------------------------------------------

def print_hybrid_fallbacks(records):

    fallbacks = [
        record
        for record in records
        if record["used_fallback"]
    ]

    print()
    print("=" * 85)
    print("HYBRID FALLBACK ANALYSIS")
    print("=" * 85)

    print()
    print(f"Fallbacks: {len(fallbacks)}/40")

    for record in fallbacks:

        print()
        print(f"Query: {record['query']}")

        print(
            f"Expected: {record['expected']}"
        )

        print(
            f"Jev prediction: "
            f"{record['jev_prediction']}"
        )

        print(
            f"Final prediction: "
            f"{record['predicted']}"
        )

        print(
            f"Confidence: "
            f"{record['confidence']}"
        )

        print(
            f"Fallback reason: "
            f"{record['fallback_reason']}"
        )

        print(
            f"Jev latency: "
            f"{record['jev_latency_ms']:.2f} ms"
        )

        print(
            f"Ministral latency: "
            f"{record['local_latency_ms']:.2f} ms"
        )

        print(
            f"Total latency: "
            f"{record['latency_ms']:.2f} ms"
        )

        jev_was_correct = (
            record["jev_prediction"]
            == record["expected"]
        )

        final_is_correct = record["correct"]

        if not jev_was_correct and final_is_correct:

            outcome = "RESCUED BY FALLBACK"

        elif jev_was_correct and not final_is_correct:

            outcome = "DAMAGED BY FALLBACK"

        elif final_is_correct:

            outcome = "CORRECT BEFORE AND AFTER"

        else:

            outcome = "STILL INCORRECT"

        print(f"Outcome: {outcome}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    validate_test_cases()

    if not os.getenv("TYPESAFE_API_KEY"):

        raise RuntimeError(
            "TYPESAFE_API_KEY missing from .env"
        )

    print()
    print("=" * 85)
    print("CHAPTER 18 — ROUTER GENERALIZATION BENCHMARK")
    print("=" * 85)

    print()
    print(f"Local model: {ROUTER_MODEL}")
    print(f"Jev model: {JEV_MODEL}")

    print(
        f"Hybrid confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print()
    print("40 questions × 3 architectures")

    all_records = {}

    # Reuse the same TypeSafe client.
    #
    # Run Jev and Hybrid as separate live tests.
    # This means 80 Jev API calls overall.

    with TypeSafeClient(model=JEV_MODEL) as client:

        all_records["JEV"] = evaluate_router(
            "JEV",
            lambda query: run_jev(client, query),
        )

        all_records["HYBRID"] = evaluate_router(
            "HYBRID",
            lambda query: run_hybrid(client, query),
        )

    all_records["MINISTRAL"] = evaluate_router(
        "MINISTRAL",
        run_ministral,
    )

    summaries = {
        name: summarize_router(records)
        for name, records in all_records.items()
    }

    print_comparison(summaries)

    print_failures(all_records)

    print_hybrid_fallbacks(
        all_records["HYBRID"]
    )

    # Save everything for our eventual
    # Chapter 18 README and observability work.

    output = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "models": {
            "ministral": ROUTER_MODEL,
            "jev": JEV_MODEL,
        },

        "confidence_threshold": CONFIDENCE_THRESHOLD,

        "summaries": summaries,

        "records": all_records,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 85)
    print("BENCHMARK COMPLETE")
    print("=" * 85)

    print()
    print(f"Results saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()