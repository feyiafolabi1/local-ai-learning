# ==================================================
# CHAPTER 18 — ENGINEERING COPILOT TOOLS
# ==================================================
#
# Deterministic tools for AI systems engineering.
#
# Supported:
#   Model weight memory
#   KV-cache memory
#   Tokens per second
#   Percentage latency reduction
#   Effective training batch size
#   GPU runtime cost
#   Unit conversions
#   Available system RAM
#
# No LLM performs the arithmetic.
# ==================================================

import re
import time


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def number(text):
    return float(text.replace(",", ""))


def extract(pattern, text, description):
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Could not determine {description}."
        )

    return number(match.group(1))


def result(tool, answer, explanation, start):
    return {
        "tool": tool,
        "answer": answer,
        "explanation": explanation,
        "latency_ms": (
            time.perf_counter() - start
        ) * 1000,
    }


# --------------------------------------------------
# MODEL WEIGHT MEMORY
# --------------------------------------------------

def model_memory(query, start):

    match = re.search(
        r"([\d,.]+)\s*(billion|million)\s+parameters?",
        query,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError("Missing parameter count.")

    count = number(match.group(1))

    scale = (
        1_000_000_000
        if match.group(2).lower() == "billion"
        else 1_000_000
    )

    parameters = count * scale

    bytes_per_parameter = extract(
        r"([\d,.]+)\s*bytes?\s+per\s+parameter",
        query,
        "bytes per parameter",
    )

    total_bytes = (
        parameters * bytes_per_parameter
    )

    gib = total_bytes / (1024 ** 3)

    gb = total_bytes / 1_000_000_000

    return result(
        "MODEL_MEMORY",
        f"{gib:.4f} GiB ({gb:.4f} GB)",
        (
            f"{parameters:,.0f} parameters × "
            f"{bytes_per_parameter:g} bytes/parameter "
            f"= {total_bytes:,.0f} bytes. "
            "This excludes KV cache, activations, "
            "quantization metadata, and runtime overhead."
        ),
        start,
    )


# --------------------------------------------------
# KV-CACHE MEMORY
# --------------------------------------------------

def kv_cache_memory(query, start):

    layers = extract(
        r"([\d,]+)\s+layers?",
        query,
        "layer count",
    )

    kv_heads = extract(
        r"([\d,]+)\s+KV\s+heads?",
        query,
        "KV-head count",
    )

    head_dim = extract(
        r"head\s+dimension\s+(?:of\s+)?([\d,]+)",
        query,
        "head dimension",
    )

    sequence = extract(
        r"context\s+length\s+(?:of\s+)?([\d,]+)",
        query,
        "context length",
    )

    batch_match = re.search(
        r"batch\s+size\s+(?:of\s+)?([\d,]+)",
        query,
        re.IGNORECASE,
    )

    batch = (
        number(batch_match.group(1))
        if batch_match
        else 1
    )

    # Use BF16 unless the question specifies FP8.
    # Other formats require explicit handling.

    if re.search(r"\bFP8\b", query, re.IGNORECASE):
        bytes_per_element = 1
        dtype = "FP8"

    else:
        bytes_per_element = 2
        dtype = "BF16 (assumed)"

    total_bytes = (
        2
        * batch
        * layers
        * kv_heads
        * sequence
        * head_dim
        * bytes_per_element
    )

    gib = total_bytes / (1024 ** 3)

    explanation = (
        "KV cache = "
        "2 (K and V) × batch × layers × "
        "KV heads × sequence length × "
        "head dimension × bytes/element.\n"
        f"Values: 2 × {batch:g} × {layers:g} × "
        f"{kv_heads:g} × {sequence:g} × "
        f"{head_dim:g} × {bytes_per_element}.\n"
        f"Precision: {dtype}. "
        "Excludes allocator overhead and padding."
    )

    return result(
        "KV_CACHE_MEMORY",
        f"{gib:.4f} GiB",
        explanation,
        start,
    )


# --------------------------------------------------
# TOKENS PER SECOND
# --------------------------------------------------

def token_throughput(query, start):

    tokens = extract(
        r"([\d,.]+)\s+(?:output\s+)?tokens",
        query,
        "output token count",
    )

    seconds = extract(
        r"\bin\s+([\d,.]+)\s+seconds?",
        query,
        "generation time",
    )

    if seconds <= 0:
        raise ValueError(
            "Generation time must be positive."
        )

    speed = tokens / seconds

    return result(
        "TOKEN_THROUGHPUT",
        f"{speed:.2f} tokens/second",
        f"{tokens:g} tokens ÷ {seconds:g} seconds.",
        start,
    )


# --------------------------------------------------
# LATENCY REDUCTION
# --------------------------------------------------

def latency_reduction(query, start):

    match = re.search(
        r"from\s+([\d,.]+)\s*(?:ms|milliseconds?)?"
        r"\s+to\s+([\d,.]+)",
        query,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            "Expected a starting and ending latency."
        )

    before = number(match.group(1))
    after = number(match.group(2))

    if before <= 0:
        raise ValueError(
            "Starting latency must be positive."
        )

    reduction = (
        (before - after) / before
    ) * 100

    return result(
        "LATENCY_REDUCTION",
        f"{reduction:.2f}%",
        (
            f"({before:g} - {after:g}) "
            f"÷ {before:g} × 100."
        ),
        start,
    )


# --------------------------------------------------
# EFFECTIVE TRAINING BATCH SIZE
# --------------------------------------------------

def effective_batch_size(query, start):

    microbatch = extract(
        r"microbatch\s+size\s+([\d,]+)",
        query,
        "microbatch size",
    )

    accumulation = extract(
        r"gradient\s+accumulation\s+([\d,]+)",
        query,
        "gradient accumulation steps",
    )

    gpus = extract(
        r"([\d,]+)\s+GPUs?",
        query,
        "GPU count",
    )

    effective = (
        microbatch * accumulation * gpus
    )

    return result(
        "EFFECTIVE_BATCH_SIZE",
        f"{effective:g} samples",
        (
            f"{microbatch:g} samples/GPU/microstep × "
            f"{accumulation:g} accumulation steps × "
            f"{gpus:g} GPUs."
        ),
        start,
    )


# --------------------------------------------------
# GPU RUNTIME COST
# --------------------------------------------------

def gpu_runtime_cost(query, start):

    hourly = extract(
        r"\$\s*([\d,.]+)\s+per\s+hour",
        query,
        "GPU hourly cost",
    )

    seconds = extract(
        r"([\d,.]+)\s+seconds?",
        query,
        "runtime",
    )

    cost = hourly * seconds / 3600

    return result(
        "GPU_RUNTIME_COST",
        f"${cost:.4f} USD",
        (
            f"${hourly:g}/hour × "
            f"{seconds:g}/3600 hours. "
            "Assumes per-second billing with no minimum."
        ),
        start,
    )


# --------------------------------------------------
# UNIT CONVERSIONS
# --------------------------------------------------

def convert_units(query, start):

    lower = query.lower()

    value = extract(
        r"convert\s+([\d,.]+)",
        query,
        "value to convert",
    )

    if (
        "milliseconds" in lower
        and "seconds" in lower
    ):

        converted = value / 1000

        return result(
            "UNIT_CONVERSION",
            f"{converted:g} seconds",
            f"{value:g} ms ÷ 1000.",
            start,
        )

    if "gib" in lower and "mib" in lower:

        converted = value * 1024

        return result(
            "UNIT_CONVERSION",
            f"{converted:,.0f} MiB",
            f"{value:g} GiB × 1024.",
            start,
        )

    raise ValueError(
        "This unit conversion is not supported yet."
    )


# --------------------------------------------------
# AVAILABLE SYSTEM MEMORY
# --------------------------------------------------

def available_ram(start):

    try:
        import psutil

    except ImportError as error:

        raise RuntimeError(
            "System RAM tool requires psutil. "
            "Install with: python3 -m pip install psutil"
        ) from error

    memory = psutil.virtual_memory()

    available_gib = (
        memory.available / (1024 ** 3)
    )

    total_gib = (
        memory.total / (1024 ** 3)
    )

    return result(
        "SYSTEM_MEMORY",
        f"{available_gib:.2f} GiB available",
        (
            f"Total RAM: {total_gib:.2f} GiB. "
            f"Reported usage: {memory.percent:.1f}%. "
            "Available RAM is an OS estimate and "
            "can change during execution."
        ),
        start,
    )


# --------------------------------------------------
# TOOL DISPATCHER
# --------------------------------------------------

def execute_tool(query):

    start = time.perf_counter()

    lower = " ".join(query.lower().split())

    if "kv cache" in lower or "kv-cache" in lower:
        return kv_cache_memory(query, start)

    if (
        "parameters" in lower
        and "bytes per parameter" in lower
    ):
        return model_memory(query, start)

    if (
        "effective batch size" in lower
        or "gradient accumulation" in lower
    ):
        return effective_batch_size(query, start)

    if (
        "gpu costs" in lower
        or "gpu cost" in lower
    ):
        return gpu_runtime_cost(query, start)

    if (
        "tokens per second" in lower
        or "tokens/second" in lower
    ):
        return token_throughput(query, start)

    if (
        "percentage reduction" in lower
        or "percent reduction" in lower
    ):
        return latency_reduction(query, start)

    if "available ram" in lower:
        return available_ram(start)

    if lower.startswith("convert"):
        return convert_units(query, start)

    raise ValueError(
        "No supported deterministic tool matched "
        "this question. Please provide the "
        "required quantities and units."
    )


# --------------------------------------------------
# LOCAL TESTS
# --------------------------------------------------

def main():

    tests = [
        (
            "Calculate the memory in GiB needed "
            "to store 8 billion parameters "
            "at 2 bytes per parameter."
        ),

        (
            "Calculate KV cache memory for "
            "32 layers, 8 KV heads, "
            "head dimension of 128, "
            "context length 8192."
        ),

        (
            "Calculate tokens per second if "
            "a model generates 480 output tokens "
            "in 3.2 seconds."
        ),

        (
            "Calculate the percentage reduction "
            "when latency falls from 800 ms to 520 ms."
        ),

        (
            "Calculate effective batch size with "
            "microbatch size 3, gradient accumulation "
            "8, and 4 GPUs."
        ),

        (
            "If a GPU costs $2.40 per hour, "
            "calculate the cost of running it "
            "for 90 seconds."
        ),

        (
            "Convert 750 milliseconds into seconds."
        ),

        (
            "Convert 128 GiB into MiB."
        ),
    ]

    print()
    print("=" * 70)
    print("CHAPTER 18 — TOOL EXECUTION TEST")
    print("=" * 70)

    passed = 0

    for index, query in enumerate(tests, start=1):

        print()
        print(f"TEST {index}")
        print("-" * 70)

        try:

            output = execute_tool(query)

            print(f"Tool: {output['tool']}")
            print(f"Answer: {output['answer']}")
            print(output["explanation"])

            print(
                f"Execution: "
                f"{output['latency_ms']:.3f} ms"
            )

            passed += 1

        except Exception as error:

            print(
                f"FAILED: "
                f"{type(error).__name__}: {error}"
            )

    print()
    print("=" * 70)
    print(f"Tests completed: {passed}/{len(tests)}")
    print("=" * 70)


if __name__ == "__main__":
    main()