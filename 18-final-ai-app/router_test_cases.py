# --------------------------------------------------
# ROUTE DEFINITIONS
# --------------------------------------------------

ROUTES = [
    "CHAT",
    "RAG",
    "MULTIMODAL",
    "TOOL",
    "BENCHMARK_ANALYSIS",
]


# --------------------------------------------------
# ROUTING TEST CASES
# --------------------------------------------------

TEST_CASES = [

    # ----------------------------------------------
    # CHAT
    # ----------------------------------------------

    {
        "query":
            "What is grouped-query attention?",

        "expected_route":
            "CHAT",
    },

    {
        "query":
            "Explain the difference between "
            "prefill and decode.",

        "expected_route":
            "CHAT",
    },

    {
        "query":
            "Why are GPUs good for matrix "
            "multiplication?",

        "expected_route":
            "CHAT",
    },

    {
        "query":
            "What does arithmetic intensity mean?",

        "expected_route":
            "CHAT",
    },


    # ----------------------------------------------
    # RAG
    # ----------------------------------------------

    {
        "query":
            "What did I learn about KV cache "
            "during inference?",

        "expected_route":
            "RAG",
    },

    {
        "query":
            "What exact benchmark prompt did I "
            "use to test KV cache?",

        "expected_route":
            "RAG",
    },

    {
        "query":
            "How did I implement multimodal "
            "concurrency benchmarking?",

        "expected_route":
            "RAG",
    },

    {
        "query":
            "What was my conclusion about "
            "arithmetic intensity?",

        "expected_route":
            "RAG",
    },


    # ----------------------------------------------
    # MULTIMODAL
    # ----------------------------------------------

    {
        "query":
            "Look at this screenshot and tell me "
            "what error I am getting.",

        "expected_route":
            "MULTIMODAL",
    },

    {
        "query":
            "What does this benchmark graph show?",

        "expected_route":
            "MULTIMODAL",
    },

    {
        "query":
            "Can you explain what is happening "
            "in this GPU profiler screenshot?",

        "expected_route":
            "MULTIMODAL",
    },

    {
        "query":
            "Read this terminal screenshot and "
            "tell me why the command failed.",

        "expected_route":
            "MULTIMODAL",
    },


    # ----------------------------------------------
    # TOOL
    # ----------------------------------------------

    {
        "query":
            "Calculate the percentage increase "
            "from 106 tokens per second to "
            "574 tokens per second.",

        "expected_route":
            "TOOL",
    },

    {
        "query":
            "Convert 8192 MB to GB.",

        "expected_route":
            "TOOL",
    },

    {
        "query":
            "How much memory would a 70 billion "
            "parameter BF16 model require?",

        "expected_route":
            "TOOL",
    },

    {
        "query":
            "Show me my current system memory "
            "information.",

        "expected_route":
            "TOOL",
    },


    # ----------------------------------------------
    # BENCHMARK ANALYSIS
    # ----------------------------------------------

    {
        "query":
            "Why did throughput increase as "
            "concurrency increased in my "
            "multimodal benchmark?",

        "expected_route":
            "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "My TTFT increased by 19x but TPOT "
            "barely changed. What does that tell "
            "me about the bottleneck?",

        "expected_route":
            "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "Why can latency increase while "
            "overall throughput also improves?",

        "expected_route":
            "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "Given these benchmark results, "
            "does the workload look memory-bound "
            "or compute-bound?",

        "expected_route":
            "BENCHMARK_ANALYSIS",
    },
]


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def print_test_summary():

    print()
    print("=" * 70)
    print("ROUTER TEST SET")
    print("=" * 70)

    counts = {}

    for route in ROUTES:
        counts[route] = 0

    for case in TEST_CASES:
        counts[
            case["expected_route"]
        ] += 1

    print()

    print(
        f"Total test cases: "
        f"{len(TEST_CASES)}"
    )

    print()

    for route in ROUTES:

        print(
            f"{route}: "
            f"{counts[route]}"
        )


if __name__ == "__main__":
    print_test_summary()