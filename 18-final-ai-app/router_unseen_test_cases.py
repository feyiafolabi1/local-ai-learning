# ==================================================
# CHAPTER 18 — UNSEEN ROUTER EVALUATION
# ==================================================
#
# 40 new routing questions.
#
# 8 questions per route.
#
# IMPORTANT:
# Do not modify these questions or expected labels
# after evaluating the routers.
#
# This is our held-out evaluation set.
# ==================================================


ROUTES = [
    "CHAT",
    "RAG",
    "MULTIMODAL",
    "TOOL",
    "BENCHMARK_ANALYSIS",
]


TEST_CASES = [

    # ==============================================
    # CHAT — GENERAL KNOWLEDGE
    # ==============================================

    {
        "query":
            "What is the difference between "
            "self-attention and cross-attention?",
        "expected_route": "CHAT",
    },

    {
        "query":
            "Why is fused multiply-add useful "
            "for matrix operations?",
        "expected_route": "CHAT",
    },

    {
        "query":
            "Explain the purpose of activation "
            "functions in neural networks.",
        "expected_route": "CHAT",
    },

    {
        "query":
            "How does the roofline model help "
            "engineers understand GPU performance?",
        "expected_route": "CHAT",
    },

    {
        "query":
            "What is the difference between "
            "data parallelism and tensor parallelism?",
        "expected_route": "CHAT",
    },

    {
        "query":
            "What does a GPU wavefront do?",
        "expected_route": "CHAT",
    },

    {
        "query":
            "Why does batch normalization behave "
            "differently during training and inference?",
        "expected_route": "CHAT",
    },

    {
        "query":
            "How does speculative decoding "
            "preserve the target model's "
            "output distribution?",
        "expected_route": "CHAT",
    },


    # ==============================================
    # RAG — REPOSITORY KNOWLEDGE
    # ==============================================

    {
        "query":
            "Which two inference-serving "
            "frameworks did I experiment "
            "with in Chapter 16?",
        "expected_route": "RAG",
    },

    {
        "query":
            "What was my measured GPU speedup "
            "over CPU for the largest matrix "
            "multiplication in Chapter 15?",
        "expected_route": "RAG",
    },

    {
        "query":
            "What problem forced me to resize "
            "images during my multimodal "
            "concurrency experiments?",
        "expected_route": "RAG",
    },

    {
        "query":
            "Which model did I export and "
            "quantize to Q4_K_M in Chapter 14?",
        "expected_route": "RAG",
    },

    {
        "query":
            "What changed between the first "
            "and second versions of my "
            "fine-tuning dataset?",
        "expected_route": "RAG",
    },

    {
        "query":
            "In my RAG evaluation chapter, "
            "which question improved from "
            "vector rank 6 to reranked rank 1?",
        "expected_route": "RAG",
    },

    {
        "query":
            "Why did I use different images "
            "instead of repeatedly sending "
            "the same image in Chapter 17?",
        "expected_route": "RAG",
    },

    {
        "query":
            "Which Python files did I use "
            "to investigate matrix multiplication "
            "and data transfer overhead?",
        "expected_route": "RAG",
    },


    # ==============================================
    # MULTIMODAL — VISUAL UNDERSTANDING
    # ==============================================

    {
        "query":
            "The profiler heatmap I uploaded "
            "has a bright horizontal band. "
            "What does that indicate?",
        "expected_route": "MULTIMODAL",
    },

    {
        "query":
            "Read the exception message "
            "in this photo of my terminal.",
        "expected_route": "MULTIMODAL",
    },

    {
        "query":
            "Which bar in this screenshot "
            "represents prefill time?",
        "expected_route": "MULTIMODAL",
    },

    {
        "query":
            "Compare the two latency curves "
            "shown in the figure I uploaded.",
        "expected_route": "MULTIMODAL",
    },

    {
        "query":
            "What setting is selected in "
            "this dashboard screenshot?",
        "expected_route": "MULTIMODAL",
    },

    {
        "query":
            "Can you spot the missing connection "
            "in this GPU architecture diagram?",
        "expected_route": "MULTIMODAL",
    },

    {
        "query":
            "Transcribe the warning text "
            "from the image I'm showing you.",
        "expected_route": "MULTIMODAL",
    },

    {
        "query":
            "Does the graph in this attached "
            "photo contain any obvious outliers?",
        "expected_route": "MULTIMODAL",
    },


    # ==============================================
    # TOOL — DETERMINISTIC OPERATIONS
    # ==============================================

    {
        "query":
            "Calculate the memory in GiB needed "
            "to store 8 billion parameters "
            "at 2 bytes per parameter.",
        "expected_route": "TOOL",
    },

    {
        "query":
            "Convert 750 milliseconds "
            "into seconds.",
        "expected_route": "TOOL",
    },

    {
        "query":
            "Calculate the percentage reduction "
            "when latency falls from 800 ms "
            "to 520 ms.",
        "expected_route": "TOOL",
    },

    {
        "query":
            "Calculate tokens per second if "
            "a model generates 480 output tokens "
            "in 3.2 seconds.",
        "expected_route": "TOOL",
    },

    {
        "query":
            "Convert 128 GiB of GPU memory "
            "into MiB.",
        "expected_route": "TOOL",
    },

    {
        "query":
            "Check how much available RAM "
            "my Mac currently has.",
        "expected_route": "TOOL",
    },

    {
        "query":
            "Calculate the effective batch size "
            "with microbatch size 3, gradient "
            "accumulation 8, and 4 GPUs.",
        "expected_route": "TOOL",
    },

    {
        "query":
            "If a GPU costs $2.40 per hour, "
            "calculate the cost of running "
            "it for 90 seconds.",
        "expected_route": "TOOL",
    },


    # ==============================================
    # BENCHMARK_ANALYSIS — PERFORMANCE REASONING
    # ==============================================

    {
        "query":
            "My decode token rate stayed flat "
            "as batch size doubled, but TTFT "
            "rose sharply. What bottleneck "
            "might that indicate?",
        "expected_route": "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "My kernel's measured GFLOPS doubled "
            "after increasing tile size, while "
            "HBM traffic decreased. "
            "How should I interpret this?",
        "expected_route": "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "Concurrency increased from 2 to 16. "
            "P95 latency tripled, but aggregate "
            "tokens per second doubled. "
            "What tradeoff am I seeing?",
        "expected_route": "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "Why might my quantized model "
            "use less VRAM but generate tokens "
            "no faster than BF16 in my tests?",
        "expected_route": "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "The same prompt had much lower "
            "TTFT on its second run. "
            "How would I distinguish prompt "
            "caching from GPU warmup?",
        "expected_route": "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "My attention kernel slows down "
            "as sequence length grows, even "
            "though GPU compute utilization "
            "remains low. What should I investigate?",
        "expected_route": "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "A distributed training step spends "
            "40 percent of its execution time "
            "in all-reduce. What does that "
            "imply for scaling efficiency?",
        "expected_route": "BENCHMARK_ANALYSIS",
    },

    {
        "query":
            "HBM bandwidth was near saturation "
            "while arithmetic throughput stayed "
            "low during decode profiling. "
            "What does this tell me about "
            "the performance bottleneck?",
        "expected_route": "BENCHMARK_ANALYSIS",
    },
]


# ==================================================
# VALIDATION
# ==================================================

def validate_test_cases():

    assert len(TEST_CASES) == 40, (
        "Expected exactly 40 test cases."
    )

    counts = {
        route: 0
        for route in ROUTES
    }

    seen_queries = set()

    for case in TEST_CASES:

        route = case["expected_route"]

        assert route in ROUTES, (
            f"Invalid route: {route}"
        )

        query = case["query"]

        assert query not in seen_queries, (
            f"Duplicate query: {query}"
        )

        seen_queries.add(query)

        counts[route] += 1

    for route, count in counts.items():

        assert count == 8, (
            f"{route} has {count} questions, "
            "expected 8."
        )

    print()
    print("=" * 70)
    print("UNSEEN ROUTER EVALUATION SET")
    print("=" * 70)

    print()

    print(f"Total questions: {len(TEST_CASES)}")

    for route, count in counts.items():

        print(
            f"{route:<22} {count}"
        )

    print()
    print("Validation: PASS")
    print("Test set is ready.")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    validate_test_cases()