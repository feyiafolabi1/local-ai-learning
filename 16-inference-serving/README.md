# Chapter 16 — Inference Serving

Explored serving an LLM with vLLM and SGLang, focusing on time to first token (TTFT), decode timing, latency, and throughput under concurrent requests.

## Setup

- RunPod RTX 4090 with 24 GB VRAM
- Model: `Qwen/Qwen2.5-3B-Instruct`
- vLLM: `0.29.0`
- SGLang: `0.5.20`

The original pod files were lost when the pod was stopped. These three scripts were recovered from the experiment code preserved in the conversation. The results below are recorded observations from those runs, not new measurements.

## Files

| File | Experiment |
| --- | --- |
| `benchmark_single.py` | One streaming request: TTFT, inter-chunk timing, total latency, and chunks/sec |
| `benchmark_concurrency.py` | Non-streaming requests at concurrency 1, 4, and 8: latency and output token throughput |
| `benchmark_concurrency_ttft.py` | Streaming requests at concurrency 1, 4, and 8: TTFT, latency, and output token throughput |

The streaming concurrency script uses `stream_options.include_usage=true` to obtain completion token counts. Streaming chunks are not guaranteed to correspond one-to-one with model tokens.

## Run the vLLM experiments

In a compatible GPU environment with vLLM installed, start the server:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.80
```

In a second terminal on the same machine, install the client dependency and run the scripts from this folder:

```bash
python -m pip install requests
python benchmark_single.py
python benchmark_concurrency.py
python benchmark_concurrency_ttft.py
```

All scripts target `http://localhost:8000/v1/chat/completions`; they require a running model server.

## Recorded results

The single streaming vLLM request measured about **31 ms TTFT**, **8 ms between content chunks**, and **1.63 seconds total latency** across 200 content chunks.

The corrected streaming concurrency run showed:

| Concurrent requests | Output throughput | Average TTFT |
| --- | --- | --- |
| 1 | ~122 tokens/sec | ~24 ms |
| 4 | ~435 tokens/sec | ~17 ms |
| 8 | ~831 tokens/sec | ~23 ms |

Average request latency stayed around 1–1.1 seconds. Higher concurrency substantially increased aggregate throughput, demonstrating effective batching. Repeated prompts, prefix caching, warm-up, and timing noise can affect TTFT; these were small exploratory runs.

For SGLang, the built-in benchmark was run against a server on port 30000:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://localhost:30000 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset-name random \
  --num-prompts 8 \
  --random-input-len 64 \
  --random-output-len 120 \
  --request-rate inf
```

It reported approximately **619 output tokens/sec**, **184 ms mean TTFT**, **8.4 ms mean time per output token**, and **940 ms mean end-to-end latency**. Different prompts, output lengths, and benchmark methods mean these results do not establish a winner between the engines.

## Key takeaways

- TTFT measures how long the user waits for the first output; total latency measures the full request.
- Throughput measures aggregate work completed per second.
- Continuous batching lets the server process multiple sequences efficiently.
- PagedAttention manages KV-cache memory in blocks, reducing wasted allocation and fragmentation.
- Both vLLM and SGLang expose OpenAI-compatible APIs; meaningful performance comparisons require matched workloads.
