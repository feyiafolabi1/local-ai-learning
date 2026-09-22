# Chapter 17 — Multimodal Inference

This chapter explored how vision-language models process images and how multimodal inputs affect inference performance.

## Core Architecture

A simplified vision-language pipeline looks like:

```text
image
  ↓
image preprocessing
  ↓
vision encoder
  ↓
visual features
  ↓
projector / merger
  ↓
LLM-compatible visual tokens
        +
text tokens
  ↓
transformer prefill
  ↓
KV cache
  ↓
autoregressive decode
  ↓
text output
```

The vision encoder extracts useful image features.

The projector maps those visual features into a representation compatible with the language model.

The LLM then reasons over both visual and text information.

---

## Image Patches and Visual Tokens

Images are commonly divided into patches or regions before being processed by a vision transformer.

```text
image
  ↓
patches
  ↓
visual embeddings
  ↓
self-attention
  ↓
contextualized visual features
```

The visual representations participate in attention like other transformer representations.

Higher-resolution images can produce more visual information and therefore increase multimodal inference cost.

---

## Experiment 1 — Vision Grounding

Model:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Serving engine:

```text
vLLM
```

GPU:

```text
RTX 4090
```

The model successfully analyzed multiple images and correctly identified objects, colors, environments, and other visual details.

A separate sanity-check script confirmed that the model was actually grounding its responses in the images rather than simply completing the text prompt.

---

## Experiment 2 — Text vs Image + Text

The same multimodal model was used for both requests so the comparison isolated the cost of adding vision input.

### Text Only

```text
TTFT:               41.88 ms
TPOT/chunk:           8.01 ms
Total latency:        1.052 s
Completion tokens:  127
```

### Image + Text

```text
TTFT:              797.42 ms
TPOT/chunk:           8.41 ms
Total latency:        1.690 s
Completion tokens:  107
```

### Comparison

```text
TTFT increase:      19.04x
TPOT increase:       1.05x
Latency increase:    1.61x
```

The image request dramatically increased time to first token, while decode speed changed very little.

Multimodal inference adds significant front-end work:

```text
image decode / preprocessing
        +
vision encoder
        +
visual projection
        +
larger transformer prefill
```

Once decoding begins, the workload behaves much more like normal LLM generation.

---

## Prefill vs Decode

For the text-only request:

```text
TTFT ≈ 42 ms
total latency ≈ 1052 ms
```

Only about 4% of total latency occurred before the first token.

For the image + text request:

```text
TTFT ≈ 797 ms
total latency ≈ 1690 ms
```

About 47% of total latency occurred before the first token.

This demonstrates that the dominant bottleneck depends heavily on the workload.

```text
short text prompt + long response
→ decode-heavy

large prompt
→ prefill-heavy

multimodal request
→ vision + prefill can become a major part of latency
```

TTFT is not purely transformer prefill. It also includes work such as image preprocessing, vision encoding, projection, scheduling, and the initial LLM prefill.

---

## Image Resolution and Visual Token Cost

One high-resolution image caused the request to exceed the configured context length:

```text
Input length: 16410
Maximum context: 16384
```

The images were then resized so their longest side was approximately 1024 pixels.

This greatly reduced multimodal TTFT.

```text
higher image resolution
        ↓
more visual information / tokens
        ↓
more vision compute
        ↓
larger prefill workload
        ↓
higher TTFT and memory usage
```

Image resolution is therefore an important multimodal inference-performance variable.

It also showed why workload characteristics matter: changing the image itself can change the amount of computation and context required.

---

## Experiment 3 — Multimodal Concurrency

Eight different images were used to reduce the chance that repeatedly using the same image would make the benchmark artificially favorable through caching.

The images were also resized to similar resolutions to make the comparison more controlled.

| Concurrency | Avg TTFT | Avg TPOT | Avg Latency | Throughput |
|---|---:|---:|---:|---:|
| 1 | 162.08 ms | 8.00 ms | 1.012 s | 105.56 tok/s |
| 4 | 273.51 ms | 9.02 ms | 1.246 s | 308.30 tok/s |
| 8 | 229.29 ms | 10.36 ms | 1.374 s | 574.27 tok/s |

Throughput increased strongly with concurrency:

```text
1 request
→ ~106 tok/s

4 requests
→ ~308 tok/s

8 requests
→ ~574 tok/s
```

From concurrency 1 to concurrency 8, throughput increased by roughly 5.4x.

At the same time, average request latency increased only moderately:

```text
1.012 s
→ 1.246 s
→ 1.374 s
```

TPOT also increased only moderately:

```text
8.00 ms
→ 9.02 ms
→ 10.36 ms
```

This shows that vLLM was able to batch the multimodal workload effectively and improve total GPU utilization.

The TTFT values were not perfectly monotonic:

```text
162 ms
→ 274 ms
→ 229 ms
```

This should not be interpreted as concurrency 8 being inherently faster than concurrency 4.

Possible causes include:

```text
batch formation
scheduler timing
GPU warm state
image aspect ratios
different visual-token counts
normal benchmark variation
```

The more useful conclusion is that TTFT remained within the same general range rather than exploding as concurrency increased.

---

## Multimodal vs Text-Only Serving

In Chapter 16, text-only serving at concurrency 8 reached approximately:

```text
831 output tokens/sec
```

The multimodal workload at concurrency 8 reached approximately:

```text
574 output tokens/sec
```

The lower multimodal throughput makes sense because multimodal requests add additional work:

```text
image preprocessing
        +
vision encoder
        +
projection
        +
visual-token prefill
        +
normal LLM inference
```

The decode stage can still batch efficiently, but the server must perform additional work before generation begins.

---

## Workload-Dependent Optimization

One of the main lessons from this chapter is that there is no single universal inference bottleneck.

```text
short prompt + long output
→ decode may dominate

long prompt + short output
→ prefill may dominate

image + text
→ vision processing + prefill may become large

high concurrency
→ scheduling, batching, and KV-cache management matter

multi-GPU inference
→ communication may become important
```

The correct optimization strategy therefore begins with measuring the actual workload.

```text
workload
  ↓
measure
  ↓
identify bottleneck
  ↓
optimize
  ↓
measure again
```

---

## Benchmarking Lessons

Caching and input characteristics can easily distort inference benchmarks.

A fair benchmark should control variables such as:

```text
model
GPU
server configuration
output length
image resolution
prompt structure
cache reuse
```

Using multiple different images helped avoid repeatedly benchmarking the same visual input.

Resizing the images reduced large differences in visual-token workload.

The sanity-check experiment also confirmed that performance numbers were not being collected from meaningless outputs: the vision model correctly described all of the tested images with strong detail.

---

## Main Takeaways

```text
Multimodal inference adds a vision pipeline before LLM generation.

Images can dramatically increase TTFT while affecting TPOT much less.

Vision processing and larger prefill workloads explain much of this TTFT increase.

Image resolution can strongly affect multimodal context length and inference cost.

Multimodal requests can shift latency from being mostly decode-heavy toward a more balanced prefill/decode split.

vLLM can still achieve strong throughput scaling with concurrent multimodal requests.

Performance optimization depends on the workload.

Measure first, identify the bottleneck, then optimize.
```

## Files

```text
benchmark_text_vs_image.py
benchmark_multimodal_concurrency.py
multimodal_sanity_check.py
```