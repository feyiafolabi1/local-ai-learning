# Chapter 15 — GPU Software Stack

This chapter focused on what actually happens between high-level AI code and the GPU.

## 1. The GPU Software Stack

A framework like PyTorch does not directly execute math on the GPU. It dispatches tensor operations to the appropriate hardware backend.

```text
Python / PyTorch
      ↓
CUDA / ROCm / MPS
      ↓
optimized libraries + kernels
      ↓
GPU hardware
```

Typical vendor stacks:

```text
NVIDIA
PyTorch
→ CUDA
→ cuBLAS / cuDNN / NCCL
→ CUDA kernels
→ NVIDIA GPU

AMD
PyTorch
→ ROCm
→ HIP / rocBLAS / MIOpen / RCCL
→ HIP kernels
→ AMD GPU

Apple
PyTorch
→ MPS
→ Metal
→ Metal kernels
→ Apple GPU
```

CUDA and ROCm are broader software ecosystems, while CUDA and HIP provide programming interfaces for custom GPU kernels.

---

## 2. Kernels and GPU Execution

A GPU kernel is a function that runs on the GPU.

For a simple operation:

```text
y[i] = x[i] * 2
```

many work-items can process different elements in parallel.

```text
AMD:
work-item
   ↓
wavefront
   ↓
workgroup
   ↓
Compute Unit

NVIDIA:
thread
   ↓
warp
   ↓
block
   ↓
Streaming Multiprocessor
```

Custom kernels can sometimes outperform general-purpose kernels when optimized for a specific workload.

Kernel fusion can also combine operations such as:

```text
matmul
+ bias
+ activation
```

into one kernel, reducing intermediate memory reads and writes.

---

## 3. Memory and Performance

GPU memory is hierarchical:

```text
fast / small
Registers
   ↓
LDS / Shared Memory
   ↓
Cache
   ↓
HBM / VRAM
slow / large
```

Good kernels try to load data from slower memory once and reuse it in faster memory.

Two important workload types are:

```text
compute-bound
→ limited by how quickly the GPU can perform math

memory-bound
→ limited by how quickly data can be supplied to the GPU
```

Arithmetic intensity describes how much computation is performed per byte of data moved.

```text
high arithmetic intensity
→ lots of math per byte

low arithmetic intensity
→ little math per byte
```

The roofline idea combines arithmetic intensity, memory bandwidth, and peak compute to determine which resource limits performance.

---

## 4. Prefill, Decode, and KV Cache

LLM inference has two main phases:

```text
prompt
  ↓
PREFILL
  ↓
first output token
  ↓
DECODE
  ↓
token
  ↓
token
  ↓
token...
```

**Prefill** processes many prompt tokens together and usually performs large parallel matrix operations.

**Decode** generates tokens sequentially and repeatedly accesses model weights and the KV cache, making it more sensitive to memory bandwidth.

The KV cache stores previous Key and Value vectors so they do not have to be recomputed every decode step.

```text
new token
   ↓
compute new Q, K, V
   ↓
Q attends to cached previous K/V
   ↓
generate next token
   ↓
append new K/V to cache
```

---

## 5. Multi-GPU Communication

Once a workload spans multiple GPUs, communication becomes another performance concern.

```text
NVIDIA → NCCL
AMD    → RCCL
```

Common collective operations include all-reduce, all-gather, and reduce-scatter.

Models can also be distributed in several ways:

```text
Data parallelism
→ same model, different data

Tensor parallelism
→ split one layer across GPUs

Pipeline parallelism
→ different layers on different GPUs

Expert parallelism
→ different MoE experts on different GPUs
```

---

## 6. Hands-On Experiments

### CPU vs GPU Matrix Multiplication

The GPU only began clearly outperforming the CPU for larger matrices.

```text
2048 × 2048
CPU: 16.259 ms
GPU: 15.345 ms

4096 × 4096
CPU: 130.981 ms
GPU: 47.263 ms

GPU speedup at 4096:
~2.77×
```

This showed that GPUs need enough parallel work to overcome execution overhead.

### Transfer vs Compute

For a 4096 × 4096 workload:

```text
CPU → GPU: 213.881 ms
GPU compute: 47.113 ms
GPU → CPU: 4.596 ms
```

The first CPU → MPS transition included setup and allocation overhead, so it was not a pure memory-copy measurement.

The main lesson was that efficient inference tries to keep model weights, activations, and the KV cache on the GPU instead of repeatedly moving data between devices.

### Arithmetic Intensity

Our elementwise operation showed a larger GPU-vs-CPU speedup than matrix multiplication.

This reinforced an important distinction:

```text
arithmetic intensity tells us
WHAT resource limits performance

not

HOW MUCH faster a GPU must be than a CPU
```

Low arithmetic intensity tends to stress memory bandwidth, while high arithmetic intensity increasingly stresses compute throughput.

---

## 7. Main Takeaways

```text
PyTorch
→ provides a portable high-level tensor API

CUDA / ROCm / Metal
→ provide GPU software environments

kernels
→ perform the actual GPU work

cuBLAS / rocBLAS
→ provide optimized matrix operations

NCCL / RCCL
→ handle multi-GPU communication
```

The biggest lesson from this chapter is that fast AI inference is not just about having a powerful GPU.

Performance depends on how effectively software uses compute, memory bandwidth, data reuse, kernel design, and communication.