import time
import torch

CPU = torch.device("cpu")
GPU = torch.device("mps") if torch.backends.mps.is_available() else None

if GPU is None:
    print("MPS GPU not available.")
    raise SystemExit

print("PyTorch version:", torch.__version__)
print("CPU:", CPU)
print("GPU:", GPU)
print()

# --------------------------------------------------
# Experiment 1: Low arithmetic intensity
# Elementwise multiply
# --------------------------------------------------

n = 20_000_000

x_cpu = torch.randn(n, device=CPU)

# CPU warm-up
_ = x_cpu * 2

start = time.perf_counter()
_ = x_cpu * 2
end = time.perf_counter()

cpu_elementwise = end - start

# Create directly on GPU to avoid transfer timing
x_gpu = torch.randn(n, device=GPU)

# GPU warm-up
_ = x_gpu * 2
torch.mps.synchronize()

start = time.perf_counter()
_ = x_gpu * 2
torch.mps.synchronize()
end = time.perf_counter()

gpu_elementwise = end - start

# --------------------------------------------------
# Experiment 2: High arithmetic intensity
# Matrix multiplication
# --------------------------------------------------

size = 4096

a_cpu = torch.randn(size, size, device=CPU)
b_cpu = torch.randn(size, size, device=CPU)

# CPU warm-up
_ = a_cpu @ b_cpu

start = time.perf_counter()
_ = a_cpu @ b_cpu
end = time.perf_counter()

cpu_matmul = end - start

a_gpu = torch.randn(size, size, device=GPU)
b_gpu = torch.randn(size, size, device=GPU)

# GPU warm-up
_ = a_gpu @ b_gpu
torch.mps.synchronize()

start = time.perf_counter()
_ = a_gpu @ b_gpu
torch.mps.synchronize()
end = time.perf_counter()

gpu_matmul = end - start

# --------------------------------------------------
# Results
# --------------------------------------------------

print("LOW ARITHMETIC INTENSITY")
print("------------------------")
print(f"Elementwise tensor size: {n:,}")
print(f"CPU time: {cpu_elementwise * 1000:.3f} ms")
print(f"GPU time: {gpu_elementwise * 1000:.3f} ms")
print(f"GPU speedup: {cpu_elementwise / gpu_elementwise:.2f}x")
print()

print("HIGH ARITHMETIC INTENSITY")
print("-------------------------")
print(f"Matrix size: {size} x {size}")
print(f"CPU time: {cpu_matmul * 1000:.3f} ms")
print(f"GPU time: {gpu_matmul * 1000:.3f} ms")
print(f"GPU speedup: {cpu_matmul / gpu_matmul:.2f}x")