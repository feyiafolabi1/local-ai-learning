import time
import torch

CPU = torch.device("cpu")
GPU = torch.device("mps") if torch.backends.mps.is_available() else None

size = 4096

print("PyTorch version:", torch.__version__)
print("CPU:", CPU)
print("GPU:", GPU)
print()
print(f"Matrix size: {size} x {size}")
print()

# Create matrices on CPU
a_cpu = torch.randn(size, size, device=CPU)
b_cpu = torch.randn(size, size, device=CPU)

if GPU is None:
    print("MPS GPU not available.")
    raise SystemExit

# -----------------------------
# 1. CPU -> GPU transfer
# -----------------------------

start = time.perf_counter()

a_gpu = a_cpu.to(GPU)
b_gpu = b_cpu.to(GPU)

torch.mps.synchronize()

end = time.perf_counter()

cpu_to_gpu_time = end - start

# -----------------------------
# 2. GPU compute
# -----------------------------

# Warm-up
_ = a_gpu @ b_gpu
torch.mps.synchronize()

start = time.perf_counter()

c_gpu = a_gpu @ b_gpu

torch.mps.synchronize()

end = time.perf_counter()

gpu_compute_time = end - start

# -----------------------------
# 3. GPU -> CPU transfer
# -----------------------------

start = time.perf_counter()

c_cpu = c_gpu.to(CPU)

torch.mps.synchronize()

end = time.perf_counter()

gpu_to_cpu_time = end - start

# -----------------------------
# 4. Total
# -----------------------------

total_time = (
    cpu_to_gpu_time
    + gpu_compute_time
    + gpu_to_cpu_time
)

print("CPU -> GPU transfer:")
print(f"{cpu_to_gpu_time * 1000:.3f} ms")
print()

print("GPU compute:")
print(f"{gpu_compute_time * 1000:.3f} ms")
print()

print("GPU -> CPU transfer:")
print(f"{gpu_to_cpu_time * 1000:.3f} ms")
print()

print("Total GPU path:")
print(f"{total_time * 1000:.3f} ms")