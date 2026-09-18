import torch
import time

print("PyTorch version:", torch.__version__)
print()

if torch.backends.mps.is_available():
    gpu_device = torch.device("mps")
else:
    gpu_device = None

cpu_device = torch.device("cpu")

print("CPU device:", cpu_device)
print("GPU device:", gpu_device)
print()

# Matrix size
size = 2048

# Create matrices on CPU
a_cpu = torch.randn(size, size, device=cpu_device)
b_cpu = torch.randn(size, size, device=cpu_device)

print(f"Matrix size: {size} x {size}")
print()

# ---------------------------
# CPU benchmark
# ---------------------------

# Warm-up
_ = a_cpu @ b_cpu

start = time.perf_counter()

c_cpu = a_cpu @ b_cpu

end = time.perf_counter()

cpu_time = end - start

print("CPU matrix multiply time:")
print(f"{cpu_time:.6f} seconds")
print()

# ---------------------------
# GPU benchmark
# ---------------------------

if gpu_device is not None:

    a_gpu = a_cpu.to(gpu_device)
    b_gpu = b_cpu.to(gpu_device)

    # Warm-up
    _ = a_gpu @ b_gpu

    # Make sure previous GPU work is finished
    torch.mps.synchronize()

    start = time.perf_counter()

    c_gpu = a_gpu @ b_gpu

    # GPU operations are asynchronous,
    # so wait until the multiplication is actually finished
    torch.mps.synchronize()

    end = time.perf_counter()

    gpu_time = end - start

    print("GPU matrix multiply time:")
    print(f"{gpu_time:.6f} seconds")
    print()

    print("Speedup:")
    print(f"{cpu_time / gpu_time:.2f}x")
