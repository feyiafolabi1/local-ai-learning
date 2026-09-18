import time
import torch

CPU = torch.device("cpu")
GPU = torch.device("mps") if torch.backends.mps.is_available() else None

sizes = [128, 256, 512, 1024, 2048, 4096]

print("PyTorch version:", torch.__version__)
print("CPU:", CPU)
print("GPU:", GPU)
print()

print(
    f"{'Size':>8} "
    f"{'CPU (ms)':>12} "
    f"{'GPU (ms)':>12} "
    f"{'Speedup':>10}"
)

print("-" * 46)

for size in sizes:
    a_cpu = torch.randn(size, size, device=CPU)
    b_cpu = torch.randn(size, size, device=CPU)

    # CPU warm-up
    _ = a_cpu @ b_cpu

    start = time.perf_counter()
    _ = a_cpu @ b_cpu
    end = time.perf_counter()

    cpu_time = end - start

    if GPU is not None:
        a_gpu = a_cpu.to(GPU)
        b_gpu = b_cpu.to(GPU)

        # GPU warm-up
        _ = a_gpu @ b_gpu
        torch.mps.synchronize()

        start = time.perf_counter()

        _ = a_gpu @ b_gpu

        torch.mps.synchronize()
        end = time.perf_counter()

        gpu_time = end - start
        speedup = cpu_time / gpu_time

        print(
            f"{size:>8} "
            f"{cpu_time * 1000:>12.3f} "
            f"{gpu_time * 1000:>12.3f} "
            f"{speedup:>9.2f}x"
        )

    else:
        print(
            f"{size:>8} "
            f"{cpu_time * 1000:>12.3f} "
            f"{'N/A':>12} "
            f"{'N/A':>10}"
        )