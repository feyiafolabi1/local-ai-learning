import concurrent.futures
import time
import requests

URL = "http://localhost:8000/v1/chat/completions"

MODEL = "Qwen/Qwen2.5-3B-Instruct"

PROMPT = (
    "Explain why GPU memory bandwidth matters during LLM decode "
    "in about 120 words."
)

def send_request(request_id):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0.2,
        "max_tokens": 150,
        "stream": False,
    }

    start = time.perf_counter()

    response = requests.post(URL, json=payload)
    response.raise_for_status()

    end = time.perf_counter()

    data = response.json()

    completion_tokens = data["usage"]["completion_tokens"]

    return {
        "request_id": request_id,
        "latency": end - start,
        "completion_tokens": completion_tokens,
    }


def run_test(concurrency):
    print()
    print("=" * 60)
    print(f"CONCURRENCY = {concurrency}")
    print("=" * 60)

    test_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(send_request, i)
            for i in range(concurrency)
        ]

        results = [
            future.result()
            for future in futures
        ]

    test_end = time.perf_counter()

    total_wall_time = test_end - test_start

    latencies = [
        result["latency"]
        for result in results
    ]

    total_tokens = sum(
        result["completion_tokens"]
        for result in results
    )

    average_latency = sum(latencies) / len(latencies)

    throughput = total_tokens / total_wall_time

    print(f"Average request latency: {average_latency:.3f} s")
    print(f"Total wall-clock time:   {total_wall_time:.3f} s")
    print(f"Total output tokens:     {total_tokens}")
    print(f"Overall throughput:      {throughput:.2f} tokens/sec")


for concurrency in [1, 4, 8]:
    run_test(concurrency)
