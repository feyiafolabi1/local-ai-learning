import base64
import concurrent.futures
import json
import time
import requests

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

IMAGE_PATHS = [
    "/workspace/image1.jpg",
    "/workspace/image2.jpg",
    "/workspace/image3.jpg",
    "/workspace/image4.jpg",
    "/workspace/image5.jpg",
    "/workspace/image6.jpg",
    "/workspace/image7.jpg",
    "/workspace/image8.jpg",
]

PROMPT = (
    "Describe the main object or scene in this image, its most important "
    "visual details, and the surroundings in about 100 words."
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


IMAGE_DATA = [
    encode_image(path)
    for path in IMAGE_PATHS
]


def send_request(request_id):
    image_b64 = IMAGE_DATA[request_id]

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + image_b64
                            )
                        },
                    },
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 130,
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
    }

    request_start = time.perf_counter()

    first_token_time = None
    previous_token_time = None

    token_intervals = []
    completion_tokens = 0

    with requests.post(
        URL,
        json=payload,
        stream=True,
        timeout=180
    ) as response:

        if not response.ok:
            raise RuntimeError(
                f"\nRequest {request_id} failed\n"
                f"Image: {IMAGE_PATHS[request_id]}\n"
                f"HTTP status: {response.status_code}\n"
                f"Server response:\n{response.text}\n"
            )

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")

            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            chunk = json.loads(data)

            choices = chunk.get("choices", [])

            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")

                if content:
                    now = time.perf_counter()

                    if first_token_time is None:
                        first_token_time = now
                    elif previous_token_time is not None:
                        token_intervals.append(
                            now - previous_token_time
                        )

                    previous_token_time = now

            usage = chunk.get("usage")

            if usage:
                completion_tokens = usage.get(
                    "completion_tokens",
                    completion_tokens
                )

    request_end = time.perf_counter()

    latency = request_end - request_start

    if first_token_time is not None:
        ttft = first_token_time - request_start
    else:
        ttft = None

    if token_intervals:
        average_tpot = (
            sum(token_intervals)
            / len(token_intervals)
        )
    else:
        average_tpot = None

    return {
        "request_id": request_id,
        "image": IMAGE_PATHS[request_id],
        "latency": latency,
        "ttft": ttft,
        "tpot": average_tpot,
        "completion_tokens": completion_tokens,
    }


def run_test(concurrency):
    print()
    print("=" * 70)
    print(f"MULTIMODAL CONCURRENCY = {concurrency}")
    print("=" * 70)

    test_start = time.perf_counter()

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = {
            executor.submit(send_request, i): i
            for i in range(concurrency)
        }

        for future in concurrent.futures.as_completed(futures):
            request_id = futures[future]

            try:
                result = future.result()
                results.append(result)

                print(
                    f"Request {request_id} completed "
                    f"using {IMAGE_PATHS[request_id]}"
                )

            except Exception as e:
                print()
                print("REQUEST FAILED")
                print("-" * 70)
                print(e)
                print("-" * 70)

                raise

    test_end = time.perf_counter()

    total_wall_time = test_end - test_start

    latencies = [
        result["latency"]
        for result in results
    ]

    ttfts = [
        result["ttft"]
        for result in results
        if result["ttft"] is not None
    ]

    tpots = [
        result["tpot"]
        for result in results
        if result["tpot"] is not None
    ]

    total_tokens = sum(
        result["completion_tokens"]
        for result in results
    )

    average_latency = (
        sum(latencies)
        / len(latencies)
    )

    average_ttft = (
        sum(ttfts)
        / len(ttfts)
        if ttfts
        else None
    )

    average_tpot = (
        sum(tpots)
        / len(tpots)
        if tpots
        else None
    )

    throughput = (
        total_tokens / total_wall_time
        if total_wall_time > 0
        else 0
    )

    print()
    print("RESULTS")
    print("-" * 70)

    if average_ttft is not None:
        print(
            f"Average TTFT:            "
            f"{average_ttft * 1000:.2f} ms"
        )

    if average_tpot is not None:
        print(
            f"Average TPOT/chunk:      "
            f"{average_tpot * 1000:.2f} ms"
        )

    print(
        f"Average request latency: "
        f"{average_latency:.3f} s"
    )

    print(
        f"Total wall-clock time:   "
        f"{total_wall_time:.3f} s"
    )

    print(
        f"Total output tokens:     "
        f"{total_tokens}"
    )

    print(
        f"Overall throughput:      "
        f"{throughput:.2f} tokens/sec"
    )


for concurrency in [1, 4, 8]:
    run_test(concurrency)