import json
import time
import requests

URL = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": (
                "Explain the Key/Value attention cache used during "
                "transformer LLM decoding in about 150 words."
            ),
        }
    ],
    "temperature": 0.2,
    "max_tokens": 200,
    "stream": True,
}

print("Sending request...\n")

request_start = time.perf_counter()

first_token_time = None
previous_token_time = None

token_intervals = []
chunks_received = 0
generated_text = ""

with requests.post(URL, json=payload, stream=True) as response:
    response.raise_for_status()

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

        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")

        if content:
            now = time.perf_counter()

            if first_token_time is None:
                first_token_time = now
            elif previous_token_time is not None:
                token_intervals.append(now - previous_token_time)

            previous_token_time = now

            generated_text += content
            chunks_received += 1

            print(content, end="", flush=True)

request_end = time.perf_counter()

print("\n")
print("=" * 50)

total_latency = request_end - request_start

if first_token_time is not None:
    ttft = first_token_time - request_start
else:
    ttft = None

if token_intervals:
    average_inter_token = sum(token_intervals) / len(token_intervals)
else:
    average_inter_token = None

print("SERVING METRICS")
print("=" * 50)

if ttft is not None:
    print(f"TTFT:                    {ttft * 1000:.2f} ms")

if average_inter_token is not None:
    print(
        f"Average inter-chunk time: {average_inter_token * 1000:.2f} ms"
    )

print(f"Total latency:           {total_latency * 1000:.2f} ms")
print(f"Streaming chunks:        {chunks_received}")

if total_latency > 0:
    print(
        f"Approx. chunks/sec:      "
        f"{chunks_received / total_latency:.2f}"
    )
