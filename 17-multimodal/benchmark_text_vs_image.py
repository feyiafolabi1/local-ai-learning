import base64
import json
import time
import requests

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/workspace/test_car.jpg"

TEXT_PROMPT = (
    "Explain why GPU memory bandwidth matters during LLM decode "
    "in about 120 words."
)

IMAGE_PROMPT = (
    "Describe the main object in this image, its color, and the surroundings "
    "in about 120 words."
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def benchmark_request(messages, label):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 150,
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
    generated_text = ""

    with requests.post(
        URL,
        json=payload,
        stream=True,
        timeout=120
    ) as response:

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

                    generated_text += content

            usage = chunk.get("usage")

            if usage:
                completion_tokens = usage.get(
                    "completion_tokens",
                    completion_tokens
                )

    request_end = time.perf_counter()

    total_latency = request_end - request_start

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

    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    if ttft is not None:
        print(f"TTFT:              {ttft * 1000:.2f} ms")

    if average_tpot is not None:
        print(
            f"Avg TPOT/chunk:    "
            f"{average_tpot * 1000:.2f} ms"
        )

    print(
        f"Total latency:     "
        f"{total_latency:.3f} s"
    )

    print(
        f"Completion tokens: "
        f"{completion_tokens}"
    )

    print()
    print("Response:")
    print(generated_text)

    return {
        "label": label,
        "ttft": ttft,
        "tpot": average_tpot,
        "latency": total_latency,
        "completion_tokens": completion_tokens,
    }


# --------------------------------------------------
# TEXT-ONLY REQUEST
# --------------------------------------------------

text_messages = [
    {
        "role": "user",
        "content": TEXT_PROMPT,
    }
]

text_result = benchmark_request(
    text_messages,
    "TEXT ONLY"
)


# --------------------------------------------------
# IMAGE + TEXT REQUEST
# --------------------------------------------------

image_b64 = encode_image(IMAGE_PATH)

image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": IMAGE_PROMPT,
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
]

image_result = benchmark_request(
    image_messages,
    "IMAGE + TEXT"
)


# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)

if (
    text_result["ttft"] is not None
    and image_result["ttft"] is not None
):
    ttft_ratio = (
        image_result["ttft"]
        / text_result["ttft"]
    )

    print(
        f"TTFT increase:     "
        f"{ttft_ratio:.2f}x"
    )

if (
    text_result["tpot"] is not None
    and image_result["tpot"] is not None
):
    tpot_ratio = (
        image_result["tpot"]
        / text_result["tpot"]
    )

    print(
        f"TPOT increase:     "
        f"{tpot_ratio:.2f}x"
    )

latency_ratio = (
    image_result["latency"]
    / text_result["latency"]
)

print(
    f"Latency increase:  "
    f"{latency_ratio:.2f}x"
)
