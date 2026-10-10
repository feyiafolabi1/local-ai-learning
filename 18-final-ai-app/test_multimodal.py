# ==================================================
# CHAPTER 18 — MULTIMODAL INFERENCE TEST
# ==================================================
#
# Tests local image understanding with Ministral 3B.
#
# Measures:
#   Time to first visible text
#   Model loading
#   Prompt processing
#   Token generation
#   Total inference latency
#
# ==================================================

import shlex
import time
from pathlib import Path

import ollama


MODEL = "ministral-3:3b"

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def ns_to_ms(value):
    if value is None:
        return None

    return value / 1_000_000


def format_ms(value):
    if value is None:
        return "N/A"

    return f"{value:.2f} ms"


def get_image_path():

    raw = input(
        "\nImage path (drag an image here): "
    ).strip()

    parts = shlex.split(raw)

    if len(parts) != 1:
        raise ValueError(
            "Please provide exactly one image path."
        )

    path = Path(parts[0]).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Supported formats: PNG, JPG, JPEG, WEBP."
        )

    return path


def run_test():

    print()
    print("=" * 70)
    print("CHAPTER 18 — MULTIMODAL INFERENCE")
    print("=" * 70)

    print()
    print(f"Model: {MODEL}")

    image_path = get_image_path()

    question = input(
        "\nQuestion about the image: "
    ).strip()

    if not question:
        question = "Describe what you see in this image."

    print()
    print(f"Image: {image_path.name}")
    print(f"Question: {question}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI engineering assistant. "
                "Analyze the supplied image carefully. "
                "Describe only what is visible. "
                "If text is unclear, say so. "
                "Do not invent details."
            ),
        },
        {
            "role": "user",
            "content": question,
            "images": [str(image_path)],
        },
    ]

    print()
    print("Generating response...")
    print("-" * 70)

    start = time.perf_counter()

    first_text_ms = None
    final_chunk = None

    stream = ollama.chat(
        model=MODEL,
        messages=messages,
        stream=True,
        keep_alive="5m",
        options={
            "temperature": 0.2,
            "num_predict": 300,
        },
    )

    for chunk in stream:

        final_chunk = chunk

        content = chunk.message.content or ""

        if content:

            if first_text_ms is None:
                first_text_ms = (
                    time.perf_counter() - start
                ) * 1000

            print(content, end="", flush=True)

    print()

    wall_ms = (
        time.perf_counter() - start
    ) * 1000

    if final_chunk is None:
        raise RuntimeError(
            "No response received from Ollama."
        )

    load_ms = ns_to_ms(
        final_chunk.load_duration
    )

    prefill_ms = ns_to_ms(
        final_chunk.prompt_eval_duration
    )

    decode_ms = ns_to_ms(
        final_chunk.eval_duration
    )

    input_tokens = final_chunk.prompt_eval_count
    output_tokens = final_chunk.eval_count

    speed = None

    if decode_ms and output_tokens:
        speed = output_tokens / (decode_ms / 1000)

    print()
    print("=" * 70)
    print("MULTIMODAL PERFORMANCE")
    print("=" * 70)

    print()
    print(f"TTFT:              {format_ms(first_text_ms)}")
    print(f"Model loading:     {format_ms(load_ms)}")
    print(f"Prompt processing: {format_ms(prefill_ms)}")
    print(f"Token generation:  {format_ms(decode_ms)}")
    print(f"Total latency:     {format_ms(wall_ms)}")

    print()
    print(f"Input tokens:      {input_tokens}")
    print(f"Output tokens:     {output_tokens}")

    if speed is not None:
        print(f"Decode speed:      {speed:.2f} tok/s")

    print(f"Stop reason:       {final_chunk.done_reason}")


if __name__ == "__main__":
    run_test()