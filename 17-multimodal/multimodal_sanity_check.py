import base64
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
    "Describe what you see in this image. "
    "Identify the main object or scene, important visual details, "
    "and the surroundings. Keep the answer concise."
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(image_path):
    image_b64 = encode_image(image_path)

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
        "max_tokens": 120,
    }

    response = requests.post(
        URL,
        json=payload,
        timeout=120
    )

    if not response.ok:
        raise RuntimeError(
            f"Failed for {image_path}\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    data = response.json()

    return data["choices"][0]["message"]["content"]


for image_path in IMAGE_PATHS:
    print()
    print("=" * 70)
    print(image_path)
    print("=" * 70)

    description = describe_image(image_path)

    print(description)
