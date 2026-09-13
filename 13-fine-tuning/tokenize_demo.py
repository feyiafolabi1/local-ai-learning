import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"

DATASET_PATH = Path(
    "~/local-ai-learning/13-fine-tuning/train.jsonl"
).expanduser()


def main():
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        fix_mistral_regex=True,
    )

    with open(DATASET_PATH, "r") as file:
        first_line = file.readline().strip()

    example = json.loads(first_line)

    messages = example["messages"]

    print("\nOriginal messages:")

    for message in messages:
        print(f"\n{message['role'].upper()}:")
        print(message["content"])

    # Convert chat messages into the exact format
    # expected by Ministral.
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print("\n" + "=" * 60)
    print("FORMATTED CHAT TEMPLATE")
    print("=" * 60)

    print(formatted_text)

    # Convert formatted text into token IDs.
    encoded = tokenizer(
        formatted_text,
        add_special_tokens=False,
    )

    token_ids = encoded["input_ids"]

    print("\n" + "=" * 60)
    print("TOKEN INFORMATION")
    print("=" * 60)

    print(f"\nNumber of tokens: {len(token_ids)}")

    print("\nFirst 30 token IDs:")
    print(token_ids[:30])

    print("\nFirst 30 decoded tokens:")

    for token_id in token_ids[:30]:
        token_text = tokenizer.decode([token_id])

        print(
            f"{token_id:>8} -> {repr(token_text)}"
        )


if __name__ == "__main__":
    main()