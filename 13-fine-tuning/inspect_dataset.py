import json
from pathlib import Path


DATASET_PATH = Path(
    "~/local-ai-learning/13-fine-tuning/train.jsonl"
).expanduser()


def main():
    examples = []

    with open(DATASET_PATH, "r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            example = json.loads(line)
            examples.append(example)

    print(f"\nDataset path: {DATASET_PATH}")
    print(f"Number of training examples: {len(examples)}")

    for index, example in enumerate(examples, start=1):
        print("\n" + "=" * 60)
        print(f"Example {index}")

        messages = example["messages"]

        for message in messages:
            role = message["role"]
            content = message["content"]

            print(f"\n{role.upper()}:")
            print(content)

    print("\n" + "=" * 60)
    print("Dataset inspection complete.")


if __name__ == "__main__":
    main()