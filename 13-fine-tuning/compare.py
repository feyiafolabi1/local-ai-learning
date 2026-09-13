import unsloth

import torch
from unsloth import FastModel


BASE_MODEL = (
    "unsloth/"
    "Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
)

ADAPTER_PATH = (
    "/workspace/13-fine-tuning/lora_adapter"
)

MAX_SEQ_LENGTH = 2048

QUESTIONS = [
    # Seen during training
    "What is clock gating?",

    # Not seen during training
    "What is a voltage regulator?",

    # Another unseen semiconductor concept
    "What is voltage droop?",
]


def get_text_tokenizer(processor):
    if hasattr(processor, "tokenizer"):
        return processor.tokenizer

    return processor


def generate_answer(
    model,
    tokenizer,
    question,
):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,

            # Deterministic generation.
            do_sample=False,

            max_new_tokens=220,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


def main():
    print("\n" + "=" * 70)
    print("BASE VS QLORA FINE-TUNED EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load base model
    # --------------------------------------------------

    print("\nLoading base model...")

    base_model, base_processor = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    base_tokenizer = get_text_tokenizer(
        base_processor
    )

    FastModel.for_inference(base_model)

    # --------------------------------------------------
    # Generate base responses
    # --------------------------------------------------

    base_answers = {}

    for question in QUESTIONS:
        print(
            f"\nGenerating BASE response for:\n"
            f"{question}"
        )

        base_answers[question] = generate_answer(
            base_model,
            base_tokenizer,
            question,
        )

    # Free base model before loading adapter model.
    del base_model

    torch.cuda.empty_cache()

    # --------------------------------------------------
    # Load fine-tuned model
    # --------------------------------------------------

    print(
        "\nLoading base model + "
        "QLoRA adapter..."
    )

    tuned_model, tuned_processor = FastModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    tuned_tokenizer = get_text_tokenizer(
        tuned_processor
    )

    FastModel.for_inference(tuned_model)

    # --------------------------------------------------
    # Generate fine-tuned responses
    # --------------------------------------------------

    tuned_answers = {}

    for question in QUESTIONS:
        print(
            f"\nGenerating FINE-TUNED response for:\n"
            f"{question}"
        )

        tuned_answers[question] = generate_answer(
            tuned_model,
            tuned_tokenizer,
            question,
        )

    # --------------------------------------------------
    # Print comparison
    # --------------------------------------------------

    for question in QUESTIONS:

        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        print("\nBASE MODEL")
        print("-" * 70)
        print(base_answers[question])

        print("\nFINE-TUNED MODEL")
        print("-" * 70)
        print(tuned_answers[question])

        print("\nFORMAT CHECK")

        answer = tuned_answers[question]

        required_headers = [
            "CONCEPT:",
            "WHY IT MATTERS:",
            "EXAMPLE:",
        ]

        for header in required_headers:
            if header in answer:
                print(f"✅ {header}")
            else:
                print(f"❌ {header}")


if __name__ == "__main__":
    main()