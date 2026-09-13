import unsloth

import torch
from unsloth import FastModel


BASE_MODEL = (
    "unsloth/"
    "Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
)

V1_ADAPTER = (
    "/workspace/13-fine-tuning/lora_adapter"
)

V2_ADAPTER = (
    "/workspace/13-fine-tuning/lora_adapter_v2"
)

MAX_SEQ_LENGTH = 2048


QUESTIONS = [
    "What is electromigration?",
    "Explain a PLL.",
    "What is thermal throttling?",
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

            # Deterministic evaluation.
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


def load_model(model_path):
    model, processor = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    tokenizer = get_text_tokenizer(
        processor
    )

    FastModel.for_inference(model)

    return model, tokenizer


def generate_for_all_questions(
    model_path,
    label,
):
    print("\n" + "=" * 70)
    print(f"LOADING {label}")
    print("=" * 70)

    model, tokenizer = load_model(
        model_path
    )

    answers = {}

    for question in QUESTIONS:
        print(
            f"\nGenerating {label} response for:"
            f"\n{question}"
        )

        answers[question] = generate_answer(
            model,
            tokenizer,
            question,
        )

    # Remove model before loading next one.
    del model

    torch.cuda.empty_cache()

    return answers


def check_format(answer):
    required_headers = [
        "CONCEPT:",
        "WHY IT MATTERS:",
        "EXAMPLE:",
    ]

    results = {}

    for header in required_headers:
        results[header] = header in answer

    return results


def main():
    print("\n" + "=" * 70)
    print("BASE vs QLoRA V1 vs QLoRA V2")
    print("=" * 70)

    # --------------------------------------------------
    # Base
    # --------------------------------------------------

    base_answers = generate_for_all_questions(
        BASE_MODEL,
        "BASE MODEL",
    )

    # --------------------------------------------------
    # V1
    # --------------------------------------------------

    v1_answers = generate_for_all_questions(
        V1_ADAPTER,
        "QLoRA V1",
    )

    # --------------------------------------------------
    # V2
    # --------------------------------------------------

    v2_answers = generate_for_all_questions(
        V2_ADAPTER,
        "QLoRA V2",
    )

    # --------------------------------------------------
    # Print comparisons
    # --------------------------------------------------

    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        print("\nBASE MODEL")
        print("-" * 70)
        print(base_answers[question])

        print("\nQLoRA V1")
        print("-" * 70)
        print(v1_answers[question])

        print("\nQLoRA V2")
        print("-" * 70)
        print(v2_answers[question])

        print("\nV2 FORMAT CHECK")
        print("-" * 70)

        format_results = check_format(
            v2_answers[question]
        )

        for header, passed in format_results.items():
            if passed:
                print(f"✅ {header}")
            else:
                print(f"❌ {header}")

    # --------------------------------------------------
    # Overall V2 score
    # --------------------------------------------------

    total_checks = 0
    passed_checks = 0

    for question in QUESTIONS:
        results = check_format(
            v2_answers[question]
        )

        for passed in results.values():
            total_checks += 1

            if passed:
                passed_checks += 1

    print("\n" + "=" * 70)
    print("V2 FORMAT SUMMARY")
    print("=" * 70)

    print(
        f"\nPassed format checks: "
        f"{passed_checks}/{total_checks}"
    )


if __name__ == "__main__":
    main()
