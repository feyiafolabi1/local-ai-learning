import unsloth

import json
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastModel
from trl import SFTConfig, SFTTrainer


MODEL_NAME = (
    "unsloth/"
    "Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
)

DATASET_PATH = Path(
    "/workspace/13-fine-tuning/train_v2.jsonl"
)

OUTPUT_DIR = (
    "/workspace/13-fine-tuning/lora_adapter_v2"
)

MAX_SEQ_LENGTH = 2048


def load_training_data():
    rows = []

    with open(DATASET_PATH, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            example = json.loads(line)
            messages = example["messages"]

            user_message = next(
                message
                for message in messages
                if message["role"] == "user"
            )

            assistant_message = next(
                message
                for message in messages
                if message["role"] == "assistant"
            )

            rows.append(
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": user_message["content"],
                        }
                    ],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": assistant_message["content"],
                        }
                    ],
                }
            )

    return Dataset.from_list(rows)


def main():
    print("\n" + "=" * 60)
    print("CHAPTER 13 - QLORA FINE-TUNING V2")
    print("=" * 60)

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # Load quantized base model
    # --------------------------------------------------

    print("\nLoading 4-bit Ministral base model...")

    model, processor = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("\nBase model loaded.")

    # --------------------------------------------------
    # Use text tokenizer from multimodal processor
    # --------------------------------------------------

    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer

        print("\nMultimodal processor detected.")
        print(
            "Using processor.tokenizer "
            "for text-only training."
        )

    else:
        tokenizer = processor

        print("\nPlain tokenizer detected.")

    print(
        f"Tokenizer class: "
        f"{tokenizer.__class__.__name__}"
    )

    # --------------------------------------------------
    # Attach LoRA adapters
    # --------------------------------------------------

    print("\nAttaching LoRA adapters...")

    model = FastModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("\nLoRA adapters attached.")

    if hasattr(model, "print_trainable_parameters"):
        print("\nTrainable parameter summary:")
        model.print_trainable_parameters()

    # --------------------------------------------------
    # Load V2 dataset
    # --------------------------------------------------

    dataset = load_training_data()

    print("\nDataset loaded.")
    print(f"Training examples: {len(dataset)}")

    print("\nFirst prompt:")
    print(dataset[0]["prompt"])

    print("\nFirst completion:")
    print(dataset[0]["completion"])

    # --------------------------------------------------
    # Show formatted example
    # --------------------------------------------------

    example_messages = (
        dataset[0]["prompt"]
        + dataset[0]["completion"]
    )

    formatted_example = tokenizer.apply_chat_template(
        example_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print("\n" + "=" * 60)
    print("FORMATTED TRAINING EXAMPLE")
    print("=" * 60)

    print(formatted_example)

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Five complete passes through 40 examples.
        num_train_epochs=5,

        # Physical batch size.
        per_device_train_batch_size=2,

        # Effective batch size:
        # 2 × 4 = 8
        gradient_accumulation_steps=4,

        # LoRA optimizer settings.
        learning_rate=2e-4,
        warmup_steps=2,
        weight_decay=0.01,
        lr_scheduler_type="linear",

        # Show every optimizer step.
        logging_steps=1,

        seed=3407,

        max_length=MAX_SEQ_LENGTH,

        # Only assistant completion tokens
        # contribute to training loss.
        completion_only_loss=True,

        packing=False,

        # Avoid multiprocessing pickling issue.
        dataset_num_proc=1,

        # RTX 4090 supports BF16.
        bf16=True,
        fp16=False,

        report_to="none",

        # Save manually after training.
        save_strategy="no",
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    print("\nCreating SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\nTrainer created successfully.")

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING V2 TRAINING")
    print("=" * 60)

    print(
        "\nMethod: QLoRA"
        "\nBase model: 4-bit quantized and frozen"
        "\nTrainable parameters: LoRA adapters only"
        "\nDataset examples: 40"
        "\nEpochs: 5"
        "\nEffective batch size: 8"
    )

    trainer_stats = trainer.train()

    # --------------------------------------------------
    # Save V2 adapter
    # --------------------------------------------------

    print("\nTraining complete.")

    print(
        f"\nSaving V2 LoRA adapter to:"
        f"\n{OUTPUT_DIR}"
    )

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nV2 adapter saved successfully.")

    print("\nTraining statistics:")
    print(trainer_stats)


if __name__ == "__main__":
    main()
