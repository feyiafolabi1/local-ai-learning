import unsloth

import os
import torch
from unsloth import FastModel


ADAPTER_PATH = "/workspace/lora_adapter_v2"
MERGED_OUTPUT = "/workspace/ministral-3-3b-finetuned-v2-merged"

MAX_SEQ_LENGTH = 2048


def main():
    print("=" * 70)
    print("CHAPTER 14 - MERGE QLORA ADAPTER INTO BASE MODEL")
    print("=" * 70)

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    print("\nLoading base model + V2 LoRA adapter...")

    model, processor = FastModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("\nAdapter-loaded model ready.")

    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    print("\nSaving merged model...")

    model.save_pretrained_merged(
        MERGED_OUTPUT,
        tokenizer,
        save_method="merged_16bit",
    )

    print("\nMerge complete.")

    print(f"\nMerged model saved to:\n{MERGED_OUTPUT}")

    print("\nFiles:")
    for name in sorted(os.listdir(MERGED_OUTPUT)):
        path = os.path.join(MERGED_OUTPUT, name)
        size_gb = os.path.getsize(path) / (1024 ** 3)

        print(
            f"{name:50s} "
            f"{size_gb:.3f} GB"
        )


if __name__ == "__main__":
    main()