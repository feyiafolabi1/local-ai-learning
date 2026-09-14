# Chapter 14 — Model Merging, GGUF, Quantization, llama.cpp, and Ollama

In this chapter I took the **QLoRA adapter from Chapter 13** and turned it into a practical local model I could run through Ollama.

The end-to-end flow was:

```text
V2 LoRA Adapter
      │
      ▼
Merge with Base Model
      │
      ▼
Standalone 16-bit Model
      │
      ▼
Convert to GGUF
      │
      ▼
Quantize to Q4_K_M
      │
      ▼
Run with llama.cpp
      │
      ▼
Import into Ollama
      │
      ▼
Run Fine-Tuned Model Locally
```

---

## 1. Starting Point — The LoRA Adapter

Chapter 13 produced:

```text
lora_adapter_v2
```

The important files were:

```text
adapter_config.json
adapter_model.safetensors
```

`adapter_config.json` described things like:

```text
Base model:
unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit

LoRA rank:
r = 16

LoRA alpha:
16

Target modules:
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

`adapter_model.safetensors` contained the actual learned LoRA parameters.

The adapter was only around:

```text
129 MB
```

because it did not contain the entire model.

---

## 2. Why the Adapter Cannot Run by Itself

The adapter only contains the learned changes.

```text
Base Model
    +
LoRA Adapter
    │
    ▼
Fine-Tuned Behavior
```

The core LoRA equation is:

```text
W' = W + BA
```

where:

```text
W  = original model weights
A/B = trained LoRA matrices
BA = learned update
W' = effective fine-tuned weights
```

---

## 3. Merging the Adapter

Merging permanently folds the LoRA update into the base model.

Before:

```text
Base Model
    +
LoRA Adapter
    │
    ▼
Fine-Tuned Model
```

After:

```text
Merged Standalone Model
        │
        ▼
Fine-Tuned Model
```

Conceptually:

```text
        Original W
            │
            │
            ├───────────────┐
            │               │
            ▼               ▼
        Base Path        LoRA A → B
            │               │
            └──────┬────────┘
                   ▼
                  ADD
                   │
                   ▼
             W' = W + BA
```

We used Unsloth to save:

```text
merged_16bit
```

The resulting merged model was roughly:

```text
~7.2 GB
```

---

## 4. Why Merge at Higher Precision?

The QLoRA training base was 4-bit, but we merged at higher precision first.

Why?

Because the LoRA update can contain small numerical changes.

```text
Base Weight
    +
Small LoRA Update
    │
    ▼
Accurate Higher-Precision Merge
    │
    ▼
Quantize Final Model
```

This is safer than trying to inject tiny updates directly into an already heavily quantized representation.

A useful distinction:

```text
Chapter 13:
4-bit base
→ reduce VRAM during training

Chapter 14:
Q4_K_M
→ reduce model size and memory use during inference
```

---

## 5. Merge vs Convert vs Quantize

These are three different operations:

```text
MERGE
=
Base Model + LoRA Adapter
→ Standalone Fine-Tuned Model


CONVERT
=
Hugging Face Model
→ GGUF Format


QUANTIZE
=
Higher Precision
→ Lower Precision Representation
```

---

## 6. What Is GGUF?

The merged model was still in Hugging Face format:

```text
config.json
tokenizer files
safetensors shards
```

We converted it into:

```text
GGUF
```

GGUF is a model file format used heavily by llama.cpp for efficient local inference.

Flow:

```text
Hugging Face Model
        │
        ▼
convert_hf_to_gguf.py
        │
        ▼
F16 GGUF
```

Important:

```text
GGUF
≠
quantization
```

A GGUF can contain:

```text
F16
Q8
Q6
Q5
Q4
```

---

## 7. Tokenizer Conversion Problem

The first GGUF conversion failed because Ministral uses a newer tokenizer setup.

The converter initially looked for:

```text
tokenizer.model
```

but Ministral uses:

```text
tekken.json
```

We added the correct tokenizer files from the original model repository and preserved:

```text
tokenizer.json
```

as:

```text
tokenizer.json.backup
```

That allowed llama.cpp to use the correct Tekken tokenizer path.

Debugging flow:

```text
GGUF Conversion
      │
      ▼
Tokenizer Error
      │
      ▼
Inspect Ministral Files
      │
      ▼
Add tekken.json
      │
      ▼
Use Correct Tokenizer Path
      │
      ▼
Conversion Works ✅
```

---

## 8. F16 GGUF

The merged Hugging Face model was converted into:

```text
ministral-3-3b-finetuned-v2-f16.gguf
```

Size:

```text
~6.4 GB
```

At this point we had mostly changed the file format.

The big compression step had not happened yet.

---

## 9. Quantization

We then quantized the F16 GGUF using:

```text
Q4_K_M
```

Flow:

```text
F16 GGUF
~6.4 GB
    │
    ▼
llama-quantize
    │
    ▼
Q4_K_M GGUF
~2.0 GB
```

That was roughly a:

```text
~69% size reduction
```

---

## 10. What Is Q4_K_M?

Q4_K_M is a llama.cpp quantization scheme.

At a high level:

```text
Q4
→ roughly 4-bit-class compression

K
→ newer block quantization family

M
→ medium balance variant
```

It is not simply:

```text
"every weight is exactly 4 bits"
```

There are also scales, block metadata, and different treatment for some tensors.

The main idea is:

```text
smaller model
+
less memory traffic
+
faster / easier local deployment
```

while trying to preserve model quality.

---

## 11. Testing with llama.cpp

Before bringing the model into Ollama, I tested the quantized GGUF directly using:

```text
llama-cli
```

Prompt:

```text
What is electromigration?
```

The model still returned:

```text
CONCEPT:
...

WHY IT MATTERS:
...

EXAMPLE:
...
```

So the learned behavior survived:

```text
LoRA Merge           ✅
GGUF Conversion      ✅
Q4 Quantization      ✅
```

The llama.cpp test also showed performance stats around:

```text
Prompt processing:
~162 tokens/sec

Generation:
~15.2 tokens/sec
```

on the RTX 4090.

---

## 12. What Is llama.cpp?

llama.cpp is an efficient local inference runtime.

It handles things like:

```text
GGUF loading
quantized inference
token generation
KV cache
CPU / GPU execution
memory management
```

The direct stack looked like:

```text
Prompt
  │
  ▼
llama.cpp
  │
  ▼
Q4_K_M GGUF
  │
  ▼
GPU
  │
  ▼
Generated Tokens
```

---

## 13. Bringing the Model Into Ollama

The final Q4 model was downloaded to the Mac:

```text
ministral-3-3b-finetuned-v2-q4_k_m.gguf
```

Then I created a `Modelfile`:

```text
FROM ./ministral-3-3b-finetuned-v2-q4_k_m.gguf

PARAMETER temperature 0.2
PARAMETER num_ctx 4096
```

Then:

```text
ollama create ministral-ft-v2 -f Modelfile
```

and:

```text
ollama run ministral-ft-v2
```

---

## 14. Final Local Inference Flow

```text
User Prompt
    │
    ▼
Ollama
    │
    ▼
Q4_K_M GGUF
    │
    ▼
llama.cpp-style inference
    │
    ▼
Apple Silicon / Metal / CPU
    │
    ▼
Fine-Tuned Response
```

The model consistently preserved:

```text
CONCEPT:
WHY IT MATTERS:
EXAMPLE:
```

even on local Mac inference.

---

## 15. Important Result — Behavior vs Knowledge

The format behavior generalized very well, but some factual answers were still imperfect.

For example:

```text
"What is KV cache?"
```

was interpreted more like a generic key-value cache than transformer attention KV cache.

This reinforced an important lesson:

```text
Fine-Tuning
→ changes behavior

RAG
→ improves access to knowledge

Quantization
→ improves deployment efficiency
```

So:

```text
Behavior Quality
≠
Factual Quality
```

---

## 16. Model Artifact Sizes

```text
V2 LoRA Adapter
~129 MB
      │
      ▼
Merged 16-bit HF Model
~7.2 GB
      │
      ▼
F16 GGUF
~6.4 GB
      │
      ▼
Q4_K_M GGUF
~2.0–2.15 GB
```

---

## 17. Files in This Chapter

```text
14-quantization-export/
│
├── README.md
├── merge_v2.py
├── Modelfile
└── ministral-3-3b-finetuned-v2-q4_k_m.gguf
```

The GGUF is kept locally but excluded from GitHub.

GitHub should contain:

```text
README.md
merge_v2.py
Modelfile
```

---

## 18. Full Chapter Flow

```text
Chapter 13
V2 LoRA Adapter
      │
      ▼
Inspect Adapter
      │
      ▼
Rent RTX 4090
      │
      ▼
Load Base + Adapter
      │
      ▼
Merge
W' = W + BA
      │
      ▼
16-bit HF Model
~7.2 GB
      │
      ▼
Convert to GGUF
      │
      ▼
Tokenizer Issue
      │
      ▼
Fix Tekken Tokenizer
      │
      ▼
F16 GGUF
~6.4 GB
      │
      ▼
Quantize
      │
      ▼
Q4_K_M GGUF
~2.0 GB
      │
      ▼
Test with llama.cpp
      │
      ▼
Behavior Preserved ✅
      │
      ▼
Download to Mac
      │
      ▼
Import into Ollama
      │
      ▼
Run Locally
      │
      ▼
CONCEPT / WHY IT MATTERS / EXAMPLE ✅
```

---

# Key Takeaways

- LoRA adapters are small because they only contain learned low-rank updates.
- A LoRA adapter still needs the original base model unless it is merged.
- Merging folds the LoRA update into the original model weights.
- Merge, conversion, and quantization are separate operations.
- GGUF is a file format, not a quantization level.
- Q4_K_M reduced the final model from about 6.4 GB to about 2 GB.
- Quantization is lossy, so behavior should be tested afterward.
- llama.cpp is the local inference engine.
- Ollama provides a much easier model-management layer above local inference.
- The fine-tuned behavior survived merge, GGUF conversion, quantization, llama.cpp, and Ollama.
- The final result was a fine-tuned model trained in the cloud but deployed locally on a Mac.

---

# Chapter Result

At the start of this chapter I had:

```text
V2 LoRA Adapter
~129 MB
```

At the end I had:

```text
ministral-ft-v2
```

running locally through Ollama from a:

```text
Q4_K_M GGUF
~2 GB
```

The complete deployment pipeline was:

```text
LoRA Adapter
      │
      ▼
Merge
      │
      ▼
Standalone Model
      │
      ▼
GGUF
      │
      ▼
Quantize
      │
      ▼
llama.cpp
      │
      ▼
Ollama
      │
      ▼
Local Fine-Tuned Model
```

That completed the transition from:

```text
"I trained an adapter."
```

to:

```text
"I deployed my own fine-tuned local model."
```