# Chapter 13 — Fine-Tuning with LoRA and QLoRA

In this chapter I moved from simply **using** language models to actually **changing their behavior through training**.

The goal was not just to run a fine-tuning script. I wanted to understand what was happening underneath:

- How fine-tuning differs from RAG
- What supervised fine-tuning actually teaches a model
- How causal language-model training works
- What teacher forcing means
- How tokenization and chat templates affect training
- What labels are
- Why some tokens are ignored during loss calculation
- What completion-only loss means
- How LoRA works mathematically
- What the A and B LoRA matrices actually represent
- How QLoRA differs from LoRA
- Why quantization dramatically reduces training memory
- Which model layers we adapted
- What LoRA rank and alpha mean
- How batch size and gradient accumulation interact
- What epochs and optimizer steps mean
- Why gradient checkpointing saves memory
- Why decreasing training loss does not automatically mean successful fine-tuning
- Why dataset design and prompt diversity matter
- How to evaluate memorization vs generalization

The chapter eventually became a two-stage experiment:

```text
V1
│
├── 10 examples
├── limited prompt diversity
├── loss dropped substantially
├── model behavior changed
└── exact desired format FAILED
        │
        ▼
Analyze failure
        │
        ▼
Redesign dataset
        │
        ▼
V2
│
├── 40 examples
├── varied prompt wording
├── same output structure every time
├── train again
└── test on unseen concepts
        │
        ▼
9 / 9 format checks PASSED


```

---

# 1. Fine-Tuning vs RAG

Before this chapter, I had already used RAG to give local models access to external information. Fine-tuning solves a different problem.

## RAG

RAG changes the information available to the model **at runtime**.

```text
User Question
      │
      ▼
Retrieve External Information
      │
      ▼
Add Retrieved Information
to the Context Window
      │
      ▼
LLM Generates Answer
      │
      ▼
Base Model Weights
DO NOT CHANGE
```

The model temporarily receives additional knowledge.

Examples of external information could include:

- PDFs
- documentation
- databases
- vector stores
- private company data
- websites

A useful mental model is:

```text
RAG
=
Give the model better information
at inference time
```

## Fine-Tuning

Fine-tuning actually changes trainable parameters.

```text
Training Examples
       │
       ▼
Model Prediction
       │
       ▼
Compare With Desired Answer
       │
       ▼
Calculate Loss
       │
       ▼
Backpropagation
       │
       ▼
Update Trainable Parameters
       │
       ▼
Model Behavior Changes
```

So:

```text
RAG
│
└── changes CONTEXT / KNOWLEDGE available at runtime

Fine-Tuning
│
└── changes BEHAVIOR encoded in parameters
```

For example:

```text
Need current AMD product information?
        │
        ▼
RAG makes sense.


Want the assistant to always respond
using a particular structure?
        │
        ▼
Fine-tuning makes sense.
```

---

# 2. What Model Did I Fine-Tune?

The model family used in this chapter was:

```text
Ministral 3 3B Instruct
```

The actual training checkpoint was:

```text
unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit
```

The model was loaded in **4-bit quantized form**.

Training environment:

```text
RunPod
│
└── NVIDIA RTX 4090
    └── 24 GB VRAM
```

Main software:

- PyTorch
- Unsloth
- Hugging Face Transformers
- TRL
- PEFT

The base model remained frozen.

Only LoRA adapter parameters were trained.

Because the frozen base model was quantized, the overall training technique was:

```text
QLoRA
```

---

# 3. Why Use an Instruct Model?

A raw base language model primarily learns:

```text
Previous Tokens
      │
      ▼
Predict Next Token
```

An instruct model has already undergone additional training to behave more like an assistant:

```text
User Instruction
      │
      ▼
Understand Request
      │
      ▼
Generate Helpful Response
```

Our goal was not:

```text
Raw Language Model
      │
      ▼
Teach It How to Become an Assistant
```

Our goal was:

```text
Existing Assistant
      │
      ▼
Specialize Its Behavior
```

So starting from Ministral 3 3B Instruct made sense.

---

# 4. Supervised Fine-Tuning

The type of training used was:

```text
Supervised Fine-Tuning
        │
        ▼
       SFT
```

The dataset contains examples of:

```text
USER INPUT
     │
     ▼
DESIRED ASSISTANT OUTPUT
```

For example:

```text
USER:

What is clock gating?


ASSISTANT:

CONCEPT:
Clock gating disables the clock signal to inactive logic blocks.

WHY IT MATTERS:
It reduces unnecessary switching activity and therefore lowers
dynamic power.

EXAMPLE:
An idle GPU compute unit can have its clock gated until new work arrives.
```

The training dataset is effectively telling the model:

```text
"When you receive something like this..."
                    │
                    ▼
"...I want you to behave like this."
```

---

# 5. How Causal Language Model Training Works

A causal language model fundamentally predicts:

```text
the next token
```

Suppose the desired answer contains:

```text
Clock gating reduces dynamic power.
```

Training conceptually looks like:

```text
Clock
  │
  ▼
predict:
gating


Clock gating
     │
     ▼
predict:
reduces


Clock gating reduces
          │
          ▼
predict:
dynamic


Clock gating reduces dynamic
                  │
                  ▼
predict:
power
```

At every token position:

```text
Previous Tokens
      │
      ▼
Transformer
      │
      ▼
Probability Distribution
Over Possible Next Tokens
      │
      ▼
Compare Against Correct Token
      │
      ▼
Loss
```

The desired assistant answer therefore acts like an **answer key**.

---

# 6. Teacher Forcing

During training, the model is given the correct previous tokens while it learns to predict the next one.

For example:

```text
CONCEPT:
Clock gating reduces power.
```

The training process can be thought of as:

```text
Input:
CONCEPT:

Target:
Clock
```

then:

```text
Input:
CONCEPT: Clock

Target:
gating
```

then:

```text
Input:
CONCEPT: Clock gating

Target:
reduces
```

then:

```text
Input:
CONCEPT: Clock gating reduces

Target:
power
```

This is called:

```text
Teacher Forcing
```

The model is not forced to generate all of its previous training tokens correctly before it can learn the next token.

---

# 7. From Messages to Transformer Input

Our datasets start as chat messages.

Example:

```python
[
    {
        "role": "user",
        "content": "What is clock gating?"
    },
    {
        "role": "assistant",
        "content": "CONCEPT:\n..."
    }
]
```

But a transformer does not directly process Python strings.

The actual flow is:

```text
JSON / Python Messages
          │
          ▼
      Chat Template
          │
          ▼
Specially Formatted Text
          │
          ▼
       Tokenizer
          │
          ▼
       Token IDs
          │
          ▼
    Embedding Lookup
          │
          ▼
Embedding Vectors
          │
          ▼
     Transformer
```

---

# 8. Chat Templates

Different instruct models expect conversations in different formats.

The tokenizer's chat template converts:

```text
role = user
content = What is clock gating?
```

into the exact special-token structure expected by Ministral.

While experimenting with the tokenizer, I discovered that a short visible prompt could turn into hundreds of tokens because Ministral's built-in chat template contains additional system instructions and control tokens.

This gave me an important mental model:

```text
What the user sees
        │
        ▼
is not necessarily
        │
        ▼
exactly what the model receives
```

The `tokenize_demo.py` script was used to inspect this transformation directly.

---

# 9. Tokens and Labels

During language-model training, tokens can have corresponding training targets called:

```text
labels
```

Conceptually:

```text
TOKEN                       LABEL
-----                       -----

What                        -100
is                          -100
clock                       -100
gating                      -100
?                           -100

CONCEPT:                    CONCEPT:
Clock                       Clock
gating                      gating
disables                    disables
...
```

The special value:

```text
-100
```

means:

```text
IGNORE THIS POSITION
WHEN CALCULATING LOSS
```

The model can therefore see the user's question as context without being trained to reproduce the question.

---

# 10. Completion-Only Loss

Our training configuration used:

```python
completion_only_loss=True
```

Conceptually:

```text
USER PROMPT
"What is clock gating?"
       │
       │
       ├── visible to the model
       │
       └── not graded
       │
       ▼

---------------------------------

ASSISTANT COMPLETION
"CONCEPT:
 Clock gating..."
       │
       │
       └── contributes to loss
       │
       ▼
TRAINING OBJECTIVE
```

So:

```text
Prompt
=
Context

Completion
=
Answer Key
```

---

# 11. Full SFT Training Flow

```text
Training Example
       │
       ▼
Apply Chat Template
       │
       ▼
Tokenize
       │
       ▼
Forward Pass
       │
       ▼
Predict Next-Token Probabilities
       │
       ▼
Compare Against
Correct Assistant Tokens
       │
       ▼
Calculate Loss
       │
       ▼
Backpropagation
       │
       ▼
Calculate Gradients
       │
       ▼
Optimizer Update
       │
       ▼
Trainable Parameters
Slightly Change
       │
       ▼
Repeat
```

---

# 12. Why Not Fine-Tune Every Parameter?

The model contained:

```text
3,882,841,088 total parameters
```

Full fine-tuning would require updating billions of parameters.

Training also requires more than storing the weights themselves.

For trainable parameters, memory may be required for:

```text
Weights
   +
Gradients
   +
Optimizer State
   +
Activations
```

This becomes expensive very quickly.

Instead, we used:

```text
LoRA
```

---

# 13. What Is LoRA?

LoRA stands for:

```text
Low-Rank Adaptation
```

Suppose a layer contains an original weight matrix:

```text
W
```

Full fine-tuning modifies `W` directly.

LoRA instead keeps `W` frozen and learns a small correction:

```text
ΔW
```

The effective weight becomes:

```text
W' = W + ΔW
```

Rather than learning a full-sized `ΔW`, LoRA approximates it using two smaller matrices:

```text
ΔW ≈ B × A
```

---

# 14. LoRA Flow Diagram

```text
                         ORIGINAL LAYER

Input
  │
  ├──────────────────────────────────────────────┐
  │                                              │
  ▼                                              ▼
┌─────────────────────┐                 ┌─────────────────┐
│ Original Matrix W   │                 │ LoRA Matrix A   │
│                     │                 │                 │
│       FROZEN        │                 │    TRAINABLE    │
└─────────────────────┘                 └─────────────────┘
  │                                              │
  │                                              ▼
  │                                    ┌─────────────────┐
  │                                    │ LoRA Matrix B   │
  │                                    │                 │
  │                                    │    TRAINABLE    │
  │                                    └─────────────────┘
  │                                              │
  │                                              │
  └───────────────────────┐        ┌─────────────┘
                          │        │
                          ▼        ▼
                             ADD
                              │
                              ▼
                           Output


Original contribution:

    Wx

LoRA contribution:

    BAx

Final result:

    Wx + BAx

Equivalent effective weight:

    W' = W + BA
```

The important part is:

```text
Original Matrix W
      │
      └── FROZEN


LoRA Matrix A
      │
      └── TRAINABLE


LoRA Matrix B
      │
      └── TRAINABLE
```

---

# 15. Why Is It Called "Low-Rank"?

Imagine one original matrix has dimensions:

```text
4096 × 4096
```

A complete update matrix would require:

```text
4096 × 4096
=
16,777,216 parameters
```

Suppose LoRA uses:

```text
rank r = 16
```

Instead we learn:

```text
A = 16 × 4096

B = 4096 × 16
```

Parameter count:

```text
A:

16 × 4096
=
65,536


B:

4096 × 16
=
65,536


Combined:

131,072 parameters
```

Compared with:

```text
16,777,216
```

for a full matrix.

That is a massive reduction.

---

# 16. What Do the A and B Matrices Actually Learn?

The dataset does **not** simply get stored inside A and B.

Instead:

```text
Training Example
       │
       ▼
Model Prediction
       │
       ▼
Loss
       │
       ▼
Backpropagation
       │
       ▼
Gradients
       │
       ▼
Optimizer
       │
       ▼
Tiny Changes to A and B
       │
       ▼
Next Training Example
```

Over many repetitions:

```text
Initial A/B
    │
    ▼
Tiny Update
    │
    ▼
Tiny Update
    │
    ▼
Tiny Update
    │
    ▼
Tiny Update
    │
    ▼
...
    │
    ▼
Adapter Encodes
Useful Behavioral Changes
```

---

# 17. Our LoRA Configuration

We used:

```text
LoRA Rank:
r = 16

LoRA Alpha:
16

LoRA Dropout:
0
```

Adapters were attached to:

```text
ATTENTION

q_proj
k_proj
v_proj
o_proj


MLP

gate_proj
up_proj
down_proj
```

---

# 18. Where the LoRA Adapters Sit

A simplified transformer layer:

```text
Input Token Representations
           │
           ▼
┌──────────────────────────────┐
│          ATTENTION           │
│                              │
│ q_proj   ◄──── LoRA A/B      │
│ k_proj   ◄──── LoRA A/B      │
│ v_proj   ◄──── LoRA A/B      │
│ o_proj   ◄──── LoRA A/B      │
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│             MLP              │
│                              │
│ gate_proj ◄──── LoRA A/B     │
│ up_proj   ◄──── LoRA A/B     │
│ down_proj ◄──── LoRA A/B     │
└──────────────────────────────┘
           │
           ▼
Next Transformer Layer
```

An important clarification:

```text
q_proj itself
IS NOT TRAINABLE

Original q_proj
=
FROZEN

LoRA A/B attached to q_proj
=
TRAINABLE
```

The same applies to all of the other targeted matrices.

---

# 19. How Many Parameters Did We Train?

Unsloth reported:

```text
Trainable parameters:
33,751,040

Total parameters:
3,882,841,088
```

Percentage trainable:

```text
33,751,040
───────────── × 100
3,882,841,088

≈ 0.8692%
```

Approximately:

```text
0.87% TRAINABLE

99.13% FROZEN
```

Visualized:

```text
Entire Model
██████████████████████████████████████████████████

Frozen Base Parameters
█████████████████████████████████████████████████░
~99.13%

Trainable LoRA Parameters
░
~0.87%
```

Despite training less than 1% of the model, we were able to noticeably change its output behavior.

---

# 20. What Is QLoRA?

QLoRA combines:

```text
Quantization
     +
LoRA
```

## LoRA

```text
Normal-Precision Base Model
            │
            │ FROZEN
            ▼
      Original Weights
            +
      LoRA A/B Adapters
            │
            ▼
     Fine-Tuned Behavior
```

## QLoRA

```text
Base Model
    │
    ▼
Quantize Base
    │
    ▼
4-bit Base Model
    │
    │ FROZEN
    ▼
Quantized Base Weights
        +
Trainable LoRA A/B
        │
        ▼
Fine-Tuned Behavior
```

Therefore:

```text
LoRA
=
Frozen Base
+
Trainable LoRA Adapters


QLoRA
=
QUANTIZED Frozen Base
+
Trainable LoRA Adapters
```

Our experiment used:

```text
4-bit Ministral Base
        +
LoRA Adapters
```

So our overall method was:

```text
QLoRA
```

The trainable pieces themselves are still **LoRA adapter parameters**.

---

# 21. QLoRA Flow Diagram

```text
                       BASE MODEL
                           │
                           ▼
                    4-bit Quantization
                           │
                           ▼
              ┌─────────────────────────┐
              │   4-bit Base Weights    │
              │                         │
              │         FROZEN          │
              └─────────────────────────┘
                           │
                           │
                           ▼
                     Forward Pass
                           │
                           ▼
                     Model Output
                           ▲
                           │
              ┌─────────────────────────┐
              │    LoRA A/B Adapters    │
              │                         │
              │       TRAINABLE         │
              └─────────────────────────┘
                           ▲
                           │
                       Gradients
```

Training memory therefore contains approximately:

```text
┌──────────────────────────────┐
│ 4-bit Frozen Base Model      │
└──────────────────────────────┘
              +
┌──────────────────────────────┐
│ LoRA Adapter Parameters      │
└──────────────────────────────┘
              +
┌──────────────────────────────┐
│ LoRA Gradients               │
└──────────────────────────────┘
              +
┌──────────────────────────────┐
│ LoRA Optimizer State         │
└──────────────────────────────┘
              +
┌──────────────────────────────┐
│ Activations                  │
└──────────────────────────────┘
```

This uses far less VRAM than full fine-tuning.

---

# 22. LoRA vs QLoRA Summary

```text
                     LoRA                 QLoRA
                     ────                 ─────

Base weights         Frozen               Frozen

Base precision       Normal               Quantized

Adapters             LoRA                 LoRA

A/B matrices         Trainable            Trainable

Base gradients       No                   No

Memory use           Lower than           Even lower
                     full fine-tuning

Used here?           No                   YES
```

---

# 23. Batch Size

Our physical batch size was:

```text
2
```

That means two training examples were processed together on the GPU.

However, our effective batch size was larger because we used gradient accumulation.

---

# 24. Gradient Accumulation

Configuration:

```text
Physical batch size = 2

Gradient accumulation steps = 4
```

Therefore:

```text
Effective Batch Size
=
2 × 4
=
8
```

The process:

```text
Examples 1–2
     │
     ▼
Forward Pass
     │
     ▼
Backward Pass
     │
     ▼
Accumulate Gradients
     │
     X
No optimizer step yet


Examples 3–4
     │
     ▼
Forward + Backward
     │
     ▼
Accumulate Gradients
     │
     X
No optimizer step yet


Examples 5–6
     │
     ▼
Forward + Backward
     │
     ▼
Accumulate Gradients
     │
     X
No optimizer step yet


Examples 7–8
     │
     ▼
Forward + Backward
     │
     ▼
Accumulate Gradients
     │
     ▼
OPTIMIZER STEP
     │
     ▼
Update LoRA A/B
```

This lets us approximate a larger batch without needing enough VRAM to process all eight examples simultaneously.

---

# 25. Epochs

An epoch means:

```text
one complete pass through
the entire training dataset
```

For V2:

```text
40 examples
×
5 epochs
=
200 example presentations
```

With an effective batch size of 8:

```text
200 / 8
=
25 optimizer steps
```

And the actual training logs confirmed:

```text
Total steps = 25
```

---

# 26. Learning Rate

Our learning rate was:

```text
2e-4
```

which is:

```text
0.0002
```

The gradient tells the optimizer:

```text
which direction should reduce loss?
```

The learning rate controls:

```text
how far should we move
in that direction?
```

Flow:

```text
Gradient
   │
   ▼
Learning Rate
controls update size
   │
   ▼
Parameter Update
```

If the learning rate is too high:

```text
Parameter Changes
      │
      ▼
Too Large
      │
      ▼
Training Can Become Unstable
```

If it is too low:

```text
Parameter Changes
      │
      ▼
Extremely Small
      │
      ▼
Training Can Become Very Slow
```

We used a linear learning-rate schedule.

---

# 27. Gradient Checkpointing

We enabled:

```python
use_gradient_checkpointing="unsloth"
```

Normally, training stores many intermediate activations from the forward pass.

Without checkpointing:

```text
Forward Pass
     │
     ▼
Store Activation
     │
     ▼
Store Activation
     │
     ▼
Store Activation
     │
     ▼
Store Activation
     │
     ▼
Backward Pass
```

With gradient checkpointing:

```text
Forward Pass
     │
     ▼
Store Selected Checkpoints
     │
     ▼
Discard Some Activations
     │
     ▼
Backward Pass
     │
     ▼
Recompute Missing Activations
when needed
```

The tradeoff is:

```text
LESS VRAM
   │
   └─────────────┐
                 ▼
           MORE COMPUTE
```

---

# 28. Experiment V1 — Our First Attempt

The original V1 dataset contained:

```text
10 examples
```

Topics included:

- clock gating
- power gating
- dynamic power
- leakage power
- DVFS
- IR drop
- cache
- HBM
- quantization
- embeddings

Every desired assistant response used:

```text
CONCEPT:
...

WHY IT MATTERS:
...

EXAMPLE:
...
```

---

# 29. V1 Training Configuration

```text
Model:
Ministral 3 3B Instruct

Method:
QLoRA

Base:
4-bit quantized

Trainable Parameters:
33,751,040

Total Parameters:
3,882,841,088

Trainable Percentage:
~0.87%

LoRA Rank:
16

LoRA Alpha:
16

LoRA Dropout:
0

Target Modules:
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj

Physical Batch:
2

Gradient Accumulation:
4

Effective Batch:
8

Epochs:
5

Learning Rate:
2e-4

Completion-Only Loss:
True
```

---

# 30. V1 Training Results

V1 had:

```text
10 examples
×
5 epochs
```

and produced:

```text
10 optimizer steps
```

The loss sequence was approximately:

```text
Step     Loss

1        2.823
2        2.655
3        2.208
4        1.919
5        1.462
6        1.316
7        1.042
8        1.004
9        0.832
10       0.498
```

Average training loss:

```text
~1.576
```

So the training loss clearly fell:

```text
2.823
  │
  ▼
2.655
  │
  ▼
2.208
  │
  ▼
1.919
  │
  ▼
1.462
  │
  ▼
...
  │
  ▼
0.498
```

At first glance, it looked like the fine-tune had worked very well.

Then we evaluated the actual behavior.

---

# 31. V1 Failed the Actual Goal

Our desired response was:

```text
CONCEPT:
...

WHY IT MATTERS:
...

EXAMPLE:
...
```

Instead, V1 often generated a normal short paragraph.

For example:

```text
Question:
"What is clock gating?"

        │
        ▼

QLoRA V1

"Clock gating is a technique used
in digital circuit design to reduce
power consumption..."
```

The answer was:

```text
shorter       ✅
direct        ✅
technical     ✅
concise       ✅

CONCEPT:      ❌
WHY IT MATTERS: ❌
EXAMPLE:      ❌
```

Even more importantly:

```text
"What is clock gating?"
```

was itself a training example.

Yet the format still failed.

---

# 32. What V1 Actually Learned

The fine-tune still clearly changed the model.

The base model tended toward:

```text
BASE MODEL
│
├── long responses
├── multiple sections
├── extra explanation
├── many bullets
└── more verbose
```

V1 tended toward:

```text
QLoRA V1
│
├── short responses
├── direct definitions
├── fewer sections
└── more concise
```

So V1 seemed to have learned something like:

```text
"When answering technical questions,
be short and direct."
```

rather than:

```text
"Always answer using the exact
CONCEPT / WHY IT MATTERS / EXAMPLE format."
```

---

# 33. Major Lesson from V1

This was one of the biggest lessons of Chapter 13:

```text
Training Loss Decreases
         │
         ▼
Model Fits Training Better
         │
         ▼

BUT

         │
         ▼
Desired Real-World Behavior
Is NOT Automatically Guaranteed
```

In short:

```text
LOW TRAINING LOSS
        ≠
GOOD GENERALIZATION
```

Training loss answers:

```text
"How well am I fitting
the training objective?"
```

Evaluation answers:

```text
"Did I actually learn
the behavior we wanted?"
```

We need both.

---

# 34. Diagnosing the V1 Dataset

We identified two major weaknesses.

## Problem 1: Only 10 Examples

```text
V1
│
└── 10 training examples
```

This provided relatively little evidence about the general rule we wanted the model to learn.

## Problem 2: Limited Question Diversity

Many of the questions had very similar wording:

```text
What is clock gating?

What is HBM?

What is quantization?

What is an embedding?
```

The pattern was often approximately:

```text
"What is X?"
```

The model could therefore latch onto a weaker correlation:

```text
"What is X?"
      │
      ▼
give a concise technical definition
```

instead of our real goal:

```text
ANY technical question
regardless of wording
      │
      ▼
CONCEPT:
...

WHY IT MATTERS:
...

EXAMPLE:
...
```

---

# 35. Pivoting to V2

Rather than simply hammering the original 10 examples for more epochs, we changed the **dataset**.

This was deliberate.

We wanted:

```text
More Examples
      +
More Prompt Diversity
      +
Same Exact Desired Output Structure
```

V2 contained:

```text
40 examples
```

with prompt styles such as:

```text
What is clock gating?

Explain power gating.

How would you describe dynamic power?

What does leakage power mean?

Can you explain DVFS?

What exactly is IR drop?

Why do AI GPUs use HBM?

How does a vector database help RAG?

Explain reranking in a RAG system.

Can you explain self-attention?

Why use mixture-of-experts?

How would you explain learning rate?
```

The **inputs varied**.

The **topics varied**.

But the assistant output always followed:

```text
CONCEPT:
...

WHY IT MATTERS:
...

EXAMPLE:
...
```

---

# 36. The V2 Invariant

The idea behind V2 was:

```text
"What is X?"
      │
      │
"Explain Y."
      │
      │
"Why use Z?"
      │
      │
"Can you explain A?"
      │
      │
"How does B work?"
      │
      ▼

┌────────────────────────────┐
│      SAME OUTPUT FORMAT    │
├────────────────────────────┤
│                            │
│ CONCEPT:                   │
│ ...                        │
│                            │
│ WHY IT MATTERS:            │
│ ...                        │
│                            │
│ EXAMPLE:                   │
│ ...                        │
│                            │
└────────────────────────────┘
```

So from the model's perspective:

```text
Question Wording
CHANGES
      │
      ▼

Technical Topic
CHANGES
      │
      ▼

Response Structure
DOES NOT CHANGE
```

That repeated invariant gives the model a much clearer signal about what behavior it should learn.

---

# 37. V2 Training Configuration

We intentionally kept most of the training configuration the same as V1.

```text
Model:
Ministral 3 3B Instruct

Method:
QLoRA

Base Precision:
4-bit

LoRA Rank:
16

LoRA Alpha:
16

LoRA Dropout:
0

Physical Batch:
2

Gradient Accumulation:
4

Effective Batch:
8

Learning Rate:
2e-4

Epochs:
5

Completion-Only Loss:
True
```

The biggest change was:

```text
THE DATASET
```

This gave us a cleaner experiment because we could observe how improved data affected the result while keeping the rest of the training approach similar.

---

# 38. V2 Training Results

V2 contained:

```text
40 examples
```

across:

```text
5 epochs
```

Therefore:

```text
40 × 5
=
200 example presentations
```

With effective batch size 8:

```text
200 / 8
=
25 optimizer steps
```

Training runtime:

```text
~43.7 seconds
```

The loss sequence generally decreased:

```text
2.768
2.872
2.660
2.124
1.932
1.600
1.441
1.231
1.335
1.254
1.083
0.885
0.927
0.816
0.736
0.590
0.549
0.579
0.570
0.459
0.426
0.302
0.343
0.355
0.295
```

Average reported training loss:

```text
~1.125
```

But after V1, we knew that this alone was not enough.

The real question was:

```text
Did V2 generalize?
```

---

# 39. Testing Generalization

We deliberately tested V2 using technical concepts that were **not included in its training data**.

The evaluation questions included:

```text
What is electromigration?

Explain a PLL.

What is thermal throttling?
```

Why use unseen concepts?

Because:

```text
Test Training Prompt
        │
        ▼
Could Be Memorization
```

while:

```text
Test Unseen Concept
        │
        ▼
Better Test of Generalized Behavior
```

---

# 40. Three-Way Evaluation

The final comparison tested:

```text
BASE MODEL
     │
     ▼
QLoRA V1
     │
     ▼
QLoRA V2
```

## Base Model

Typical behavior:

```text
Long response
     │
     ├── detailed definition
     ├── subsections
     ├── lists
     └── extra explanation
```

## QLoRA V1

Typical behavior:

```text
Shorter response
     │
     ├── concise
     ├── direct
     └── no exact required headings
```

## QLoRA V2

Typical behavior:

```text
CONCEPT:
...

WHY IT MATTERS:
...

EXAMPLE:
...
```

That was the behavior we actually wanted.

---

# 41. Example: Unseen Electromigration Question

Question:

```text
What is electromigration?
```

The V2 model generated:

```text
CONCEPT:
Electromigration is the movement of metal atoms caused by
atomic collisions with drifting electrons.

WHY IT MATTERS:
It can cause metal interconnects to migrate and eventually
form voids or grow into whiskers.

EXAMPLE:
...
```

Automatic format check:

```text
✅ CONCEPT:
✅ WHY IT MATTERS:
✅ EXAMPLE:
```

The important fact was:

```text
Electromigration
was NOT in the V2 training dataset.
```

So V2 was applying the learned response pattern to a new concept.

---

# 42. Final V2 Evaluation Result

We evaluated:

```text
3 unseen questions
```

and each required:

```text
3 exact headings
```

Therefore:

```text
3 questions
×
3 headings
=
9 checks
```

Final result:

```text
✅ CONCEPT:
✅ WHY IT MATTERS:
✅ EXAMPLE:

✅ CONCEPT:
✅ WHY IT MATTERS:
✅ EXAMPLE:

✅ CONCEPT:
✅ WHY IT MATTERS:
✅ EXAMPLE:
```

Overall:

```text
9 / 9 FORMAT CHECKS PASSED
```

This was dramatically better than V1.

---

# 43. V1 vs V2

```text
                         V1                  V2
                         ──                  ──

Training examples          10                  40

Prompt diversity           Low                 Higher

Epochs                     5                   5

Physical batch             2                   2

Gradient accumulation      4                   4

Effective batch            8                   8

Learning rate              2e-4                2e-4

LoRA rank                  16                  16

LoRA alpha                 16                  16

QLoRA method               Same                Same

Optimizer steps            10                  25

Loss decreased             Yes                 Yes

Model behavior changed     Yes                 Yes

Exact format learned       No                  Yes

Unseen format evaluation   Failed goal         9 / 9
```

The key experimental insight:

```text
Same Base Model
      +
Same QLoRA Technique
      +
Same LoRA Rank
      +
Same Learning Rate
      +
Same Number of Epochs
      │
      │
      BUT
      ▼
Better Dataset
      │
      ▼
Much Better Behavior
```

---

# 44. Dataset Design Matters

Fine-tuning data is not simply something the model consumes.

The dataset is effectively our behavioral specification.

```text
Training Dataset
      │
      ▼
Repeated Statistical Patterns
      │
      ▼
Model Learns Those Patterns
      │
      ▼
Behavior Changes
```

If the dataset contains the wrong or ambiguous pattern, the model may learn something other than what we intended.

V1 unintentionally provided a strong signal for:

```text
"Give concise technical answers."
```

V2 provided a much stronger signal for:

```text
"No matter how the technical question
is phrased, use these exact sections."
```

---

# 45. Memorization vs Generalization

Memorization might look like:

```text
TRAINING

Question:
What is clock gating?

      │
      ▼

Desired formatted answer

      │
      ▼

MODEL


TEST

Question:
What is clock gating?

      │
      ▼

Perfect formatted answer
```

That alone does not prove much.

The model may simply have memorized the pattern associated with that input.

Generalization looks more like:

```text
TRAINING TOPICS

Clock gating
Power gating
HBM
RAG
LoRA
DRAM
Quantization
Transformers
Embeddings
etc.
       │
       │
       ▼
Every answer follows:

CONCEPT:
WHY IT MATTERS:
EXAMPLE:
       │
       ▼
Model Learns General Pattern
       │
       ▼

UNSEEN TOPICS

Electromigration
PLL
Thermal throttling
       │
       ▼

CONCEPT:
WHY IT MATTERS:
EXAMPLE:
```

That is much closer to what we observed in V2.

---

# 46. Fine-Tuning Behavior vs Knowledge

V2 successfully learned our response format.

That does **not** mean every factual statement generated by the model is automatically perfect.

This gives another important distinction:

```text
Fine-Tuning
can improve:

HOW THE MODEL ANSWERS
        │
        ▼
Format
Style
Structure
Behavior


But this does not automatically improve:

WHAT THE MODEL KNOWS
        │
        ▼
Factual Knowledge
Current Information
Underlying Reasoning Ability
```

Therefore:

```text
FORMAT QUALITY
      ≠
FACTUAL QUALITY
```

Both should be evaluated separately.

---

# 47. Does LoRA Make the Model Larger?

While an adapter remains separate, we effectively have:

```text
Base Model
    +
LoRA Adapter
```

So total storage is slightly larger than the base model alone.

However, LoRA updates can also be merged.

Remember:

```text
W' = W + BA
```

After merging:

```text
Original W
4096 × 4096

      │
      ▼

Merged W'
4096 × 4096
```

The matrix dimensions themselves do not increase.

The architecture remains the same size.

File size can still differ depending on precision:

```text
FP32
FP16
BF16
INT8
INT4
GGUF quantization
```

but precision and quantization are separate concepts from LoRA itself.

---

# 48. Adapter Storage

Our two adapters were:

```text
lora_adapter
lora_adapter_v2
```

We also preserved compressed copies:

```text
lora_adapter_v1.tar.gz
lora_adapter_v2.tar.gz
```

The adapter artifacts are intentionally **not committed to GitHub**.

The repository contains the reproducible parts of the experiment:

```text
Python Code
     +
Datasets
     +
Evaluation Scripts
     +
README
```

while the trained binary artifacts are stored separately.

The root `.gitignore` contains:

```gitignore
# Chapter 13 - fine-tuning artifacts
13-fine-tuning/lora_adapter/
13-fine-tuning/lora_adapter_v2/
13-fine-tuning/*.tar.gz
```

---

# 49. Complete QLoRA Training Pipeline

```text
                  JSONL DATASET
                        │
                        ▼
             User + Assistant Messages
                        │
                        ▼
                  Chat Template
                        │
                        ▼
                    Tokenizer
                        │
                        ▼
                    Token IDs
                        │
                        ▼
                 Embedding Vectors
                        │
                        ▼
       ┌────────────────────────────────┐
       │   4-bit Ministral Base Model   │
       │                                │
       │            FROZEN              │
       └────────────────────────────────┘
                        │
                        │
                        ▼
       ┌────────────────────────────────┐
       │       Transformer Layers       │
       │                                │
       │ q/k/v/o projections            │
       │ gate/up/down projections       │
       │                                │
       │ Original matrices = FROZEN     │
       └────────────────────────────────┘
                        │
                        │
                 LoRA A/B added
                        │
                        ▼
       ┌────────────────────────────────┐
       │        LoRA Adapters           │
       │                                │
       │          TRAINABLE             │
       └────────────────────────────────┘
                        │
                        ▼
                 Model Prediction
                        │
                        ▼
              Desired Assistant Tokens
                        │
                        ▼
                       Loss
                        │
                        ▼
                Backpropagation
                        │
                        ▼
                    Gradients
                        │
                        ▼
                    Optimizer
                        │
                        ▼
              Update LoRA A/B Only
                        │
                        ▼
                      Repeat
```

---

# 50. Full Mental Model

The complete mental model I am taking away from the chapter is:

```text
                      DATA
                       │
                       ▼
              Desired Behavior
                       │
                       ▼
              Chat Formatting
                       │
                       ▼
                 Tokenization
                       │
                       ▼
               Model Forward Pass
                       │
                       ▼
             Predicted Next Tokens
                       │
                       ▼
                Compare to Labels
                       │
                       ▼
                     Loss
                       │
                       ▼
                Backpropagation
                       │
                       ▼
                   Gradients
                       │
                       ▼
                  Optimizer
                       │
                       ▼
              Update LoRA A/B
                       │
                       ▼
                Repeat Training
                       │
                       ▼
               Learned Behavior
```

Meanwhile the base model remains:

```text
4-bit Base Weights
        │
        ▼
      FROZEN
        │
        X
No optimizer updates
```

---

# 51. Files in This Chapter

## `train.jsonl`

Original V1 training dataset.

Contains 10 supervised fine-tuning examples.

This dataset was used for the first experiment.

---

## `inspect_dataset.py`

Loads and displays examples from the original JSONL dataset.

It was useful for verifying the structure before attempting training.

---

## `tokenize_demo.py`

Used to inspect:

```text
Messages
   │
   ▼
Chat Template
   │
   ▼
Formatted Prompt
   │
   ▼
Token IDs
```

This helped make the normally invisible tokenization process concrete.

---

## `train.py`

The V1 QLoRA training script.

It:

- loads the 4-bit Ministral checkpoint
- extracts the text tokenizer from Ministral's multimodal processor
- attaches LoRA adapters
- loads the V1 dataset
- trains using completion-only loss
- saves the V1 adapter

---

## `compare.py`

Compares:

```text
Base Ministral
      vs
QLoRA V1
```

This script helped reveal that V1 changed the model's response style without successfully learning our exact output structure.

---

## `make_dataset_v2.py`

Generates the improved 40-example V2 dataset.

The critical design idea was:

```text
VARY THE INPUT

while

KEEPING THE DESIRED OUTPUT STRUCTURE CONSTANT
```

---

## `train_v2.jsonl`

The V2 training dataset.

Contains 40 examples covering semiconductor, hardware, AI, RAG, transformer, and training concepts.

---

## `train_v2.py`

The second QLoRA training script.

It uses the same overall approach as V1 while training on the improved dataset.

The resulting adapter is saved as:

```text
lora_adapter_v2
```

---

## `compare_v2.py`

Runs the final three-way evaluation:

```text
Base
  vs
V1
  vs
V2
```

It also automatically searches V2 responses for:

```text
CONCEPT:
WHY IT MATTERS:
EXAMPLE:
```

Final result:

```text
9 / 9 FORMAT CHECKS PASSED
```

---

# 52. Important Debugging Lessons

The training process also exposed several practical issues that were useful to understand.

## Unsloth Import Order

Unsloth should be imported before libraries such as TRL, Transformers, and PEFT so it can apply its optimizations correctly.

---

## Multiprocessing Pickling Error

Initial dataset preprocessing attempted multiple worker processes and produced a pickling error involving objects that could not be serialized.

The fix was:

```python
dataset_num_proc=1
```

For our tiny dataset, there was no meaningful need for parallel preprocessing anyway.

---

## Ministral Is Multimodal

Ministral 3 uses a multimodal architecture.

`FastModel.from_pretrained()` therefore returned a processor rather than just a basic tokenizer.

We extracted the text tokenizer using the equivalent of:

```python
tokenizer = processor.tokenizer
```

and passed that text tokenizer to the SFT trainer.

This allowed our text-only prompt/completion dataset to train correctly.

---

# 53. RunPod GPU Experiment

The training was performed using a cloud GPU because the local M2 MacBook Air was not the right environment for this particular CUDA / Unsloth QLoRA workflow.

RunPod configuration:

```text
GPU:
NVIDIA RTX 4090

VRAM:
24 GB

Training:
QLoRA

Model:
Ministral 3 3B Instruct

Base:
4-bit
```

The actual V2 training itself took only:

```text
~43.7 seconds
```

This demonstrates that parameter-efficient fine-tuning of a small 3B model can be surprisingly fast once the environment is configured correctly.

---

# 54. Key Takeaways

1. **Fine-tuning and RAG solve different problems.**

   RAG changes runtime context.

   Fine-tuning changes model behavior.

2. **Supervised fine-tuning trains on desired input/output examples.**

3. **Causal language models still fundamentally predict the next token.**

4. **Teacher forcing provides correct previous tokens during training.**

5. **Chat templates and tokenization determine what the transformer actually receives.**

6. **Labels determine which token predictions contribute to training loss.**

7. **Completion-only loss lets the prompt act as context while training on assistant output.**

8. **LoRA freezes original model matrices and learns small low-rank A/B adapters.**

9. **QLoRA combines LoRA with a quantized frozen base model.**

10. **Our base model used 4-bit weights.**

11. **Only 33,751,040 out of 3,882,841,088 parameters were trainable.**

12. **That is only about 0.87% of the model.**

13. **Gradient accumulation allowed a physical batch size of 2 to behave like an effective batch size of 8.**

14. **Gradient checkpointing trades additional computation for lower VRAM use.**

15. **Training loss going down does not guarantee the desired behavior was learned.**

16. **V1 changed the model but mostly taught concision rather than the exact output schema.**

17. **V1 used only 10 examples with limited prompt diversity.**

18. **V2 increased the dataset to 40 examples and varied the user-question wording.**

19. **The desired output format remained invariant across every V2 training example.**

20. **V2 generalized the format to unseen technical concepts.**

21. **The final V2 evaluation passed 9 out of 9 format checks.**

22. **Dataset design can matter as much as the fine-tuning algorithm itself.**

23. **Model behavior and factual accuracy are separate things and should be evaluated separately.**

---

# 55. The Most Important Experiment Result

The most useful result was not simply that V2 worked.

The useful result was the entire progression:

```text
Build V1 Dataset
      │
      ▼
Train V1
      │
      ▼
Loss Falls
      │
      ▼
Evaluate
      │
      ▼
Desired Format Fails
      │
      ▼
Do NOT Assume Training Worked
      │
      ▼
Analyze Failure
      │
      ├── Only 10 examples
      │
      └── Limited prompt diversity
      │
      ▼
Redesign Dataset
      │
      ▼
40 Examples
      │
      +
Varied Prompt Wording
      │
      +
Same Output Schema
      │
      ▼
Train V2
      │
      ▼
Evaluate on Unseen Concepts
      │
      ▼
9 / 9 Format Checks Pass
```

This is a much more realistic fine-tuning workflow than:

```text
Train Once
   │
   ▼
Loss Went Down
   │
   ▼
Assume Success
```

The real process is:

```text
Define Desired Behavior
        │
        ▼
Design Training Data
        │
        ▼
Train
        │
        ▼
Evaluate
        │
        ▼
Identify Failure Modes
        │
        ▼
Improve Dataset / Training
        │
        ▼
Train Again
        │
        ▼
Evaluate Again
```

---

# 56. Chapter Result

By the end of this chapter, I successfully used:

```text
Ministral 3 3B Instruct
        │
        +
4-bit Quantized Base Model
        │
        +
QLoRA
        │
        +
33.75M Trainable LoRA Parameters
        │
        +
40-Example Behavioral Dataset
        │
        ▼
Fine-Tuned Model
```

The final adapter changed the behavior of a model with nearly **3.9 billion parameters** while directly training less than **1%** of them.

Most importantly, the model applied the learned output structure to questions about concepts it had never seen in the fine-tuning dataset.

The final experiment therefore demonstrated:

```text
Good Model
   +
Parameter-Efficient Fine-Tuning
   +
Well-Designed Dataset
   +
Proper Evaluation
   │
   ▼
Successful Behavioral Adaptation
```

The biggest lesson from Chapter 13 is:

> **Fine-tuning is not just about making loss go down. It is about designing a dataset that clearly expresses the behavior you want, training the model, and then testing whether that behavior actually generalizes.**

---

# Next Chapter

The adapter currently exists separately from the original model.

The next step is to learn how model weights and adapters can be prepared for practical local inference, including concepts such as:

```text
LoRA Adapter
      │
      ▼
Merge / Export
      │
      ▼
Model Precision
      │
      ▼
Quantization
      │
      ▼
GGUF
      │
      ▼
llama.cpp
      │
      ▼
Ollama
      │
      ▼
Efficient Local Inference
```

That leads into **Chapter 14 — Quantization, GGUF, llama.cpp, and Ollama**.


