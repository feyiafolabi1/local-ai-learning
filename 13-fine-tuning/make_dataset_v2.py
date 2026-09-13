import json
from pathlib import Path


OUTPUT_PATH = Path(
    "/workspace/13-fine-tuning/train_v2.jsonl"
)


EXAMPLES = [
    {
        "question": "What is clock gating?",
        "concept": "Clock gating disables the clock signal to inactive logic blocks.",
        "why": "It reduces unnecessary switching activity and therefore lowers dynamic power.",
        "example": "An idle GPU compute unit can have its clock gated until new work arrives.",
    },
    {
        "question": "Explain power gating.",
        "concept": "Power gating disconnects an inactive circuit block from its power supply using sleep transistors.",
        "why": "It reduces leakage power when a block is not being used for an extended period.",
        "example": "A processor can power gate an unused accelerator block during a long idle interval.",
    },
    {
        "question": "How would you describe dynamic power?",
        "concept": "Dynamic power is power consumed when circuit nodes switch between logic states.",
        "why": "It increases with switching activity, capacitance, frequency, and approximately the square of supply voltage.",
        "example": "Raising a GPU clock frequency while activity remains similar generally increases dynamic power.",
    },
    {
        "question": "What does leakage power mean?",
        "concept": "Leakage power is power consumed by transistor leakage currents even when useful switching is not occurring.",
        "why": "It contributes to static power and becomes increasingly important in advanced semiconductor processes.",
        "example": "A powered but idle CPU core still consumes some energy because its transistors leak current.",
    },
    {
        "question": "Can you explain DVFS?",
        "concept": "Dynamic Voltage and Frequency Scaling changes operating voltage and frequency according to performance demand.",
        "why": "It allows a processor to trade performance against power and energy consumption.",
        "example": "A CPU can reduce voltage and clock frequency during a light workload to save energy.",
    },
    {
        "question": "What exactly is IR drop?",
        "concept": "IR drop is the reduction in supply voltage caused by current flowing through resistive power-delivery paths.",
        "why": "Excessive voltage drop can reduce timing margin and cause unreliable circuit operation.",
        "example": "A highly active GPU region can draw enough current to lower the local voltage seen by nearby logic.",
    },
    {
        "question": "Explain voltage droop.",
        "concept": "Voltage droop is a temporary reduction in supply voltage caused by rapid changes in current demand.",
        "why": "Large droops can reduce timing margin and potentially cause computation errors.",
        "example": "A sudden increase in GPU activity can momentarily pull the local supply voltage below its nominal value.",
    },
    {
        "question": "What is a voltage regulator?",
        "concept": "A voltage regulator maintains an output voltage near a desired target despite changes in input voltage or load.",
        "why": "Stable supply voltage is necessary for reliable operation of electronic circuits.",
        "example": "A regulator can maintain a processor rail at its target voltage while workload current changes.",
    },
    {
        "question": "How does a cache work at a high level?",
        "concept": "A cache is a small, fast memory that stores frequently or recently accessed data near the processor.",
        "why": "It reduces average access latency and lowers traffic to slower memory levels.",
        "example": "A CPU may retrieve reused data from L1 cache instead of fetching it again from DRAM.",
    },
    {
        "question": "What is HBM?",
        "concept": "High Bandwidth Memory is stacked DRAM connected to a processor through a very wide memory interface.",
        "why": "It provides high memory bandwidth for workloads that need large amounts of data moved quickly.",
        "example": "AI accelerators use HBM to feed model weights and activations to many parallel compute units.",
    },
    {
        "question": "Why do AI GPUs use HBM?",
        "concept": "HBM provides high-bandwidth access to large amounts of memory located close to the compute device.",
        "why": "AI workloads often require enormous quantities of weights and activations to be moved between memory and compute.",
        "example": "A large language model can require hundreds of gigabytes per second or more of memory traffic during inference.",
    },
    {
        "question": "What does memory bandwidth mean?",
        "concept": "Memory bandwidth is the rate at which data can be transferred between memory and a processor.",
        "why": "A workload can become memory-bound when compute units consume data faster than memory can supply it.",
        "example": "A GPU with higher HBM bandwidth can feed matrix-processing units more quickly during some AI workloads.",
    },
    {
        "question": "Explain memory capacity.",
        "concept": "Memory capacity is the total amount of data that a memory system can store at one time.",
        "why": "Capacity determines whether a workload's data can fit without being moved to another storage tier.",
        "example": "A 192 GB HBM system can hold larger models directly in accelerator memory than a 48 GB system.",
    },
    {
        "question": "What is SRAM?",
        "concept": "Static Random Access Memory stores bits using transistor-based storage cells and does not require periodic refresh.",
        "why": "SRAM is very fast but consumes much more chip area per bit than DRAM.",
        "example": "Processor caches commonly use SRAM because low access latency is more important than high capacity.",
    },
    {
        "question": "Can you explain DRAM?",
        "concept": "Dynamic Random Access Memory stores data using charge that must be periodically refreshed.",
        "why": "DRAM provides much higher storage density than SRAM at lower cost per bit.",
        "example": "System memory and HBM both use forms of DRAM technology.",
    },
    {
        "question": "What is a FinFET?",
        "concept": "A FinFET is a transistor structure where the gate wraps around multiple sides of a raised semiconductor fin.",
        "why": "The structure improves electrostatic control of the channel compared with older planar transistors.",
        "example": "Modern CPUs and GPUs have used FinFET processes to improve performance and reduce leakage.",
    },
    {
        "question": "Explain gate-all-around transistors.",
        "concept": "A gate-all-around transistor surrounds the conducting channel with gate material on all sides.",
        "why": "Greater gate control can improve switching behavior and reduce leakage at very small process geometries.",
        "example": "Nanosheet-based gate-all-around transistors are being introduced in advanced semiconductor nodes.",
    },
    {
        "question": "What is a chiplet?",
        "concept": "A chiplet is a smaller integrated circuit designed to be combined with other dies in one package.",
        "why": "Chiplets allow designers to mix functions and manufacturing processes while improving scalability and yield.",
        "example": "A processor package can combine separate compute dies and an I/O die using high-speed die-to-die links.",
    },
    {
        "question": "Why would a company use chiplets?",
        "concept": "Chiplets divide a large processor into multiple smaller dies that operate together as one system.",
        "why": "Smaller dies can improve manufacturing yield and allow different functions to use different process technologies.",
        "example": "CPU cores may be manufactured on an advanced node while I/O circuitry uses a cheaper mature node.",
    },
    {
        "question": "What does quantization mean in AI?",
        "concept": "Quantization represents model values using lower-precision numerical formats.",
        "why": "Lower precision can reduce memory usage, memory bandwidth requirements, and computational cost.",
        "example": "A model using 4-bit weights requires much less memory than the same model stored with 16-bit weights.",
    },
    {
        "question": "Explain an embedding.",
        "concept": "An embedding is a numerical vector representation of data such as text, images, or other objects.",
        "why": "Semantically similar items can be located near one another in vector space.",
        "example": "A RAG system can compare a question embedding with PDF chunk embeddings to retrieve relevant passages.",
    },
    {
        "question": "What is RAG?",
        "concept": "Retrieval-Augmented Generation combines information retrieval with language-model generation.",
        "why": "It allows a model to answer using external information without permanently changing its model weights.",
        "example": "A system can retrieve relevant sections from technical PDFs and provide them as context to an LLM.",
    },
    {
        "question": "How does a vector database help RAG?",
        "concept": "A vector database stores embeddings and supports similarity searches between vectors.",
        "why": "It enables efficient retrieval of semantically related documents or chunks from large collections.",
        "example": "Chroma can return PDF chunks whose embeddings are closest to an embedded user question.",
    },
    {
        "question": "What is cosine similarity used for?",
        "concept": "Cosine similarity measures how closely two vectors point in the same direction.",
        "why": "It provides a useful way to compare semantic embeddings regardless of their absolute magnitude.",
        "example": "A RAG system can rank document chunks by cosine similarity to a user's query embedding.",
    },
    {
        "question": "Explain reranking in a RAG system.",
        "concept": "Reranking uses a more precise model to reorder an initial set of retrieved candidates.",
        "why": "Fast vector retrieval may find broadly relevant results while a reranker can improve the final ordering.",
        "example": "A cross-encoder can rescore the top ten Chroma results and move the best passage to rank one.",
    },
    {
        "question": "What is a transformer model?",
        "concept": "A transformer is a neural-network architecture that processes relationships among tokens using attention mechanisms.",
        "why": "Transformers can model long-range relationships efficiently and form the basis of most modern large language models.",
        "example": "Models such as Mistral use stacked transformer layers to process and generate text.",
    },
    {
        "question": "Can you explain self-attention?",
        "concept": "Self-attention allows each token representation to incorporate information from other tokens in the same sequence.",
        "why": "It helps the model determine which parts of the input are most relevant when building contextual representations.",
        "example": "The word bank can attend to surrounding words to distinguish a financial bank from a river bank.",
    },
    {
        "question": "What is multi-head attention?",
        "concept": "Multi-head attention runs several attention operations in parallel using different learned projections.",
        "why": "Different heads can learn to focus on different relationships or features within the same sequence.",
        "example": "One attention head may emphasize nearby syntax while another captures longer-range semantic relationships.",
    },
    {
        "question": "What is an MoE model?",
        "concept": "A Mixture-of-Experts model contains multiple expert subnetworks and selectively activates a subset for each token.",
        "why": "It can increase total model capacity without using every parameter for every token.",
        "example": "A model may route each token to two experts out of a much larger pool of expert networks.",
    },
    {
        "question": "Why use mixture-of-experts?",
        "concept": "Mixture-of-Experts architectures selectively activate only part of a model's available expert parameters.",
        "why": "This can provide high model capacity while keeping computation per token lower than activating every parameter.",
        "example": "A sparse MoE model can contain hundreds of billions of parameters while using only a fraction for each token.",
    },
    {
        "question": "What is model inference?",
        "concept": "Inference is the process of using a trained model to generate predictions or outputs from new inputs.",
        "why": "Inference efficiency determines latency, throughput, memory usage, and serving cost after training is complete.",
        "example": "Generating the next token from a user prompt with Ministral is an inference operation.",
    },
    {
        "question": "Explain model training.",
        "concept": "Model training adjusts trainable parameters to reduce prediction error on training examples.",
        "why": "The process allows a neural network to learn statistical patterns represented in the dataset.",
        "example": "During QLoRA training, gradients update LoRA adapter matrices while the quantized base weights remain frozen.",
    },
    {
        "question": "What is backpropagation?",
        "concept": "Backpropagation computes how changes in trainable parameters would affect the model's loss.",
        "why": "Those gradients tell the optimizer how to adjust parameters to make desired predictions more likely.",
        "example": "A QLoRA training step can backpropagate loss through the model and update only the LoRA matrices.",
    },
    {
        "question": "What does training loss represent?",
        "concept": "Training loss measures how poorly a model's predictions match the desired outputs in the training data.",
        "why": "A decreasing loss generally indicates that the model is fitting the training examples more successfully.",
        "example": "A language-model fine-tune may begin with a loss above two and fall as desired response tokens become more likely.",
    },
    {
        "question": "Can you explain LoRA?",
        "concept": "Low-Rank Adaptation fine-tunes a model by learning small low-rank matrices while keeping the original model weights frozen.",
        "why": "It dramatically reduces the number of trainable parameters and the memory required for fine-tuning.",
        "example": "A multi-billion-parameter model can be adapted by training tens of millions of LoRA parameters instead of every weight.",
    },
    {
        "question": "What is QLoRA?",
        "concept": "QLoRA combines a quantized frozen base model with trainable LoRA adapters.",
        "why": "Quantizing the base model reduces memory usage while LoRA limits the number of parameters that require gradients and optimizer state.",
        "example": "A 3B model can be loaded in 4-bit form while approximately one percent of its parameters are trained through LoRA adapters.",
    },
    {
        "question": "Explain gradient accumulation.",
        "concept": "Gradient accumulation combines gradients from several mini-batches before performing an optimizer update.",
        "why": "It allows training to achieve a larger effective batch size without holding the entire batch in GPU memory at once.",
        "example": "A physical batch size of two with four accumulation steps produces an effective batch size of eight.",
    },
    {
        "question": "What is gradient checkpointing?",
        "concept": "Gradient checkpointing saves memory by storing fewer intermediate activations during the forward pass and recomputing some during backpropagation.",
        "why": "It reduces GPU memory usage at the cost of additional computation.",
        "example": "QLoRA training can use gradient checkpointing to fit longer sequences or larger models into limited VRAM.",
    },
    {
        "question": "What is an epoch in training?",
        "concept": "An epoch is one complete pass through the entire training dataset.",
        "why": "Multiple epochs allow the model to see each training example repeatedly and continue refining its parameters.",
        "example": "Training for five epochs on forty examples exposes the model to two hundred example presentations.",
    },
    {
        "question": "How would you explain learning rate?",
        "concept": "The learning rate controls the size of each optimizer update to trainable model parameters.",
        "why": "A rate that is too high can make training unstable while one that is too low can make learning unnecessarily slow.",
        "example": "A LoRA fine-tune might begin with a learning rate such as 2e-4 and reduce it during training.",
    },
]


def build_answer(example):
    return (
        f"CONCEPT:\n"
        f"{example['concept']}\n\n"
        f"WHY IT MATTERS:\n"
        f"{example['why']}\n\n"
        f"EXAMPLE:\n"
        f"{example['example']}"
    )


def main():
    with open(OUTPUT_PATH, "w") as file:
        for example in EXAMPLES:
            row = {
                "messages": [
                    {
                        "role": "user",
                        "content": example["question"],
                    },
                    {
                        "role": "assistant",
                        "content": build_answer(example),
                    },
                ]
            }

            file.write(
                json.dumps(row) + "\n"
            )

    print(
        f"Wrote {len(EXAMPLES)} examples to:"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
