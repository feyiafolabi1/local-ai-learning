# Chapter 18 — AI Engineering Copilot

A local-first AI engineering assistant combining intelligent routing, retrieval-augmented generation (RAG), deterministic tool execution, multimodal inference, conversation memory, and detailed performance observability.

This project is the final application in my 18-chapter Local AI Learning journey, bringing together concepts explored throughout the repository.

The primary goal is to build and evaluate an end-to-end AI system, with particular emphasis on routing accuracy, inference performance, retrieval quality, hardware efficiency, and measurable system behavior.

## 1. Project Overview

The AI Engineering Copilot supports five execution routes:

| Route | Purpose | Execution |
|---|---|---|
| CHAT | General AI and hardware questions | Local Ministral 3B |
| RAG | Questions about my repository and experiments | ChromaDB + reranker + Ministral |
| BENCHMARK_ANALYSIS | Analyze inference performance and bottlenecks | Local Ministral 3B |
| TOOL | Deterministic engineering calculations | Python |
| MULTIMODAL | Understand images, charts, and screenshots | Local Ministral 3B vision |

Requests are classified using Jev, with an optional hybrid routing mode that falls back to a local Ministral classifier.

The application also supports bounded conversation memory, contextual query rewriting, streaming responses, and per-request performance instrumentation.

### Key Features

- Five specialized execution routes
- Jev and hybrid routing modes
- Local LLM inference using Ollama
- Repository-aware RAG with cross-encoder reranking
- Deterministic engineering calculators
- Local image understanding
- Session-based conversation memory
- Context-aware follow-up routing
- Streaming token generation
- Detailed latency and throughput measurements
- Cold-start and warm-inference analysis

---

## 2. System Architecture

```text
                       USER REQUEST
                            |
                            v
                    CONVERSATION MEMORY
                            |
                            v
                   CONTEXTUAL RESOLUTION
                            |
                            v
                       JEV ROUTER
                            |
                  Optional Hybrid Fallback
                            |
                            v
                      ROUTE DECISION
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
         CHAT              RAG              MULTIMODAL
          |                 |                  |
          |           Query Embedding          |
          |                 |                  |
          |           ChromaDB Search          |
          |                 |                  |
          |           Cross-Encoder            |
          |              Reranking             |
          |                 |                  |
          |           Source Context           |
          |                 |                  |
          v                 v                  v
       Ministral         Ministral          Ministral
          |                 |                Vision
          |                 |                  |
          +-----------------+------------------+
                            |
                 +----------+----------+
                 |                     |
                 v                     v
                TOOL            BENCHMARK_ANALYSIS
                 |                     |
                 v                     v
          Python Calculators        Ministral
                 |                     |
                 +----------+----------+
                            |
                            v
                         RESPONSE
                            |
                            v
                    PERFORMANCE METRICS
                            |
                            v
                    CONVERSATION MEMORY
```

The system uses a routing-first architecture to select the appropriate execution path.

Importantly, not every request requires LLM generation. Deterministic calculations can execute directly in Python, avoiding unnecessary model inference.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Local inference | Ollama |
| Main language model | Ministral 3B |
| Routing model | Jev via TypeSafe SDK |
| Hybrid routing fallback | Ministral 3B |
| Embedding model | EmbeddingGemma |
| Vector database | ChromaDB |
| Reranker | MS MARCO MiniLM cross-encoder |
| Vision inference | Ministral 3B |
| Conversation memory | Python deque |
| Performance instrumentation | Python time.perf_counter + Ollama metrics |
| Development hardware | Apple M2 MacBook Air, 8 GB unified memory |

The application is local-first rather than fully offline.

Inference, embeddings, retrieval, and image processing run locally through Ollama and Python.

Jev routing uses an external API and sends text routing requests to TypeSafe. Image bytes are not sent to Jev by the application.

---

## 4. Intelligent Routing

The router classifies user requests into five categories:

```python
ROUTES = [
    "CHAT",
    "RAG",
    "BENCHMARK_ANALYSIS",
    "TOOL",
    "MULTIMODAL",
]
```

Two routing modes are available.

### Jev Routing

Jev is the default router.

Advantages:

- Low classification latency
- No local GPU inference required for classification
- Reduced competition for limited unified memory
- High accuracy on the evaluated question sets

Example:

```bash
python3 copilot.py --router jev
```

### Hybrid Routing

Hybrid routing begins with Jev and uses local Ministral as a fallback when the confidence score falls below a configured threshold or the external routing request fails.

```text
User Question
      |
      v
     Jev
      |
      v
 Confidence Check
      |
      +---- Sufficient Confidence ----> Accept Jev
      |
      +---- Low Confidence ------------> Ministral Fallback
      |
      v
 Final Route
```

Example:

```bash
python3 copilot.py --router hybrid --threshold 0.60
```

The application currently defaults to a 0.60 confidence threshold.

### Router Evaluation

We evaluated three routing approaches using a separate set of 40 questions covering all five routes.

| Router | Accuracy | Median Latency | P95 Latency |
|---|---:|---:|---:|
| Jev | 40/40 (100%) | 126.79 ms | 203.39 ms |
| Hybrid | 40/40 (100%) | 132.51 ms | 387.92 ms |
| Local Ministral | 39/40 (97.5%) | 411.23 ms | 624.25 ms |

The hybrid evaluation used a 0.70 confidence threshold.

The current application threshold of 0.60 has not been separately benchmarked on this test set.

The 40-question evaluation showed:

- Jev provided the lowest median routing latency.
- Local Ministral achieved high accuracy but required more inference time.
- Hybrid routing preserved 100% accuracy but introduced unnecessary fallbacks on two questions.
- One cold local fallback took approximately 8.6 seconds.

An earlier 20-question evaluation produced:

- Jev: 19/20
- Hybrid: 20/20

### Routing Conclusions

Jev was the strongest option for default routing in these experiments.

Hybrid routing provides a fallback mechanism but does not guarantee better latency or accuracy.

The small evaluation sets are not sufficient to establish general production accuracy. Results also depend on question wording, API behavior, and model state.

The evaluation questions and benchmark scripts are retained for reproducibility.

---

## 5. Retrieval-Augmented Generation

The RAG pipeline enables the copilot to answer questions about experiments, implementations, and performance measurements throughout the Local AI Learning repository.

### Ingestion Pipeline

Repository Markdown and Python files are processed into structured chunks.

Python parsing uses AST-based structure to preserve useful code boundaries.

```text
Repository Files
      |
      v
Structured Parsing
      |
      v
842 Raw Chunks
      |
      v
Low-Information Filtering
      |
      v
501 Indexed Chunks
      |
      v
EmbeddingGemma
      |
      v
Persistent ChromaDB Index
```

Filtering removes low-information content before indexing.

### Retrieval Pipeline

```text
User Question
      |
      v
EmbeddingGemma Query Embedding
      |
      v
ChromaDB Vector Search
      |
      v
Top 10 Candidates
      |
      v
Cross-Encoder Reranking
      |
      v
Relevance Filtering
      |
      v
Up to 4 Selected Chunks
      |
      v
Source-Formatted Context
      |
      v
Ministral Response
```

The reranker uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Additional ranking logic accounts for source relevance and query characteristics.

For example, conceptual questions may benefit from README content, while implementation questions may benefit from Python source files.

Retrieved chunks are presented using source identifiers such as:

```text
[SOURCE 1]
[SOURCE 2]
[SOURCE 3]
```

The model is instructed to ground repository-specific answers in these retrieved sources.

### RAG Performance Profiling

Profiling identified the following latency breakdown for one representative search:

| Stage | Latency |
|---|---:|
| Query embedding | 1,674.98 ms |
| ChromaDB query | 23.68 ms |
| Candidate construction | 0.05 ms |
| Reranking | 810.80 ms |
| Total search | 2,510.62 ms |

The query embedding operation accounted for approximately 67% of observed search latency, while reranking accounted for approximately 32%.

The vector database lookup itself represented only a small fraction of total latency.

### Important Observation

Vector search was not the primary retrieval bottleneck.

The dominant costs were query embedding and cross-encoder inference.

The measured embedding latency may include model startup or runtime scheduling overhead, rather than pure embedding computation.

Repeated warm measurements would be required to isolate these effects more precisely.

---

## 6. Cold Starts and End-to-End Latency

The application measures several distinct performance components:

- Routing latency
- Module import latency
- RAG initialization latency
- Retrieval and reranking latency
- Model loading latency
- Prompt processing latency
- Token generation latency
- Model-level time to first token
- Application-level time to first token
- End-to-end request latency

### Model-Level vs. Application-Level TTFT

Model-level TTFT measures the time from initiating the Ollama generation request to receiving the first visible output text.

Application-level TTFT measures the time from the beginning of the user request, including routing and preprocessing.

These are different measurements.

For RAG:

```text
Application-Level TTFT
        =
Routing
        +
Module Import / Initialization
        +
Retrieval and Reranking
        +
Model-Level TTFT
        +
Minor Application Overhead
```

### Cold vs. Warm RAG

A representative two-request experiment produced:

| Metric | Cold Request | Warm Request |
|---|---:|---:|
| RAG module import | 5,436.92 ms | 0.42 ms |
| RAG engine initialization | 2,362.79 ms | 0 ms |
| Search and reranking | 2,368.30 ms | 3,936.24 ms |
| Model-level TTFT | 5,244.12 ms | 1,118.18 ms |
| Application-level TTFT | 15,651.90 ms | 5,476.01 ms |
| Total request latency | 24,175.31 ms | 14,866.71 ms |

The first request incurred substantial Python import, RAG initialization, and model-loading overhead.

The second request avoided most initialization work.

However, retrieval was slower on the second request, demonstrating that not every pipeline stage necessarily improves with warming.

These measurements illustrate runtime behavior on the test machine rather than guaranteed production latency.

### Streaming

The copilot streams generated text as it becomes available.

Streaming improves perceived responsiveness by displaying output before generation completes.

It does not inherently increase the model's decoding throughput.

---

## 7. Deterministic Tool Execution

The TOOL route executes engineering calculations directly in Python.

This avoids asking a language model to perform arithmetic that can be calculated deterministically.

### Available Tools

| Tool | Function |
|---|---|
| MODEL_MEMORY | Estimate model-weight memory |
| KV_CACHE_MEMORY | Calculate KV-cache memory |
| TOKEN_THROUGHPUT | Calculate tokens per second |
| LATENCY_REDUCTION | Calculate percentage latency changes |
| EFFECTIVE_BATCH_SIZE | Calculate effective training batch size |
| GPU_RUNTIME_COST | Estimate GPU runtime cost |
| UNIT_CONVERSION | Convert supported units |
| SYSTEM_MEMORY | Inspect available system RAM |

### Example: Model Memory

For a model with 8 billion parameters stored at 2 bytes per parameter:

```text
8,000,000,000 parameters × 2 bytes
= 16,000,000,000 bytes

= 16 GB
≈ 14.9012 GiB
```

The calculation excludes KV cache, activations, temporary buffers, and runtime overhead.

### Example: KV-Cache Memory

For a transformer with:

```text
Layers:          32
KV heads:        8
Head dimension: 128
Context length: 8192
Batch size:      1
Precision:       BF16
```

The KV-cache equation is:

```text
KV Memory =
2 × Batch × Layers × KV Heads
  × Context Length × Head Dimension
  × Bytes Per Element
```

Therefore:

```text
2 × 1 × 32 × 8 × 8192 × 128 × 2
= 1,073,741,824 bytes
= 1 GiB
```

### Measured Tool Performance

| Tool | Python Execution |
|---|---:|
| MODEL_MEMORY | 0.54 ms |
| KV_CACHE_MEMORY | 1.22 ms |

In the integrated application, the total request time was approximately 0.2 seconds, dominated by Jev routing rather than calculation.

This demonstrates a useful system-design principle:

**Use an LLM where language understanding is valuable, and deterministic software where precise computation is required.**

The current dispatcher supports specific calculation types and input formats. It is not a general-purpose natural-language calculator.

---

## 8. Multimodal Inference

The MULTIMODAL route analyzes images using Ministral 3B through Ollama.

Supported image formats:

```text
PNG
JPG
JPEG
WEBP
```

Users can submit a screenshot, chart, or image and ask questions about its contents.

Examples include:

- Explaining performance graphs
- Interpreting terminal screenshots
- Reading benchmark tables
- Identifying visible error messages
- Describing hardware diagrams

### Execution Flow

```text
Image + Question
       |
       v
Text-Based Route Classification
       |
       v
MULTIMODAL
       |
       v
Local Image Processing
       |
       v
Ministral Vision
       |
       v
Streamed Explanation
       |
       v
Performance Report
```

Image data is supplied directly to local Ollama.

Only textual routing information is sent to Jev.

### Multimodal Performance

A representative image-analysis request produced:

| Metric | Measurement |
|---|---:|
| Jev routing | 231.84 ms |
| Model-level TTFT | 11,779.39 ms |
| Application-level TTFT | 12,011.51 ms |
| Model loading | 6,059.69 ms |
| Prompt processing | 5,300.02 ms |
| Token generation | 12,496.09 ms |
| Input tokens | 1,062 |
| Output tokens | 400 |
| Decode speed | 32.01 tokens/sec |
| End-to-end latency | 24,384.63 ms |

The model correctly extracted several numerical values from a supplied GPU market-share infographic.

However, it also introduced unsupported interpretations, demonstrating that visual extraction accuracy and factual reasoning accuracy must be evaluated separately.

### Key Observation

Visual input substantially increased prompt-processing cost compared with representative short text-only requests.

However, decoding remained around 32 tokens/second, close to observed text-only decoding speeds on the same MacBook.

This suggests that much of the additional multimodal latency occurred before token generation.

The comparison is illustrative because the prompts and workloads were not identical.

---

## 9. Conversation Memory

The copilot maintains a bounded in-memory conversation history.

The current configuration retains the most recent four completed exchanges.

Each exchange contains:

- User question
- Assistant answer
- Execution route
- Optional image filename

The application does not persist the conversation to disk.

Images themselves are not retained in the memory structure.

### Example

```text
USER:
Explain prefill and decode.

ASSISTANT:
Prefill processes input tokens...
Decode generates output tokens...

USER:
Which one is typically more bandwidth-bound?
```

Conversation memory helps resolve what "which one" refers to.

The model receives a bounded amount of previous context rather than the entire conversation history.

### Contextual Query Rewriting

Ambiguous follow-up questions can be expanded using the previous conversation topic.

Example:

```text
Previous question:
What did I learn about multimodal inference
in Chapter 17?

Follow-up:
And what happened to throughput
at concurrency 8?
```

The memory layer supplies context indicating that the follow-up refers to previously recorded Chapter 17 measurements.

### Evidence-Aware Routing

The application distinguishes between two intentions:

**Evidence retrieval**

```text
What throughput did I measure at concurrency 8?
```

Expected route:

```text
RAG
```

**Performance analysis**

```text
Why might throughput increase
as concurrency increases?
```

Expected route:

```text
BENCHMARK_ANALYSIS
```

A narrow application-level routing policy can override Jev when a follow-up clearly requests existing repository evidence.

The original Jev prediction and any application override are logged separately.

This preserves transparency in routing evaluation.

### Contextual Routing Test

An initial Chapter 17 follow-up was incorrectly classified as BENCHMARK_ANALYSIS.

After introducing contextual query rewriting, the same follow-up was correctly classified as RAG with confidence 1.000.

A subsequent causal-analysis question correctly selected BENCHMARK_ANALYSIS.

These are successful integration tests, not a comprehensive multi-turn accuracy benchmark.

---

## 10. Observability

Each request generates a performance report.

Depending on the execution route, measurements include:

```text
ROUTING
    Jev prediction
    Final route
    Confidence
    Routing latency
    Fallback / override information

RETRIEVAL
    Module import
    Engine initialization
    Query embedding
    Vector search
    Cross-encoder reranking
    Selected chunks

INFERENCE
    Model-level TTFT
    Application-level TTFT
    Model loading
    Prompt processing
    Token generation
    Input tokens
    Output tokens
    Decode speed
    Stop reason

TOOL EXECUTION
    Selected tool
    Python execution latency

SESSION
    Request count
    Average routing latency
    Average execution latency
    Average end-to-end latency
    Estimated Jev cost
```

### Why This Matters

A slow response does not necessarily mean slow token generation.

Latency may come from:

- External routing
- Python imports
- Model startup
- Embedding computation
- Vector retrieval
- Cross-encoder reranking
- Image processing
- Prompt prefill
- Autoregressive decoding

Measuring these components separately helps identify the actual bottleneck.

---

## 11. Running the Application

### Requirements

- Python 3
- Ollama
- Local Ministral 3B model
- EmbeddingGemma
- ChromaDB
- Sentence Transformers
- TypeSafe SDK and an active Jev API key
- Python dependencies used by the project

This project was developed and tested on an Apple M2 MacBook Air with 8 GB unified memory.

### Local Models

Ensure Ollama is installed and running.

Pull the required models:

```bash
ollama pull ministral-3:3b
ollama pull embeddinggemma
```

### API Configuration

Create a `.env` file inside the Chapter 18 directory:

```text
TYPESAFE_API_KEY=your_api_key_here
```

The `.env` file must remain excluded from Git.

Do not send passwords, API keys, confidential work material, or other sensitive information through the external router.

### Build the Repository Index

From the Chapter 18 directory:

```bash
python3 ingest_repo.py
python3 build_index.py
```

The persistent ChromaDB index is generated locally and should not be committed to Git.

### Start the Copilot

Default Jev routing:

```bash
python3 copilot.py --router jev
```

Hybrid routing:

```bash
python3 copilot.py --router hybrid --threshold 0.60
```

Increase the response-token limit:

```bash
python3 copilot.py --router jev --max-tokens 600
```

### Interactive Commands

| Command | Action |
|---|---|
| `/image` | Attach and analyze an image |
| `/memory` | Display recent conversation history |
| `/clear` | Clear conversational memory |
| `exit` | End the session |

### Example Questions

**CHAT**

```text
Explain the difference between BF16 and FP32.
```

**RAG**

```text
What did I learn about multimodal inference
in Chapter 17?
```

**BENCHMARK_ANALYSIS**

```text
In my LLM inference benchmark, GPU utilization
increased as batch size increased.

Analyze possible bottlenecks and suggest
profiling experiments.
```

**TOOL**

```text
Calculate the memory in GiB needed to store
8 billion parameters at 2 bytes per parameter.
```

**MULTIMODAL**

```text
/image
```

Then provide a local image path and ask a question about the image.

---

## 12. Project Files

Key application modules include:

| File | Responsibility |
|---|---|
| `copilot.py` | Main application and execution dispatcher |
| `conversation_memory.py` | Bounded memory and contextual routing |
| `copilot_tools.py` | Deterministic engineering calculators |
| `router.py` | Local Ministral router |
| `jev_router.py` | Jev router |
| `hybrid_router.py` | Hybrid routing logic |
| `rag.py` | Retrieval and reranking pipeline |
| `ingest_repo.py` | Repository parsing and chunking |
| `build_index.py` | Embedding generation and vector indexing |
| `answer_rag.py` | Standalone grounded RAG answering |
| `test_multimodal.py` | Independent vision inference test |
| `test_retrieval.py` | Retrieval evaluation |
| `test_reranking.py` | Reranking evaluation |
| `router_test_cases.py` | Original routing test cases |
| `router_unseen_test_cases.py` | Additional routing evaluation set |
| `benchmark_unseen_routers.py` | Routing benchmark comparisons |

The project also uses a locally generated ChromaDB index and a gitignored `.env` file.

---

## 13. Known Limitations

### Small-Model Reasoning Accuracy

Ministral occasionally provides incorrect technical explanations.

For example, in one test, it incorrectly reversed the typical performance characteristics of LLM prefill and decode.

Conversation memory can preserve context, but it does not guarantee factual correctness.

### Unsupported Causal Claims

The model sometimes attributes performance differences to specific bottlenecks without sufficient profiling evidence.

A measured throughput difference does not automatically establish its cause.

### Routing Ambiguity

Some questions can reasonably fit multiple categories.

For example, a general question about GPU utilization may be classified as CHAT, while an explicit request to investigate measured utilization belongs to BENCHMARK_ANALYSIS.

Routing accuracy depends on the intended task and how clearly the question expresses it.

### Heuristic Contextual Rewriting

The memory layer uses deterministic heuristics for detecting and rewriting conversational follow-ups.

It does not perform full semantic query rewriting.

Ambiguous references and longer conversations can still cause routing or retrieval errors.

### Retrieval Grounding

Retrieved sources improve factual grounding but do not guarantee that generated conclusions are supported.

Source citations and causal explanations still require validation.

### Constrained Tool Parsing

The deterministic toolbox supports predefined operations and question formats.

It does not support arbitrary mathematical expressions or open-ended tool execution.

### Output Truncation

Some generation requests reach the configured output-token limit.

The resulting responses can be incomplete.

### Local Hardware Constraints

The 8 GB unified-memory environment limits model selection, concurrent model residency, and available context.

Large models or multiple simultaneously loaded models may introduce substantial memory pressure.

### Evaluation Scope

Routing benchmarks used small, balanced question sets.

Inference measurements were collected on a specific local hardware configuration.

The results should not be generalized to other models, GPUs, hardware platforms, or production workloads without further testing.

---

## 14. Lessons Learned

### 1. Routing and Generation Are Different Problems

An application can select the correct route while generating an inaccurate answer.

Routing accuracy alone is not a complete measure of assistant quality.

### 2. End-to-End Latency Matters

Model inference latency excludes substantial application overhead.

For an actual user, routing, imports, retrieval, preprocessing, and inference all contribute to response time.

### 3. Cold Starts Can Dominate Responsiveness

Model loading and Python initialization can account for several seconds of additional latency.

Warm measurements should not be interpreted without understanding initialization costs.

### 4. Vector Search Is Not Necessarily the RAG Bottleneck

In this project, embedding generation and reranking were significantly more expensive than the ChromaDB query.

Optimization should follow measurement rather than assumptions.

### 5. Deterministic Tools Can Be More Efficient Than LLM Generation

Engineering calculations executed in milliseconds, while local LLM generation often required several seconds.

A well-designed assistant should use the simplest reliable execution method for each task.

### 6. Conversation Memory Requires Context-Aware Routing

Remembering previous exchanges is not sufficient by itself.

Follow-up questions also need appropriate route selection and, when necessary, contextual retrieval.

### 7. Correct Numbers Do Not Guarantee Correct Reasoning

An answer can reproduce accurate benchmark measurements while giving unsupported explanations for the observed differences.

Evidence retrieval and causal analysis must remain distinct.

### 8. Hardware Characteristics Shape Application Design

Unified memory capacity, model loading, prompt-processing cost, and decoding throughput all influenced architectural decisions.

Efficient AI application development requires understanding the systems executing the models.

---

## 15. Future Improvements

Potential improvements include:

- Larger multi-turn routing evaluation sets
- Automated technical correctness tests
- Better contextual query rewriting
- Retrieval-backed performance analysis
- Stronger citation verification
- Structured tool argument extraction
- More robust input validation
- Prompt-length and memory-budget controls
- Image preprocessing and resolution experiments
- RAG embedding latency optimization
- Persistent conversation storage
- Automated regression testing
- Comparing additional local models
- Quantifying cold-start versus warm-start behavior across repeated trials

---

## 16. Connection to the Local AI Learning Journey

This final application integrates concepts from the preceding chapters.

| Area | Concepts Applied |
|---|---|
| Local inference | Ollama, token generation, model execution |
| Grounding | RAG, source citations, relevance filtering |
| Retrieval | Embeddings, vector search, reranking |
| Memory | Conversation history, contextual resolution |
| Tool calling | Structured execution and deterministic calculations |
| Agents | Routing and execution orchestration |
| Fine-tuning | Understanding model adaptation and limitations |
| Quantization | Model memory and numerical precision |
| Benchmarking | TTFT, TPOT, throughput, latency |
| Serving | Inference performance and request concurrency |
| Multimodal | Vision-language inference and image processing |

The project demonstrates how these individual components interact in a complete application.

---

## 17. Final Takeaways

The AI Engineering Copilot demonstrates a working local-first assistant with five specialized execution routes, conversation memory, retrieval grounding, multimodal understanding, and detailed observability.

The most important outcomes were not simply getting the models to produce answers.

They were learning how to:

- Measure system behavior
- Identify performance bottlenecks
- Separate routing quality from answer quality
- Build retrieval pipelines
- Use deterministic tools appropriately
- Understand model loading and inference overhead
- Integrate multiple AI components
- Evaluate failures systematically

The project also illustrates the limitations of smaller local models and the importance of grounding, measurement, and careful interpretation.

**The central engineering lesson is that AI system performance and reliability depend on the complete execution pipeline—not just the underlying language model.**

---

## Next Project

# AI Training and Inference Project

The next stage of this learning journey moves deeper into the systems that execute and optimize modern AI workloads.

Planned topics include:

- GPU architecture and memory hierarchy
- Roofline performance modeling
- HIP and Triton kernel programming
- GEMM optimization
- Attention mechanisms
- FlashAttention
- KV-cache optimization
- PagedAttention
- Quantization and low-precision computation
- Speculative decoding
- Mixture-of-Experts execution
- vLLM and SGLang
- ROCm and AMD Instinct GPUs
- Multi-GPU communication
- Distributed training
- AI-assisted kernel optimization
- End-to-end inference serving optimization

The long-term objective is to develop practical GPU performance engineering skills through implementations, profiling, optimization experiments, and open-source contributions.

---

**Project:** Local AI Learning — Chapter 18  
**Focus:** AI Applications, Inference Systems, RAG, Routing, and Performance Engineering  
**Hardware:** Apple M2 MacBook Air, 8 GB Unified Memory  
**Language:** Python