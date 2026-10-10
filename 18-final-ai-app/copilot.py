# ==================================================
# CHAPTER 18 — AI ENGINEERING COPILOT
# ==================================================
#
# Routing:
#   Jev
#   Hybrid: Jev + local Ministral fallback
#
# Execution:
#   CHAT
#   RAG
#   BENCHMARK_ANALYSIS
#   TOOL
#   MULTIMODAL
#
# Features:
#   Contextual query rewriting
#   Evidence-aware routing safeguards
#   Bounded conversation memory
#   Streaming local inference
#   Retrieval + cross-encoder reranking
#   Deterministic engineering calculations
#   Image understanding
#   End-to-end observability
#
# ==================================================

import argparse
import importlib
import os
import shlex
import time

from pathlib import Path
from statistics import mean

import ollama

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from conversation_memory import ConversationMemory
from copilot_tools import execute_tool

from router_test_cases import ROUTES
from router import ROUTER_MODEL

from jev_router import (
    MODEL as JEV_MODEL,
    route_query as jev_route_query,
)

from hybrid_router import run_local_fallback


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

ROUTER_MODES = ["jev", "hybrid"]

DEFAULT_THRESHOLD = 0.60
DEFAULT_MAX_TOKENS = 400
DEFAULT_KEEP_ALIVE = "5m"

MEMORY_TURNS = 4

SUPPORTED_IMAGES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# --------------------------------------------------
# SYSTEM PROMPTS
# --------------------------------------------------

CHAT_PROMPT = """
You are an AI Engineering Copilot.

Explain AI systems, GPU architecture, machine
learning, training, and inference accurately.

Use recent conversation history to understand
follow-up questions.

Distinguish measurements, established facts,
hypotheses, and illustrative examples.

Do not invent experimental results.
"""


BENCHMARK_PROMPT = """
You are an AI systems performance engineer.

Analyze the user's performance observations.

Explain:
1. What measurements demonstrate.
2. Plausible performance bottlenecks.
3. What remains uncertain.
4. Experiments that could verify hypotheses.

Consider TTFT, TPOT, prefill, decode, batching,
memory bandwidth, arithmetic intensity,
quantization, and GPU utilization.

Previous assistant answers are not independent
evidence. Do not invent measurements or imply
that hypotheses were experimentally verified.
"""


RAG_PROMPT = """
You are an AI Engineering Copilot answering
questions about the user's local-ai-learning
repository.

Use ONLY the CURRENT retrieved repository
sources for claims about experiments,
implementations, and measurements.

A previous question may be provided solely
to resolve references in the current question.

Do not treat conversation history as evidence.

Cite relevant sources using [SOURCE 1],
[SOURCE 2], and so on.

If the sources are insufficient, say so.

Do not invent measurements or citations.
"""


VISION_PROMPT = """
You are an AI Engineering Copilot analyzing
an image supplied by the user.

Read visible measurements, labels, and text.

Distinguish observations from hypotheses.

Use conversation context when relevant.

Do not invent visual details or treat claims
printed in an image as independently verified.
"""


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_field(obj, name, default=None):

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def ns_to_ms(value):

    if value is None:
        return None

    return value / 1_000_000


def format_ms(value):

    if value is None:
        return "N/A"

    return f"{value:.2f} ms"


def validate_image_path(raw_path):

    parts = shlex.split(
        raw_path.strip()
    )

    if len(parts) != 1:
        raise ValueError(
            "Provide exactly one image path."
        )

    path = Path(
        parts[0]
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError(
            "Supported formats: PNG, JPG, JPEG, WEBP."
        )

    return path


# --------------------------------------------------
# ROUTING POLICY
# --------------------------------------------------

def apply_routing_policy(
    predicted_route,
    resolution,
    has_image=False,
):

    # An attached image must use the vision path.
    #
    # The actual image is never sent to Jev.

    if has_image:

        if predicted_route != "MULTIMODAL":

            return (
                "MULTIMODAL",
                "ATTACHED_IMAGE_REQUIRES_VISION",
            )

        return "MULTIMODAL", None

    # A narrowly identified repository evidence
    # follow-up must use retrieval.
    #
    # This does not change Jev's original
    # prediction or confidence.

    if (
        resolution.preferred_route == "RAG"
        and predicted_route != "RAG"
    ):

        return (
            "RAG",
            resolution.policy_reason,
        )

    return predicted_route, None


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

class EngineeringCopilot:

    def __init__(
        self,
        router_mode="jev",
        threshold=DEFAULT_THRESHOLD,
        max_tokens=DEFAULT_MAX_TOKENS,
        keep_alive=DEFAULT_KEEP_ALIVE,
    ):

        if router_mode not in ROUTER_MODES:
            raise ValueError(
                "Invalid router mode."
            )

        if not 0 <= threshold <= 1:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        if max_tokens < 1:
            raise ValueError(
                "max_tokens must be positive."
            )

        self.router_mode = router_mode
        self.threshold = threshold

        self.max_tokens = max_tokens
        self.keep_alive = keep_alive

        self.generator = ROUTER_MODEL

        self.memory = ConversationMemory(
            max_turns=MEMORY_TURNS
        )

        self.client = None
        self.client_context = None

        self.rag_engine = None

        self.history = []

        self.fallback_count = 0
        self.override_count = 0
        self.total_jev_cost = 0.0

    # ----------------------------------------------
    # START / STOP
    # ----------------------------------------------

    def start(self):

        if not os.getenv("TYPESAFE_API_KEY"):

            raise RuntimeError(
                "TYPESAFE_API_KEY missing from .env"
            )

        self.client_context = TypeSafeClient(
            model=JEV_MODEL
        )

        self.client = (
            self.client_context.__enter__()
        )

    def stop(self):

        if self.client_context is not None:

            self.client_context.__exit__(
                None, None, None
            )

            self.client_context = None
            self.client = None

    # ----------------------------------------------
    # CONTEXT-AWARE ROUTING
    # ----------------------------------------------

    def route(
        self,
        query,
        resolution,
        has_image=False,
    ):

        start = time.perf_counter()

        # The rewritten request provides Jev
        # with conversational context.
        #
        # Previous assistant answers are excluded.

        routing_query = (
            resolution.rewritten_query
        )

        if has_image:

            routing_query = (
                "An image is attached. "
                "The current task requires "
                "visual understanding.\n"
                + routing_query
            )

        jev_prediction = None
        confidence = None
        cost = 0.0

        fallback_reason = None
        router_used = "JEV"

        try:

            result = jev_route_query(
                self.client,
                routing_query,
            )

            jev_prediction = (
                result["predicted"]
            )

            confidence = (
                result["confidence"]
            )

            cost = (
                result["estimated_cost"]
            )

            if jev_prediction not in ROUTES:

                fallback_reason = (
                    "INVALID_JEV_ROUTE"
                )

        except Exception as error:

            if self.router_mode == "jev":
                raise

            fallback_reason = (
                f"JEV_ERROR: "
                f"{type(error).__name__}"
            )

        # Hybrid confidence gate.

        if (
            self.router_mode == "hybrid"
            and fallback_reason is None
            and (
                confidence is None
                or confidence < self.threshold
            )
        ):

            fallback_reason = (
                "LOW_CONFIDENCE"
            )

        if fallback_reason is None:

            predicted_route = jev_prediction

        elif self.router_mode == "hybrid":

            local_result = run_local_fallback(
                routing_query
            )

            predicted_route = (
                local_result["predicted"]
            )

            router_used = (
                "MINISTRAL_FALLBACK"
            )

            self.fallback_count += 1

        else:

            raise ValueError(
                f"Invalid Jev route: "
                f"{jev_prediction}"
            )

        if predicted_route not in ROUTES:

            raise ValueError(
                f"Invalid predicted route: "
                f"{predicted_route}"
            )

        # Apply our narrowly scoped application
        # policy AFTER obtaining the model route.

        final_route, override_reason = (
            apply_routing_policy(
                predicted_route,
                resolution,
                has_image=has_image,
            )
        )

        if override_reason:
            self.override_count += 1

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        self.total_jev_cost += cost

        return {
            "route": final_route,
            "model_prediction": predicted_route,
            "jev_prediction": jev_prediction,
            "router_used": router_used,
            "confidence": confidence,
            "fallback_reason": fallback_reason,
            "override_reason": override_reason,
            "latency_ms": latency_ms,
            "cost": cost,
        }

    # ----------------------------------------------
    # BUILD GENERATION MESSAGES
    # ----------------------------------------------

    def build_messages(
        self,
        system_prompt,
        current_message,
        image_path=None,
        include_history=True,
    ):

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if include_history:

            messages.extend(
                self.memory.history_messages()
            )

        user_message = {
            "role": "user",
            "content": current_message,
        }

        if image_path is not None:

            user_message["images"] = [
                str(image_path)
            ]

        messages.append(user_message)

        return messages

    # ----------------------------------------------
    # STREAMING INFERENCE
    # ----------------------------------------------

    def generate_stream(
        self,
        messages,
        request_start,
    ):

        model_start = time.perf_counter()

        model_ttft_ms = None
        app_ttft_ms = None

        answer_parts = []
        final_chunk = None

        stream = ollama.chat(
            model=self.generator,
            messages=messages,
            stream=True,
            keep_alive=self.keep_alive,
            options={
                "temperature": 0.2,
                "num_predict": self.max_tokens,
            },
        )

        print()
        print("GENERATED ANSWER")
        print("-" * 70)

        for chunk in stream:

            final_chunk = chunk

            message = get_field(
                chunk,
                "message",
                {},
            )

            content = get_field(
                message,
                "content",
                "",
            )

            if content:

                if (
                    model_ttft_ms is None
                    and content.strip()
                ):

                    now = time.perf_counter()

                    model_ttft_ms = (
                        now - model_start
                    ) * 1000

                    app_ttft_ms = (
                        now - request_start
                    ) * 1000

                print(
                    content,
                    end="",
                    flush=True,
                )

                answer_parts.append(content)

        print()

        model_wall_ms = (
            time.perf_counter() - model_start
        ) * 1000

        if final_chunk is None:

            raise RuntimeError(
                "Ollama returned no streaming data."
            )

        load_ms = ns_to_ms(
            get_field(
                final_chunk,
                "load_duration",
            )
        )

        prefill_ms = ns_to_ms(
            get_field(
                final_chunk,
                "prompt_eval_duration",
            )
        )

        decode_ms = ns_to_ms(
            get_field(
                final_chunk,
                "eval_duration",
            )
        )

        ollama_total_ms = ns_to_ms(
            get_field(
                final_chunk,
                "total_duration",
            )
        )

        input_tokens = get_field(
            final_chunk,
            "prompt_eval_count",
        )

        output_tokens = get_field(
            final_chunk,
            "eval_count",
        )

        done_reason = get_field(
            final_chunk,
            "done_reason",
        )

        speed = None

        if (
            decode_ms is not None
            and decode_ms > 0
            and output_tokens
        ):

            speed = (
                output_tokens
                / (decode_ms / 1000)
            )

        return {
            "kind": "generation",
            "answer": "".join(answer_parts),

            "model_ttft_ms": model_ttft_ms,
            "app_ttft_ms": app_ttft_ms,

            "model_wall_ms": model_wall_ms,
            "ollama_total_ms": ollama_total_ms,

            "load_ms": load_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,

            "input_tokens": input_tokens,
            "output_tokens": output_tokens,

            "tokens_per_second": speed,
            "done_reason": done_reason,
        }

    # ----------------------------------------------
    # CHAT
    # ----------------------------------------------

    def execute_chat(
        self,
        query,
        request_start,
    ):

        messages = self.build_messages(
            CHAT_PROMPT,
            query,
        )

        return self.generate_stream(
            messages,
            request_start,
        )

    # ----------------------------------------------
    # BENCHMARK ANALYSIS
    # ----------------------------------------------

    def execute_benchmark(
        self,
        query,
        request_start,
    ):

        messages = self.build_messages(
            BENCHMARK_PROMPT,
            query,
        )

        return self.generate_stream(
            messages,
            request_start,
        )

    # ----------------------------------------------
    # RAG
    # ----------------------------------------------

    def execute_rag(
        self,
        query,
        request_start,
        resolution,
    ):

        import_start = time.perf_counter()

        rag_module = importlib.import_module(
            "rag"
        )

        import_ms = (
            time.perf_counter() - import_start
        ) * 1000

        init_ms = 0.0

        if self.rag_engine is None:

            print(
                "[RAG] Initializing engine..."
            )

            init_start = time.perf_counter()

            self.rag_engine = (
                rag_module.RAGEngine()
            )

            init_ms = (
                time.perf_counter() - init_start
            ) * 1000

        # IMPORTANT:
        #
        # Retrieval uses the contextual query.
        # It does NOT use previous AI answers.

        search_query = (
            resolution.retrieval_query
        )

        if resolution.is_followup:

            print(
                "[RAG] Contextual retrieval: YES"
            )

        print(
            "[RAG] Retrieving and reranking..."
        )

        search_start = time.perf_counter()

        results, internal_metrics = (
            self.rag_engine.search(
                search_query,
                top_k=4,
            )
        )

        search_ms = (
            time.perf_counter() - search_start
        ) * 1000

        rag_metrics = {
            "import_ms": import_ms,
            "initialization_ms": init_ms,
            "search_ms": search_ms,
            "contextual_search":
                resolution.is_followup,
            "internal_metrics": internal_metrics,
            "chunks_selected": len(results),
        }

        # Do not generate repository facts when
        # retrieval returns no supporting chunks.

        if not results:

            answer = (
                "I couldn't find supporting "
                "repository sources for this "
                "question, so I can't confirm "
                "the requested measurement."
            )

            return {
                "kind": "retrieval_miss",
                "answer": answer,
                "rag_metrics": rag_metrics,
            }

        context = rag_module.format_context(
            results
        )

        # RAG deliberately excludes previous
        # assistant answers.
        #
        # Earlier questions can resolve pronouns,
        # but only current sources are evidence.

        current_message = (
            f"CURRENT QUESTION:\n{query}\n\n"
            f"CONTEXTUAL INTERPRETATION:\n"
            f"{resolution.rewritten_query}\n\n"
            f"RETRIEVED REPOSITORY SOURCES:\n"
            f"{context}"
        )

        messages = self.build_messages(
            RAG_PROMPT,
            current_message,
            include_history=False,
        )

        generation = self.generate_stream(
            messages,
            request_start,
        )

        generation["rag_metrics"] = (
            rag_metrics
        )

        return generation

    # ----------------------------------------------
    # TOOL
    # ----------------------------------------------

    def execute_tool_route(self, query):

        tool_result = execute_tool(
            query
        )

        return {
            "kind": "tool",
            "answer": tool_result["answer"],
            "tool_result": tool_result,
        }

    # ----------------------------------------------
    # MULTIMODAL
    # ----------------------------------------------

    def execute_multimodal(
        self,
        query,
        request_start,
        image_path,
    ):

        if image_path is None:

            raise ValueError(
                "MULTIMODAL needs an image. "
                "Use /image to attach one."
            )

        print()
        print(
            f"[MULTIMODAL] Image: "
            f"{image_path.name}"
        )

        messages = self.build_messages(
            VISION_PROMPT,
            query,
            image_path=image_path,
        )

        generation = self.generate_stream(
            messages,
            request_start,
        )

        generation["image_name"] = (
            image_path.name
        )

        return generation

    # ----------------------------------------------
    # EXECUTION DISPATCHER
    # ----------------------------------------------

    def execute(
        self,
        route,
        query,
        request_start,
        resolution,
        image_path=None,
    ):

        if route == "CHAT":

            return self.execute_chat(
                query,
                request_start,
            )

        if route == "RAG":

            return self.execute_rag(
                query,
                request_start,
                resolution,
            )

        if route == "BENCHMARK_ANALYSIS":

            return self.execute_benchmark(
                query,
                request_start,
            )

        if route == "TOOL":

            return self.execute_tool_route(
                query
            )

        if route == "MULTIMODAL":

            return self.execute_multimodal(
                query,
                request_start,
                image_path,
            )

        raise ValueError(
            f"Unsupported route: {route}"
        )

    # ----------------------------------------------
    # PERFORMANCE REPORT
    # ----------------------------------------------

    def print_performance(
        self,
        routing,
        result,
        execution_ms,
        total_ms,
        previous_turns,
    ):

        print()
        print("=" * 75)
        print("PERFORMANCE REPORT")
        print("=" * 75)

        print()
        print(
            f"Final route: {routing['route']}"
        )

        print(
            f"Router used: {routing['router_used']}"
        )

        print(
            f"Jev prediction: "
            f"{routing['jev_prediction']}"
        )

        print(
            f"Routing latency: "
            f"{format_ms(routing['latency_ms'])}"
        )

        if routing["override_reason"]:

            print()
            print(
                f"Application override: "
                f"{routing['override_reason']}"
            )

        # ------------------------------------------
        # MEMORY
        # ------------------------------------------

        print()
        print("CONVERSATION MEMORY")
        print("-" * 75)

        print(
            f"Previous exchanges: {previous_turns}"
        )

        print(
            f"Capacity: {MEMORY_TURNS}"
        )

        # ------------------------------------------
        # TOOL METRICS
        # ------------------------------------------

        if result["kind"] == "tool":

            tool = result["tool_result"]

            print()
            print("TOOL EXECUTION")
            print("-" * 75)

            print(
                f"Tool: {tool['tool']}"
            )

            print(
                f"Tool latency: "
                f"{format_ms(tool['latency_ms'])}"
            )

            print(
                "LLM generation: NOT REQUIRED"
            )

        # ------------------------------------------
        # RAG METRICS
        # ------------------------------------------

        if "rag_metrics" in result:

            rag = result["rag_metrics"]

            print()
            print("RAG TIMINGS")
            print("-" * 75)

            print(
                f"Module import: "
                f"{format_ms(rag['import_ms'])}"
            )

            print(
                f"Initialization: "
                f"{format_ms(rag['initialization_ms'])}"
            )

            print(
                f"Search + rerank: "
                f"{format_ms(rag['search_ms'])}"
            )

            print(
                f"Contextual retrieval: "
                f"{rag['contextual_search']}"
            )

            print(
                f"Chunks selected: "
                f"{rag['chunks_selected']}"
            )

            internal = rag.get(
                "internal_metrics"
            )

            if isinstance(internal, dict):

                print()
                print("RAG INTERNAL STAGES")

                for key, value in internal.items():

                    if (
                        key.endswith("_ms")
                        and isinstance(
                            value,
                            (int, float),
                        )
                    ):

                        print(
                            f"  {key}: "
                            f"{value:.2f} ms"
                        )

        # ------------------------------------------
        # INFERENCE METRICS
        # ------------------------------------------

        if result["kind"] == "generation":

            if "image_name" in result:

                print()
                print(
                    f"Image: {result['image_name']}"
                )

            print()
            print("TIME TO FIRST TOKEN")
            print("-" * 75)

            print(
                f"Model-level TTFT: "
                f"{format_ms(result['model_ttft_ms'])}"
            )

            print(
                f"Application-level TTFT: "
                f"{format_ms(result['app_ttft_ms'])}"
            )

            print()
            print("INFERENCE")
            print("-" * 75)

            print(
                f"Model loading: "
                f"{format_ms(result['load_ms'])}"
            )

            print(
                f"Prompt processing: "
                f"{format_ms(result['prefill_ms'])}"
            )

            print(
                f"Token generation: "
                f"{format_ms(result['decode_ms'])}"
            )

            print(
                f"Ollama total: "
                f"{format_ms(result['ollama_total_ms'])}"
            )

            print()
            print(
                f"Input tokens: "
                f"{result['input_tokens']}"
            )

            print(
                f"Output tokens: "
                f"{result['output_tokens']}"
            )

            speed = result["tokens_per_second"]

            if speed is not None:

                print(
                    f"Decode speed: "
                    f"{speed:.2f} tok/s"
                )

            print()
            print(
                f"Stop reason: "
                f"{result['done_reason']}"
            )

            if result["done_reason"] == "length":

                print(
                    "WARNING: Output token "
                    "limit reached."
                )

        # ------------------------------------------
        # END-TO-END
        # ------------------------------------------

        print()
        print("END-TO-END")
        print("-" * 75)

        print(
            f"Execution: "
            f"{format_ms(execution_ms)}"
        )

        print(
            f"Total request: "
            f"{format_ms(total_ms)}"
        )

        print(
            f"Estimated Jev cost: "
            f"${routing['cost']:.8f}"
        )

    # ----------------------------------------------
    # HANDLE ONE QUESTION
    # ----------------------------------------------

    def ask(
        self,
        query,
        image_path=None,
    ):

        request_start = time.perf_counter()

        previous_turns = (
            self.memory.count()
        )

        resolution = (
            self.memory.resolve(query)
        )

        has_image = (
            image_path is not None
        )

        print()
        print("=" * 75)
        print("AI ENGINEERING COPILOT")
        print("=" * 75)

        print()
        print(f"Question: {query}")

        if has_image:

            print(
                f"Attached image: "
                f"{image_path.name}"
            )

        # ------------------------------------------
        # ROUTE
        # ------------------------------------------

        routing = self.route(
            query,
            resolution,
            has_image=has_image,
        )

        print()
        print("ROUTING")
        print("-" * 75)

        print(
            f"Mode: {self.router_mode}"
        )

        print(
            f"Contextual follow-up: "
            f"{resolution.is_followup}"
        )

        print(
            f"Prior turns: {previous_turns}"
        )

        print(
            f"Jev prediction: "
            f"{routing['jev_prediction']}"
        )

        print(
            f"Final decision: "
            f"{routing['route']}"
        )

        if routing["confidence"] is not None:

            print(
                f"Jev confidence: "
                f"{routing['confidence']:.3f}"
            )

        if routing["fallback_reason"]:

            print(
                f"Fallback: "
                f"{routing['fallback_reason']}"
            )

        if routing["override_reason"]:

            print(
                f"Routing override: "
                f"{routing['override_reason']}"
            )

        # ------------------------------------------
        # EXECUTE
        # ------------------------------------------

        execution_start = time.perf_counter()

        result = self.execute(
            routing["route"],
            query,
            request_start,
            resolution,
            image_path=image_path,
        )

        execution_ms = (
            time.perf_counter() - execution_start
        ) * 1000

        total_ms = (
            time.perf_counter() - request_start
        ) * 1000

        # Generation answers stream directly.
        # Tool and retrieval-miss answers don't.

        if result["kind"] == "tool":

            tool = result["tool_result"]

            print()
            print("TOOL RESULT")
            print("-" * 75)

            print(
                f"Answer: {tool['answer']}"
            )

            print()
            print(tool["explanation"])

        elif result["kind"] == "retrieval_miss":

            print()
            print("RAG ANSWER")
            print("-" * 75)

            print(result["answer"])

        # ------------------------------------------
        # REPORT
        # ------------------------------------------

        self.print_performance(
            routing,
            result,
            execution_ms,
            total_ms,
            previous_turns,
        )

        # ------------------------------------------
        # REMEMBER
        # ------------------------------------------

        memory_answer = result["answer"]

        if result["kind"] == "tool":

            memory_answer += (
                "\n"
                + result["tool_result"][
                    "explanation"
                ]
            )

        self.memory.remember(
            question=query,
            answer=memory_answer,
            route=routing["route"],
            image_name=(
                image_path.name
                if has_image
                else None
            ),
        )

        print()
        print(
            f"[MEMORY] Saved exchange: "
            f"{self.memory.count()}/"
            f"{MEMORY_TURNS}"
        )

        # ------------------------------------------
        # SESSION RECORD
        # ------------------------------------------

        record = {
            "query": query,
            "route": routing["route"],
            "jev_prediction": routing[
                "jev_prediction"
            ],
            "override_reason": routing[
                "override_reason"
            ],
            "kind": result["kind"],

            "routing_ms": routing["latency_ms"],
            "execution_ms": execution_ms,
            "total_ms": total_ms,

            "contextual_followup":
                resolution.is_followup,

            "model_ttft_ms": result.get(
                "model_ttft_ms"
            ),

            "app_ttft_ms": result.get(
                "app_ttft_ms"
            ),

            "tokens_per_second": result.get(
                "tokens_per_second"
            ),
        }

        if "rag_metrics" in result:

            record["rag_metrics"] = (
                result["rag_metrics"]
            )

        if result["kind"] == "tool":

            record["tool_latency_ms"] = (
                result["tool_result"][
                    "latency_ms"
                ]
            )

        self.history.append(record)

        return record

    # ----------------------------------------------
    # SESSION SUMMARY
    # ----------------------------------------------

    def print_summary(self):

        print()
        print("=" * 75)
        print("SESSION SUMMARY")
        print("=" * 75)

        print()
        print(
            f"Requests: {len(self.history)}"
        )

        print(
            f"Router: {self.router_mode}"
        )

        print(
            f"Hybrid fallbacks: "
            f"{self.fallback_count}"
        )

        print(
            f"Routing overrides: "
            f"{self.override_count}"
        )

        print(
            f"Remembered exchanges: "
            f"{self.memory.count()}"
        )

        print(
            f"Estimated Jev cost: "
            f"${self.total_jev_cost:.8f}"
        )

        if not self.history:
            return

        print()
        print(
            f"Average routing: "
            f"{mean(r['routing_ms'] for r in self.history):.2f} ms"
        )

        print(
            f"Average execution: "
            f"{mean(r['execution_ms'] for r in self.history):.2f} ms"
        )

        print(
            f"Average total: "
            f"{mean(r['total_ms'] for r in self.history):.2f} ms"
        )

        ttft_values = [
            r["app_ttft_ms"]
            for r in self.history
            if r["app_ttft_ms"] is not None
        ]

        if ttft_values:

            print(
                f"Average app TTFT: "
                f"{mean(ttft_values):.2f} ms"
            )


# --------------------------------------------------
# OFFLINE ROUTING POLICY TEST
# --------------------------------------------------

def run_routing_self_test():

    memory = ConversationMemory()

    memory.remember(
        (
            "What did I learn about multimodal "
            "inference in Chapter 17?"
        ),
        "Previous repository answer.",
        "RAG",
    )

    # Simulate Jev's observed wrong prediction.

    resolution = memory.resolve(
        "And what happened to throughput "
        "at concurrency 8?"
    )

    route, reason = apply_routing_policy(
        "BENCHMARK_ANALYSIS",
        resolution,
    )

    assert route == "RAG"
    assert reason is not None

    # Causal analysis must remain eligible
    # for BENCHMARK_ANALYSIS.

    resolution = memory.resolve(
        "Why did throughput decrease "
        "at concurrency 8?"
    )

    route, reason = apply_routing_policy(
        "BENCHMARK_ANALYSIS",
        resolution,
    )

    assert route == "BENCHMARK_ANALYSIS"
    assert reason is None

    # Image input must use MULTIMODAL.

    resolution = memory.resolve(
        "What does this image show?"
    )

    route, reason = apply_routing_policy(
        "CHAT",
        resolution,
        has_image=True,
    )

    assert route == "MULTIMODAL"
    assert reason is not None

    # An ordinary Jev prediction stays intact.

    memory.clear()

    resolution = memory.resolve(
        "Explain GPU arithmetic intensity."
    )

    route, reason = apply_routing_policy(
        "CHAT",
        resolution,
    )

    assert route == "CHAT"
    assert reason is None

    print()
    print("=" * 65)
    print("APPLICATION ROUTING POLICY TESTS")
    print("=" * 65)
    print("Tests passed: 4/4")
    print("No Jev API calls were made.")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="AI Engineering Copilot"
    )

    parser.add_argument(
        "--router",
        choices=ROUTER_MODES,
        default="jev",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )

    parser.add_argument(
        "--keep-alive",
        default=DEFAULT_KEEP_ALIVE,
    )

    parser.add_argument(
        "--self-test-routing",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test_routing:

        run_routing_self_test()
        return

    copilot = EngineeringCopilot(
        router_mode=args.router,
        threshold=args.threshold,
        max_tokens=args.max_tokens,
        keep_alive=args.keep_alive,
    )

    try:

        copilot.start()

        print()
        print("=" * 75)
        print("CHAPTER 18 — AI ENGINEERING COPILOT")
        print("=" * 75)

        print()
        print(
            f"Router: {args.router}"
        )

        print(
            f"Generator: {ROUTER_MODEL}"
        )

        print(
            f"Memory: {MEMORY_TURNS} exchanges"
        )

        print()
        print("Commands:")
        print("  /image   — Analyze an image")
        print("  /memory  — Inspect memory")
        print("  /clear   — Clear memory")
        print("  exit     — Quit")
        print()
        print("Or enter a normal question.")

        while True:

            try:

                query = input(
                    "\nYou: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print()
                break

            if query.lower() in (
                "exit",
                "quit",
            ):
                break

            if not query:
                continue

            if query.lower() == "/memory":

                copilot.memory.show()
                continue

            if query.lower() == "/clear":

                copilot.memory.clear()

                print(
                    "Conversation memory cleared."
                )

                continue

            try:

                if query.lower() == "/image":

                    raw_path = input(
                        "Image path "
                        "(drag from Finder): "
                    )

                    image_path = (
                        validate_image_path(
                            raw_path
                        )
                    )

                    image_question = input(
                        "Question about image: "
                    ).strip()

                    if not image_question:

                        image_question = (
                            "Describe what is visible "
                            "in this image."
                        )

                    copilot.ask(
                        image_question,
                        image_path=image_path,
                    )

                else:

                    copilot.ask(query)

            except Exception as error:

                print()
                print(
                    f"REQUEST ERROR: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

    finally:

        copilot.print_summary()
        copilot.stop()


if __name__ == "__main__":
    main()