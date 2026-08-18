from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = [
    "llama3.2:3b",
    "ministral-3:3b",
    "phi4-mini",
    "gemma3:4b",
    "nemotron-3-nano:4b",
]

OUTPUT_FILE = Path.home() / "ollama_benchmark_suite.xlsx"

MODEL_OPTIONS = {
    "temperature": 0,
    "num_ctx": 4096,
    "num_predict": 900,
}

SUMMARY_HEADERS = [
    "run_id",
    "timestamp",
    "model",
    "category",
    "task_name",
    "score",
    "prompt",
    "answer",
    "thinking",
    "scoring_details",
    "prompt_tokens",
    "generated_tokens",
    "prompt_tokens_per_second",
    "generation_tokens_per_second",
    "load_seconds",
    "prompt_seconds",
    "generation_seconds",
    "total_seconds",
    "status",
]


# ============================================================
# SOURCE PASSAGES
# ============================================================

SUMMARY_PASSAGE_1 = """
Northstar Computing announced that it will acquire LightPath Systems
for $500 million in cash. LightPath develops optical interconnect
technology used to connect servers inside AI data centres. Northstar
expects the acquisition to reduce networking costs and improve the
energy efficiency of its future AI clusters. The transaction is
expected to close in the fourth quarter of 2027, subject to regulatory
approval. LightPath will continue operating under its existing name
after the acquisition.
""".strip()

SUMMARY_PASSAGE_2 = """
The City of Brookdale approved a three-year electric bus pilot program.
The city will purchase 24 electric buses and install charging equipment
at two transit depots. The program will cost $38 million, with half of
the funding coming from a federal transportation grant. Service will
begin in March 2028 on the city's three busiest routes. Officials expect
the buses to reduce fuel costs and local air pollution. The city will
publish a performance review after the first 18 months of operation.
""".strip()

SUMMARY_PASSAGE_3 = """
Artemis Health reported results from a twelve-month remote-care trial
involving 1,200 adults with hypertension. Participants received
connected blood-pressure monitors and monthly virtual consultations.
The company reported that 68 percent of participants achieved their
target blood pressure by the end of the trial. Artemis said the program
reduced unnecessary clinic visits by 22 percent. The study did not
include a control group, and the company said a randomized follow-up
study is planned for 2028.
""".strip()


# ============================================================
# BENCHMARK TASKS
# ============================================================

TASKS = [
    # --------------------------------------------------------
    # TECHNICAL EXPLANATION
    # --------------------------------------------------------
    {
        "category": "technical",
        "task_name": "HBM versus DDR5",
        "scorer": "technical_hbm",
        "prompt": (
            "Explain why HBM gives AI accelerators higher memory "
            "bandwidth than DDR5. Include the roles of bus width, "
            "stacking, proximity to the GPU, and power efficiency. "
            "Keep the answer under 250 words and do not invent "
            "specifications."
        ),
    },
    {
        "category": "technical",
        "task_name": "Quantization and inference",
        "scorer": "technical_quantization",
        "prompt": (
            "Explain why quantizing a language model from 16-bit weights "
            "to 4-bit weights can reduce memory usage and sometimes "
            "increase inference speed. Include memory bandwidth, model "
            "quality, dequantization, and hardware support. Keep the "
            "answer under 220 words and do not invent benchmark numbers."
        ),
    },
    {
        "category": "technical",
        "task_name": "KV cache growth",
        "scorer": "technical_kv_cache",
        "prompt": (
            "Explain what the KV cache is during transformer inference "
            "and why its memory usage grows as context length increases. "
            "Include attention keys and values, token history, batch "
            "size, and model dimensions. Keep the answer under 220 words "
            "and avoid unsupported numerical specifications."
        ),
    },

    # --------------------------------------------------------
    # MATH
    # --------------------------------------------------------
    {
        "category": "math",
        "task_name": "Throughput calculation",
        "scorer": "math_throughput",
        "prompt": (
            "A system processes 2,400 prompt tokens in 30 seconds and "
            "generates 900 output tokens in 15 seconds. Return JSON only "
            "using exactly these keys: prompt_tokens_per_second, "
            "generation_tokens_per_second, overall_tokens_per_second. "
            "Overall throughput means all 3,300 tokens divided by the "
            "combined 45 seconds."
        ),
    },
    {
        "category": "math",
        "task_name": "Memory capacity calculation",
        "scorer": "math_memory",
        "prompt": (
            "A server contains 8 accelerators. Each accelerator has "
            "192 GB of memory. Return JSON only using exactly these keys: "
            "total_memory_gb, total_memory_tb_decimal, memory_after_25_"
            "percent_reserved_gb. Use 1 TB = 1000 GB."
        ),
    },
    {
        "category": "math",
        "task_name": "Speed improvement calculation",
        "scorer": "math_speedup",
        "prompt": (
            "Model A generates 28 tokens per second and Model B generates "
            "42 tokens per second. Return JSON only using exactly these "
            "keys: absolute_increase_tokens_per_second, percent_faster, "
            "speedup_factor. Round percent_faster to two decimal places."
        ),
    },

    # --------------------------------------------------------
    # CODING
    # --------------------------------------------------------
    {
        "category": "coding",
        "task_name": "Compress integer ranges",
        "scorer": "coding_compress_ranges",
        "function_name": "compress_ranges",
        "prompt": (
            "Write a Python function named compress_ranges(numbers). "
            "The input is a sorted list of unique integers. Return a "
            "string that compresses consecutive values into ranges. "
            "For example, [1, 2, 3, 5, 7, 8] becomes "
            "\"1-3,5,7-8\". An empty list returns an empty string. "
            "Return Python code only. Do not use imports, files, network "
            "access, eval, exec, or input."
        ),
    },
    {
        "category": "coding",
        "task_name": "Moving average",
        "scorer": "coding_moving_average",
        "function_name": "moving_average",
        "prompt": (
            "Write a Python function named moving_average(values, "
            "window_size). Return a list containing the arithmetic mean "
            "of every consecutive window. Return an empty list when "
            "window_size is larger than the input. Raise ValueError when "
            "window_size is less than 1. Return Python code only. Do not "
            "use imports, files, network access, eval, exec, or input."
        ),
    },
    {
        "category": "coding",
        "task_name": "Group records by field",
        "scorer": "coding_group_by",
        "function_name": "group_by",
        "prompt": (
            "Write a Python function named group_by(records, key). "
            "records is a list of dictionaries. Return a dictionary that "
            "maps each distinct value of records[i][key] to a list of "
            "the matching original dictionaries, preserving their input "
            "order. Raise KeyError naturally when a record lacks the "
            "key. Return Python code only. Do not use imports, files, "
            "network access, eval, exec, or input."
        ),
    },

    # --------------------------------------------------------
    # JSON EXTRACTION
    # --------------------------------------------------------
    {
        "category": "json",
        "task_name": "Accelerator extraction",
        "scorer": "json_accelerator",
        "prompt": (
            "Extract the information from the text below. Return valid "
            "JSON only, with exactly these keys: company, product, "
            "memory_gb, memory_type, launch_year, power_watts.\n\n"
            "Text: During its 2027 product briefing, AMD introduced the "
            "Orion X accelerator. The product includes 432 GB of HBM4 "
            "memory and has a rated power of 1,400 watts."
        ),
    },
    {
        "category": "json",
        "task_name": "Vehicle extraction",
        "scorer": "json_vehicle",
        "prompt": (
            "Extract the information from the text below. Return valid "
            "JSON only, with exactly these keys: manufacturer, model, "
            "model_year, range_km, battery_kwh, starting_price_cad.\n\n"
            "Text: Aurora Motors announced the 2029 Nova EV. The vehicle "
            "has a 92 kWh battery, an estimated driving range of 610 km, "
            "and a Canadian starting price of $58,400."
        ),
    },
    {
        "category": "json",
        "task_name": "Employment extraction",
        "scorer": "json_employment",
        "prompt": (
            "Extract the information from the text below. Return valid "
            "JSON only, with exactly these keys: employee, company, role, "
            "start_date, salary_cad, location.\n\n"
            "Text: Priya Shah will join Cedar Robotics as Director of "
            "Machine Learning on September 14, 2027. The position is "
            "based in Toronto and has an annual salary of CAD 185,000."
        ),
    },

    # --------------------------------------------------------
    # SUMMARIZATION
    # --------------------------------------------------------
    {
        "category": "summary",
        "task_name": "Acquisition summary",
        "scorer": "summary_acquisition",
        "prompt": (
            "Summarize the passage below in no more than 90 words. "
            "Include the purchase price, what LightPath develops, the "
            "expected benefits, expected closing period, regulatory "
            "condition, and what happens to the LightPath name. Use "
            "only information from the passage.\n\n"
            f"{SUMMARY_PASSAGE_1}"
        ),
    },
    {
        "category": "summary",
        "task_name": "Electric bus pilot summary",
        "scorer": "summary_bus",
        "prompt": (
            "Summarize the passage below in no more than 90 words. "
            "Include the number of buses, total cost, funding source, "
            "service start, expected benefits, and review schedule. "
            "Use only information from the passage.\n\n"
            f"{SUMMARY_PASSAGE_2}"
        ),
    },
    {
        "category": "summary",
        "task_name": "Remote-care trial summary",
        "scorer": "summary_health",
        "prompt": (
            "Summarize the passage below in no more than 90 words. "
            "Include the study duration, participant count, intervention, "
            "blood-pressure result, clinic-visit result, and major study "
            "limitation. Use only information from the passage.\n\n"
            f"{SUMMARY_PASSAGE_3}"
        ),
    },
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def nanoseconds_to_seconds(value: int | None) -> float:
    return (value or 0) / 1_000_000_000


def tokens_per_second(
    token_count: int | None,
    duration_nanoseconds: int | None,
) -> float:
    duration = nanoseconds_to_seconds(duration_nanoseconds)

    if not token_count or duration <= 0:
        return 0.0

    return token_count / duration


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_any(text: str, phrases: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(phrase.lower() in normalized for phrase in phrases)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    candidates = []

    if fenced_match:
        candidates.append(fenced_match.group(1))

    candidates.append(text)

    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def extract_python_code(text: str) -> str:
    fenced_match = re.search(
        r"```(?:python)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced_match:
        return fenced_match.group(1).strip()

    return text.strip()


# ============================================================
# TECHNICAL SCORING
# ============================================================

def score_technical_answer(
    answer: str,
    required_concepts: dict[str, list[str]],
    requested_topics: dict[str, list[str]],
    false_claims: dict[str, list[str]],
    word_limit: int,
) -> tuple[float, dict[str, Any]]:
    normalized = normalize_text(answer)

    concept_results = {
        name: contains_any(normalized, phrases)
        for name, phrases in required_concepts.items()
    }

    topic_results = {
        name: contains_any(normalized, phrases)
        for name, phrases in requested_topics.items()
    }

    detected_false_claims = [
        name
        for name, phrases in false_claims.items()
        if contains_any(normalized, phrases)
    ]

    concept_score = (
        sum(concept_results.values())
        / max(len(concept_results), 1)
    ) * 60

    topic_score = (
        sum(topic_results.values())
        / max(len(topic_results), 1)
    ) * 20

    length_score = 10 if word_count(answer) <= word_limit else 0
    no_numeric_specs_score = 10 if not re.search(r"\d", answer) else 0

    false_claim_penalty = min(
        len(detected_false_claims) * 10,
        30,
    )

    total = (
        concept_score
        + topic_score
        + length_score
        + no_numeric_specs_score
        - false_claim_penalty
    )

    total = round(max(0, min(100, total)), 2)

    return total, {
        "concepts": concept_results,
        "requested_topics": topic_results,
        "word_count": word_count(answer),
        "numeric_specs_present": bool(re.search(r"\d", answer)),
        "false_claims": detected_false_claims,
        "score": total,
    }


def score_hbm(answer: str) -> tuple[float, dict[str, Any]]:
    return score_technical_answer(
        answer=answer,
        required_concepts={
            "wide_interface": [
                "wide bus",
                "wider bus",
                "wide interface",
                "wider interface",
            ],
            "parallel_transfer": [
                "in parallel",
                "simultaneously",
                "parallel lanes",
                "parallel connections",
            ],
            "stacked_dram": [
                "stacked dram",
                "stacked dies",
                "memory dies are stacked",
                "3d stacked",
            ],
            "dense_packaging": [
                "through-silicon vias",
                "tsv",
                "multiple channels",
                "compact package",
            ],
            "close_placement": [
                "close to the gpu",
                "close to the accelerator",
                "same package",
                "shorter distance",
                "shorter trace",
            ],
            "energy_per_bit": [
                "energy per bit",
                "power per bit",
                "less energy",
                "lower energy",
                "power efficient",
                "power-efficient",
            ],
        },
        requested_topics={
            "bus_width": ["bus width", "wide bus", "wide interface"],
            "stacking": ["stack", "stacked"],
            "proximity": ["proximity", "close to", "same package"],
            "power": [
                "power efficiency",
                "power-efficient",
                "energy per bit",
            ],
        },
        false_claims={
            "one_bit_at_a_time": ["one bit at a time"],
            "pci_explanation": [
                "ddr5 through pcie",
                "eliminating pcie",
            ],
            "thinner_silicon": [
                "thinner silicon",
                "thinner die structure",
            ],
            "refresh_reduction": [
                "reduces the need for refresh",
                "fewer refresh cycles",
            ],
            "capacitor_materials": [
                "hafnium oxide",
                "capacitors sandwiched",
            ],
        },
        word_limit=250,
    )


def score_quantization(answer: str) -> tuple[float, dict[str, Any]]:
    return score_technical_answer(
        answer=answer,
        required_concepts={
            "fewer_bits": [
                "fewer bits",
                "4-bit",
                "lower precision",
                "reduced precision",
            ],
            "smaller_model": [
                "less memory",
                "smaller memory",
                "smaller model",
                "reduced memory",
            ],
            "bandwidth_reduction": [
                "memory bandwidth",
                "less data",
                "fewer bytes",
                "reduced bandwidth",
            ],
            "speed_effect": [
                "faster inference",
                "increase inference speed",
                "higher throughput",
            ],
            "quality_tradeoff": [
                "quality loss",
                "accuracy loss",
                "degradation",
                "precision trade-off",
                "precision tradeoff",
            ],
            "dequantization_or_kernels": [
                "dequantization",
                "dequantize",
                "quantized kernel",
                "specialized kernel",
            ],
            "hardware_support": [
                "hardware support",
                "supported hardware",
                "native 4-bit",
                "optimized hardware",
            ],
        },
        requested_topics={
            "memory_bandwidth": ["memory bandwidth"],
            "model_quality": ["quality", "accuracy"],
            "dequantization": ["dequantization", "dequantize"],
            "hardware": ["hardware support", "hardware"],
        },
        false_claims={
            "always_lossless": [
                "no quality loss",
                "always maintains quality",
            ],
            "always_faster": [
                "always faster",
                "guaranteed to be faster",
            ],
            "parameters_removed": [
                "removes parameters",
                "fewer parameters",
            ],
        },
        word_limit=220,
    )


def score_kv_cache(answer: str) -> tuple[float, dict[str, Any]]:
    return score_technical_answer(
        answer=answer,
        required_concepts={
            "stores_keys": [
                "stores keys",
                "cached keys",
                "key tensors",
            ],
            "stores_values": [
                "stores values",
                "cached values",
                "value tensors",
            ],
            "previous_tokens": [
                "previous tokens",
                "past tokens",
                "token history",
                "earlier tokens",
            ],
            "avoid_recompute": [
                "avoid recomputing",
                "does not recompute",
                "reuse",
                "reuses",
            ],
            "context_growth": [
                "grows with context",
                "longer context",
                "each new token",
                "context length increases",
            ],
            "batch_growth": [
                "batch size",
                "larger batches",
            ],
            "model_dimensions": [
                "layers",
                "attention heads",
                "head dimension",
                "model dimensions",
            ],
        },
        requested_topics={
            "keys_values": ["keys", "values"],
            "history": ["token history", "previous tokens", "past tokens"],
            "batch": ["batch size"],
            "dimensions": [
                "model dimensions",
                "layers",
                "attention heads",
            ],
        },
        false_claims={
            "stores_weights": [
                "stores model weights",
                "copy of the model weights",
            ],
            "constant_size": [
                "constant memory",
                "does not grow",
                "fixed size regardless of context",
            ],
            "training_only": [
                "only used during training",
            ],
        },
        word_limit=220,
    )


# ============================================================
# MATH SCORING
# ============================================================

def score_exact_json_math(
    answer: str,
    expected: dict[str, float],
    tolerance: float = 0.02,
) -> tuple[float, dict[str, Any]]:
    parsed = extract_json_object(answer)

    if parsed is None:
        return 0.0, {
            "valid_json": False,
            "score": 0,
        }

    score = 20
    exact_keys = set(parsed.keys()) == set(expected.keys())

    if exact_keys:
        score += 20

    field_results = {}
    points_per_field = 60 / len(expected)

    for key, expected_value in expected.items():
        actual = parsed.get(key)

        try:
            actual_number = float(actual)
            correct = abs(actual_number - expected_value) <= tolerance
        except (TypeError, ValueError):
            correct = False

        field_results[key] = {
            "actual": actual,
            "expected": expected_value,
            "correct": correct,
        }

        if correct:
            score += points_per_field

    score = round(min(100, score), 2)

    return score, {
        "valid_json": True,
        "exact_keys": exact_keys,
        "fields": field_results,
        "score": score,
    }


# ============================================================
# JSON EXTRACTION SCORING
# ============================================================

def score_exact_json_extraction(
    answer: str,
    expected: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    parsed = extract_json_object(answer)

    if parsed is None:
        return 0.0, {
            "valid_json": False,
            "score": 0,
        }

    score = 20
    exact_schema = set(parsed.keys()) == set(expected.keys())

    if exact_schema:
        score += 20

    points_per_value = 50 / len(expected)
    value_results = {}

    for key, expected_value in expected.items():
        actual = parsed.get(key)

        if isinstance(expected_value, str):
            correct = (
                str(actual).strip().lower()
                == expected_value.lower()
            )
        else:
            try:
                correct = float(actual) == float(expected_value)
            except (TypeError, ValueError):
                correct = False

        value_results[key] = {
            "actual": actual,
            "expected": expected_value,
            "correct": correct,
        }

        if correct:
            score += points_per_value

    unsupported_keys = sorted(
        set(parsed.keys()) - set(expected.keys())
    )

    if not unsupported_keys:
        score += 10

    score = round(min(100, score), 2)

    return score, {
        "valid_json": True,
        "exact_schema": exact_schema,
        "unsupported_keys": unsupported_keys,
        "fields": value_results,
        "score": score,
    }


# ============================================================
# SUMMARY SCORING
# ============================================================

def score_summary(
    answer: str,
    key_points: dict[str, list[str]],
    required_entities: list[str],
    unsupported_claims: dict[str, list[str]],
    word_limit: int = 90,
) -> tuple[float, dict[str, Any]]:
    normalized = normalize_text(answer)

    point_results = {
        name: contains_any(normalized, phrases)
        for name, phrases in key_points.items()
    }

    coverage_score = (
        sum(point_results.values()) / len(point_results)
    ) * 60

    detected_unsupported = [
        name
        for name, phrases in unsupported_claims.items()
        if contains_any(normalized, phrases)
    ]

    faithfulness_score = max(
        0,
        20 - len(detected_unsupported) * 10,
    )

    length_score = 10 if word_count(answer) <= word_limit else 0

    entity_score = 10 if all(
        entity.lower() in normalized
        for entity in required_entities
    ) else 0

    total = round(
        max(
            0,
            min(
                100,
                coverage_score
                + faithfulness_score
                + length_score
                + entity_score,
            ),
        ),
        2,
    )

    return total, {
        "key_points": point_results,
        "unsupported_claims": detected_unsupported,
        "word_count": word_count(answer),
        "score": total,
    }


# ============================================================
# CODE SAFETY AND EXECUTION
# ============================================================

FORBIDDEN_AST_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
}

FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
)


def inspect_code_safety(
    code: str,
    function_name: str,
) -> tuple[bool, list[str], ast.AST | None]:
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, [f"syntax_error: {error}"], None

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_AST_NODES):
            violations.append(type(node).__name__)

        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_AST_CALLS
            ):
                violations.append(
                    f"forbidden_call:{node.func.id}"
                )

        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
        ):
            violations.append(
                f"dunder_attribute:{node.attr}"
            )

    function_found = any(
        isinstance(node, ast.FunctionDef)
        and node.name == function_name
        for node in ast.walk(tree)
    )

    if not function_found:
        violations.append(
            f"missing_function:{function_name}"
        )

    return len(violations) == 0, violations, tree


def execute_generated_code(
    answer: str,
    function_name: str,
    tests: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    code = extract_python_code(answer)

    if not code:
        return 0.0, {
            "code_extracted": False,
            "score": 0,
        }

    score = 10

    safe, violations, tree = inspect_code_safety(
        code,
        function_name,
    )

    syntax_valid = tree is not None

    if syntax_valid:
        score += 10

    if not safe:
        return float(score), {
            "code_extracted": True,
            "syntax_valid": syntax_valid,
            "safe": False,
            "violations": violations,
            "tests_passed": 0,
            "tests_total": len(tests),
            "score": score,
        }

    score += 5

    test_data = json.dumps(tests)

    test_harness = f"""
import json

tests = json.loads({test_data!r})
passed = 0
results = []

for test in tests:
    args = test["args"]
    expected = test.get("expected")
    expected_exception = test.get("exception")

    try:
        actual = {function_name}(*args)

        if expected_exception:
            correct = False
        else:
            correct = actual == expected

        error_name = None

    except Exception as error:
        actual = None
        error_name = type(error).__name__
        correct = error_name == expected_exception

    if correct:
        passed += 1

    results.append({{
        "args": args,
        "expected": expected,
        "expected_exception": expected_exception,
        "actual": actual,
        "actual_exception": error_name,
        "correct": correct,
    }})

print(json.dumps({{
    "passed": passed,
    "total": len(tests),
    "results": results,
}}))
"""

    combined_code = code + "\n\n" + test_harness

    try:
        with tempfile.TemporaryDirectory() as temp_directory:
            script_path = (
                Path(temp_directory)
                / "generated_solution.py"
            )

            script_path.write_text(
                combined_code,
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                text=True,
                timeout=4,
                cwd=temp_directory,
                env={"PATH": os.environ.get("PATH", "")},
                check=False,
            )

        output_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.strip()
        ]

        if not output_lines:
            raise ValueError(
                "Generated program produced no test output."
            )

        test_result = json.loads(output_lines[-1])

        tests_passed = int(test_result["passed"])
        tests_total = int(test_result["total"])

        score += (tests_passed / tests_total) * 70
        score += 5
        score = round(min(100, score), 2)

        return score, {
            "code_extracted": True,
            "syntax_valid": True,
            "safe": True,
            "violations": [],
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "test_results": test_result["results"],
            "stderr": completed.stderr,
            "completed_before_timeout": True,
            "score": score,
        }

    except subprocess.TimeoutExpired:
        return float(score), {
            "code_extracted": True,
            "syntax_valid": True,
            "safe": True,
            "tests_passed": 0,
            "tests_total": len(tests),
            "completed_before_timeout": False,
            "error": "timeout",
            "score": score,
        }

    except Exception as error:
        return float(score), {
            "code_extracted": True,
            "syntax_valid": True,
            "safe": True,
            "tests_passed": 0,
            "tests_total": len(tests),
            "completed_before_timeout": True,
            "error": str(error),
            "score": score,
        }


# ============================================================
# SCORE ROUTER
# ============================================================

def score_answer(
    task: dict[str, Any],
    answer: str,
) -> tuple[float, dict[str, Any]]:
    scorer = task["scorer"]

    if scorer == "technical_hbm":
        return score_hbm(answer)

    if scorer == "technical_quantization":
        return score_quantization(answer)

    if scorer == "technical_kv_cache":
        return score_kv_cache(answer)

    if scorer == "math_throughput":
        return score_exact_json_math(
            answer,
            {
                "prompt_tokens_per_second": 80,
                "generation_tokens_per_second": 60,
                "overall_tokens_per_second": 73.3333333333,
            },
        )

    if scorer == "math_memory":
        return score_exact_json_math(
            answer,
            {
                "total_memory_gb": 1536,
                "total_memory_tb_decimal": 1.536,
                "memory_after_25_percent_reserved_gb": 1152,
            },
        )

    if scorer == "math_speedup":
        return score_exact_json_math(
            answer,
            {
                "absolute_increase_tokens_per_second": 14,
                "percent_faster": 50,
                "speedup_factor": 1.5,
            },
        )

    if scorer == "json_accelerator":
        return score_exact_json_extraction(
            answer,
            {
                "company": "AMD",
                "product": "Orion X",
                "memory_gb": 432,
                "memory_type": "HBM4",
                "launch_year": 2027,
                "power_watts": 1400,
            },
        )

    if scorer == "json_vehicle":
        return score_exact_json_extraction(
            answer,
            {
                "manufacturer": "Aurora Motors",
                "model": "Nova EV",
                "model_year": 2029,
                "range_km": 610,
                "battery_kwh": 92,
                "starting_price_cad": 58400,
            },
        )

    if scorer == "json_employment":
        return score_exact_json_extraction(
            answer,
            {
                "employee": "Priya Shah",
                "company": "Cedar Robotics",
                "role": "Director of Machine Learning",
                "start_date": "September 14, 2027",
                "salary_cad": 185000,
                "location": "Toronto",
            },
        )

    if scorer == "coding_compress_ranges":
        return execute_generated_code(
            answer,
            "compress_ranges",
            [
                {"args": [[]], "expected": ""},
                {"args": [[4]], "expected": "4"},
                {"args": [[1, 2, 3]], "expected": "1-3"},
                {"args": [[1, 3, 5]], "expected": "1,3,5"},
                {
                    "args": [[1, 2, 3, 5, 7, 8]],
                    "expected": "1-3,5,7-8",
                },
                {
                    "args": [[-3, -2, -1, 2]],
                    "expected": "-3--1,2",
                },
                {
                    "args": [[0, 1, 2, 4, 5, 9]],
                    "expected": "0-2,4-5,9",
                },
                {
                    "args": [[10, 11, 13, 14, 15, 20]],
                    "expected": "10-11,13-15,20",
                },
            ],
        )

    if scorer == "coding_moving_average":
        return execute_generated_code(
            answer,
            "moving_average",
            [
                {"args": [[1, 2, 3], 1], "expected": [1.0, 2.0, 3.0]},
                {"args": [[1, 2, 3], 2], "expected": [1.5, 2.5]},
                {"args": [[1, 2, 3, 4], 3], "expected": [2.0, 3.0]},
                {"args": [[], 1], "expected": []},
                {"args": [[1, 2], 3], "expected": []},
                {
                    "args": [[5, 5, 5], 2],
                    "expected": [5.0, 5.0],
                },
                {
                    "args": [[1, 2, 3], 0],
                    "exception": "ValueError",
                },
                {
                    "args": [[1, 2, 3], -1],
                    "exception": "ValueError",
                },
            ],
        )

    if scorer == "coding_group_by":
        return execute_generated_code(
            answer,
            "group_by",
            [
                {"args": [[], "team"], "expected": {}},
                {
                    "args": [
                        [
                            {"name": "A", "team": "red"},
                            {"name": "B", "team": "blue"},
                            {"name": "C", "team": "red"},
                        ],
                        "team",
                    ],
                    "expected": {
                        "red": [
                            {"name": "A", "team": "red"},
                            {"name": "C", "team": "red"},
                        ],
                        "blue": [
                            {"name": "B", "team": "blue"},
                        ],
                    },
                },
                {
                    "args": [
                        [{"value": 1}, {"value": 1}],
                        "value",
                    ],
                    "expected": {
                        "1": None
                    },
                },
                {
                    "args": [
                        [{"name": "A"}],
                        "team",
                    ],
                    "exception": "KeyError",
                },
            ],
        )

    if scorer == "summary_acquisition":
        return score_summary(
            answer,
            key_points={
                "price": ["$500 million", "500 million"],
                "technology": ["optical interconnect"],
                "cost_benefit": ["reduce networking costs"],
                "energy_benefit": ["energy efficiency"],
                "closing": [
                    "fourth quarter of 2027",
                    "q4 2027",
                ],
                "regulatory": ["regulatory approval"],
                "name": [
                    "existing name",
                    "retain its name",
                    "keep its name",
                ],
            },
            required_entities=["Northstar", "LightPath"],
            unsupported_claims={
                "stock_payment": ["paid in shares"],
                "completed": ["already completed"],
                "renaming": ["will be renamed"],
            },
        )

    if scorer == "summary_bus":
        return score_summary(
            answer,
            key_points={
                "bus_count": ["24 electric buses", "24 buses"],
                "cost": ["$38 million", "38 million"],
                "funding": ["federal transportation grant"],
                "start": ["march 2028"],
                "fuel": ["reduce fuel costs"],
                "pollution": ["air pollution"],
                "review": ["18 months"],
            },
            required_entities=["Brookdale"],
            unsupported_claims={
                "all_routes": ["all city routes"],
                "full_federal_funding": [
                    "fully funded by the federal government",
                ],
                "completed": ["already completed"],
            },
        )

    if scorer == "summary_health":
        return score_summary(
            answer,
            key_points={
                "duration": ["twelve-month", "12-month"],
                "participants": ["1,200"],
                "monitor": ["blood-pressure monitors"],
                "virtual": ["virtual consultations"],
                "target": ["68 percent", "68%"],
                "visits": ["22 percent", "22%"],
                "limitation": ["no control group"],
            },
            required_entities=["Artemis"],
            unsupported_claims={
                "randomized": [
                    "the trial was randomized",
                    "randomized controlled trial",
                ],
                "causation": [
                    "proved the program caused",
                ],
            },
        )

    return 0.0, {
        "error": f"Unknown scorer: {scorer}",
        "score": 0,
    }


# ============================================================
# MODEL EXECUTION
# ============================================================

def run_task(
    model: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print(f"Model: {model}")
    print(f"Task:  {task['task_name']}")
    print("=" * 72)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": task["prompt"],
                }
            ],
            options=MODEL_OPTIONS,
            stream=False,
        )

        message = response.get("message") or {}

        answer = message.get("content") or ""
        thinking = message.get("thinking") or ""

        prompt_tokens = response.get(
            "prompt_eval_count"
        ) or 0

        generated_tokens = response.get(
            "eval_count"
        ) or 0

        prompt_duration = response.get(
            "prompt_eval_duration"
        ) or 0

        generation_duration = response.get(
            "eval_duration"
        ) or 0

        load_duration = response.get(
            "load_duration"
        ) or 0

        total_duration = response.get(
            "total_duration"
        ) or 0

        prompt_speed = tokens_per_second(
            prompt_tokens,
            prompt_duration,
        )

        generation_speed = tokens_per_second(
            generated_tokens,
            generation_duration,
        )

        score, scoring_details = score_answer(
            task,
            answer,
        )

        print("\nAnswer:")
        print(answer if answer else "[No answer produced]")

        print("\nResult:")
        print(f"Quality score:    {score:.2f}/100")
        print(f"Generation speed: {generation_speed:.2f} tok/s")
        print(f"Generated tokens: {generated_tokens}")
        print(
            f"Total time:       "
            f"{nanoseconds_to_seconds(total_duration):.2f}s"
        )

        return {
            "model": model,
            "category": task["category"],
            "task_name": task["task_name"],
            "score": score,
            "prompt": task["prompt"],
            "answer": answer,
            "thinking": thinking,
            "scoring_details": json.dumps(
                scoring_details,
                ensure_ascii=False,
                indent=2,
            ),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "prompt_tokens_per_second": round(
                prompt_speed,
                2,
            ),
            "generation_tokens_per_second": round(
                generation_speed,
                2,
            ),
            "load_seconds": round(
                nanoseconds_to_seconds(load_duration),
                3,
            ),
            "prompt_seconds": round(
                nanoseconds_to_seconds(prompt_duration),
                3,
            ),
            "generation_seconds": round(
                nanoseconds_to_seconds(
                    generation_duration
                ),
                3,
            ),
            "total_seconds": round(
                nanoseconds_to_seconds(total_duration),
                3,
            ),
            "status": "success",
        }

    except Exception as error:
        print(f"\nError: {error}")

        return {
            "model": model,
            "category": task["category"],
            "task_name": task["task_name"],
            "score": 0,
            "prompt": task["prompt"],
            "answer": "",
            "thinking": "",
            "scoring_details": json.dumps(
                {"error": str(error)},
                indent=2,
            ),
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "prompt_tokens_per_second": 0,
            "generation_tokens_per_second": 0,
            "load_seconds": 0,
            "prompt_seconds": 0,
            "generation_seconds": 0,
            "total_seconds": 0,
            "status": f"error: {error}",
        }


# ============================================================
# EXCEL
# ============================================================

def style_header(sheet) -> None:
    header_fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78",
    )

    for cell in sheet[1]:
        cell.font = Font(
            bold=True,
            color="FFFFFF",
        )
        cell.fill = header_fill
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    sheet.row_dimensions[1].height = 35


def format_sheet(sheet) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    widths = {
        "run_id": 12,
        "timestamp": 20,
        "model": 25,
        "category": 16,
        "task_name": 30,
        "score": 12,
        "prompt": 48,
        "answer": 58,
        "thinking": 50,
        "scoring_details": 55,
        "prompt_tokens": 16,
        "generated_tokens": 18,
        "prompt_tokens_per_second": 24,
        "generation_tokens_per_second": 28,
        "load_seconds": 15,
        "prompt_seconds": 16,
        "generation_seconds": 20,
        "total_seconds": 15,
        "status": 18,
    }

    header_columns = {
        cell.value: cell.column
        for cell in sheet[1]
    }

    for header, column_number in header_columns.items():
        column_letter = get_column_letter(column_number)
        sheet.column_dimensions[column_letter].width = (
            widths.get(header, 16)
        )

    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

    style_header(sheet)


def load_or_create_workbook():
    if OUTPUT_FILE.exists():
        workbook = load_workbook(OUTPUT_FILE)
    else:
        workbook = Workbook()
        workbook.active.title = "Summary"
        workbook["Summary"].append(SUMMARY_HEADERS)

    if "Summary" not in workbook.sheetnames:
        summary = workbook.create_sheet("Summary", 0)
        summary.append(SUMMARY_HEADERS)

    if "Leaderboard" not in workbook.sheetnames:
        leaderboard = workbook.create_sheet(
            "Leaderboard",
            1,
        )
        leaderboard.append(
            [
                "run_id",
                "model",
                "average_quality_score",
                "technical_score",
                "math_score",
                "coding_score",
                "json_score",
                "summary_score",
                "average_generation_tok_s",
                "total_time_seconds",
            ]
        )

    return workbook


def next_run_id(workbook) -> str:
    run_numbers = []

    for sheet_name in workbook.sheetnames:
        if sheet_name.startswith("Run_"):
            number_text = sheet_name.replace("Run_", "")

            if number_text.isdigit():
                run_numbers.append(int(number_text))

    return f"Run_{max(run_numbers, default=0) + 1:03d}"


def build_leaderboard_rows(
    run_id: str,
    results: list[dict[str, Any]],
) -> list[list[Any]]:
    rows = []

    for model in MODELS:
        model_results = [
            result
            for result in results
            if result["model"] == model
        ]

        category_scores = defaultdict(list)

        for result in model_results:
            category_scores[result["category"]].append(
                result["score"]
            )

        averaged_categories = {
            category: (
                sum(scores) / len(scores)
                if scores else 0
            )
            for category, scores in category_scores.items()
        }

        average_score = sum(
            result["score"]
            for result in model_results
        ) / len(model_results)

        average_speed = sum(
            result["generation_tokens_per_second"]
            for result in model_results
        ) / len(model_results)

        total_time = sum(
            result["total_seconds"]
            for result in model_results
        )

        rows.append(
            [
                run_id,
                model,
                round(average_score, 2),
                round(
                    averaged_categories.get("technical", 0),
                    2,
                ),
                round(
                    averaged_categories.get("math", 0),
                    2,
                ),
                round(
                    averaged_categories.get("coding", 0),
                    2,
                ),
                round(
                    averaged_categories.get("json", 0),
                    2,
                ),
                round(
                    averaged_categories.get("summary", 0),
                    2,
                ),
                round(average_speed, 2),
                round(total_time, 2),
            ]
        )

    rows.sort(
        key=lambda row: row[2],
        reverse=True,
    )

    return rows


def save_results(results: list[dict[str, Any]]) -> str:
    workbook = load_or_create_workbook()

    summary_sheet = workbook["Summary"]
    leaderboard_sheet = workbook["Leaderboard"]

    run_id = next_run_id(workbook)
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    run_sheet = workbook.create_sheet(run_id)
    run_sheet.append(SUMMARY_HEADERS)

    for result in results:
        row = [
            run_id,
            timestamp,
            result["model"],
            result["category"],
            result["task_name"],
            result["score"],
            result["prompt"],
            result["answer"],
            result["thinking"],
            result["scoring_details"],
            result["prompt_tokens"],
            result["generated_tokens"],
            result["prompt_tokens_per_second"],
            result["generation_tokens_per_second"],
            result["load_seconds"],
            result["prompt_seconds"],
            result["generation_seconds"],
            result["total_seconds"],
            result["status"],
        ]

        summary_sheet.append(row)
        run_sheet.append(row)

    for row in build_leaderboard_rows(
        run_id,
        results,
    ):
        leaderboard_sheet.append(row)

    format_sheet(summary_sheet)
    format_sheet(run_sheet)
    format_sheet(leaderboard_sheet)

    workbook.save(OUTPUT_FILE)

    return run_id


# ============================================================
# CONSOLE LEADERBOARD
# ============================================================

def print_run_leaderboard(
    results: list[dict[str, Any]],
) -> None:
    rows = build_leaderboard_rows(
        "current",
        results,
    )

    print("\n" + "=" * 72)
    print("QUALITY LEADERBOARD")
    print("=" * 72)

    for position, row in enumerate(rows, start=1):
        print(
            f"{position}. {row[1]}: "
            f"{row[2]:.2f}/100 quality, "
            f"{row[8]:.2f} tok/s average"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Local Ollama Fifteen-Task Benchmark")
    print(f"Models: {len(MODELS)}")
    print(f"Tasks per model: {len(TASKS)}")
    print(
        f"Total model responses: "
        f"{len(MODELS) * len(TASKS)}"
    )

    results = []

    for model in MODELS:
        for task in TASKS:
            results.append(
                run_task(model, task)
            )

    try:
        run_id = save_results(results)
    except PermissionError:
        print(
            "\nThe Excel workbook is currently open."
        )
        print(
            "Close Microsoft Excel completely, "
            "then run the benchmark again."
        )
        return

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)
    print(f"Workbook saved to:\n{OUTPUT_FILE}")
    print(f"New run sheet: {run_id}")

    print_run_leaderboard(results)


if __name__ == "__main__":
    main()