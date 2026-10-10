# ==================================================
# CHAPTER 18 — CONTEXT-AWARE CONVERSATION MEMORY
# ==================================================
#
# Features:
#   - Bounded conversation history
#   - Follow-up detection
#   - Contextual query rewriting
#   - Repository evidence-intent detection
#   - Routing recommendations
#   - Session-only storage
#
# Design:
#   Jev remains the primary router.
#
#   A narrow application-level safeguard can
#   override a prediction when a follow-up clearly
#   requests an existing repository measurement.
#
#   Causal/analytical questions are NOT forced
#   into RAG.
#
# ==================================================

import re

from collections import deque
from dataclasses import dataclass


FOLLOWUP_STARTS = (
    "and ",
    "also ",
    "but ",
    "what about",
    "how about",
    "what if",
    "what was",
    "what were",
    "what did",
    "what happened",
    "how much",
    "how many",
    "why ",
    "why?",
    "can you elaborate",
    "explain more",
    "tell me more",
    "go deeper",
    "compared to",
    "which one",
)


REFERENCE_PATTERN = re.compile(
    r"\b("
    r"it|its|that|those|these|them|"
    r"which one|the same|earlier|"
    r"previous|previously"
    r")\b",
    re.IGNORECASE,
)


EVIDENCE_PATTERN = re.compile(
    r"\b("
    r"what\s+(?:was|were|did|happened)|"
    r"how\s+(?:much|many)|"
    r"measured|observed|recorded|reported|"
    r"found|achieved|reached|"
    r"results?|numbers?|values?|"
    r"remind|recall|show me"
    r")\b",
    re.IGNORECASE,
)


ANALYSIS_PATTERN = re.compile(
    r"\b("
    r"why|how come|what caused|what causes|"
    r"explain|analyze|analyse|"
    r"bottleneck|root cause|"
    r"optimize|optimise|predict|"
    r"hypothesis|hypothesize|"
    r"what if|what would|"
    r"how would|how could|should"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class Turn:
    question: str
    answer: str
    route: str
    image_name: str | None = None


@dataclass
class QueryResolution:
    original_query: str
    rewritten_query: str
    retrieval_query: str

    is_followup: bool

    preferred_route: str | None
    policy_reason: str | None

    previous_question: str | None


class ConversationMemory:

    def __init__(
        self,
        max_turns=4,
        max_answer_chars=1200,
        max_question_chars=500,
    ):

        self.turns = deque(
            maxlen=max_turns
        )

        self.max_turns = max_turns

        self.max_answer_chars = max_answer_chars
        self.max_question_chars = max_question_chars

    # ----------------------------------------------
    # FOLLOW-UP DETECTION
    # ----------------------------------------------

    def is_followup(self, query):

        if not self.turns:
            return False

        text = query.strip().lower()

        if not text or len(text) > 180:
            return False

        if text.startswith(FOLLOWUP_STARTS):
            return True

        return bool(
            REFERENCE_PATTERN.search(text)
        )

    # ----------------------------------------------
    # CLASSIFY FOLLOW-UP INTENT
    # ----------------------------------------------

    def requests_existing_evidence(self, query):

        # Asking WHY something happened is
        # analysis, not simply measurement recall.

        if ANALYSIS_PATTERN.search(query):
            return False

        # Asking WHAT was measured, observed,
        # reported, etc. is evidence retrieval.

        return bool(
            EVIDENCE_PATTERN.search(query)
        )

    # ----------------------------------------------
    # CONTEXTUAL QUERY RESOLUTION
    # ----------------------------------------------

    def resolve(self, query):

        query = query.strip()

        if not self.is_followup(query):

            return QueryResolution(
                original_query=query,
                rewritten_query=query,
                retrieval_query=query,
                is_followup=False,
                preferred_route=None,
                policy_reason=None,
                previous_question=None,
            )

        previous = self.turns[-1]

        # Provide a standalone description of
        # the current request and its topic.
        #
        # We do not insert previous assistant
        # answers into the routing query.

        rewritten = (
            "Current user request in the context "
            "of an ongoing conversation.\n"
            f"Previous topic: {previous.question}\n"
            f"Current request: {query}"
        )

        retrieval_query = (
            f"{previous.question}\n"
            f"Follow-up question: {query}"
        )

        preferred_route = None
        policy_reason = None

        # Narrow application-level policy:
        #
        # 1. The immediate previous route was RAG.
        # 2. This is a conversational follow-up.
        # 3. The user asks for an existing result.
        # 4. The user is NOT asking for causal
        #    analysis or a hypothetical.
        #
        # No chapter number, metric, or specific
        # experimental result is hardcoded.

        if (
            previous.route == "RAG"
            and self.requests_existing_evidence(
                query
            )
        ):

            preferred_route = "RAG"

            policy_reason = (
                "FOLLOWUP_REPOSITORY_EVIDENCE"
            )

            rewritten = (
                "Find previously recorded evidence "
                "in the user's local-ai-learning "
                "repository.\n"
                f"Previous topic: {previous.question}\n"
                f"Current retrieval request: {query}\n"
                "The user is asking for an existing "
                "result, not for a new causal analysis."
            )

        return QueryResolution(
            original_query=query,
            rewritten_query=rewritten,
            retrieval_query=retrieval_query,
            is_followup=True,
            preferred_route=preferred_route,
            policy_reason=policy_reason,
            previous_question=previous.question,
        )

    # ----------------------------------------------
    # SAVE A COMPLETED EXCHANGE
    # ----------------------------------------------

    def remember(
        self,
        question,
        answer,
        route,
        image_name=None,
    ):

        self.turns.append(
            Turn(
                question=str(question).strip()[
                    :self.max_question_chars
                ],
                answer=str(answer).strip()[
                    :self.max_answer_chars
                ],
                route=route,
                image_name=image_name,
            )
        )

    # ----------------------------------------------
    # GENERATOR HISTORY
    # ----------------------------------------------

    def history_messages(self):

        messages = []

        for turn in self.turns:

            content = turn.question

            if turn.image_name:

                content += (
                    "\n[An image was attached "
                    "previously, but the image "
                    "is not retained in memory.]"
                )

            messages.append({
                "role": "user",
                "content": content,
            })

            messages.append({
                "role": "assistant",
                "content": turn.answer,
            })

        return messages

    # ----------------------------------------------
    # MEMORY MANAGEMENT
    # ----------------------------------------------

    def count(self):
        return len(self.turns)

    def clear(self):
        self.turns.clear()

    def show(self):

        print()
        print("=" * 65)
        print("CONVERSATION MEMORY")
        print("=" * 65)

        print(
            f"Remembered exchanges: "
            f"{self.count()}/{self.max_turns}"
        )

        if not self.turns:

            print("Memory is empty.")
            return

        for index, turn in enumerate(
            self.turns,
            start=1,
        ):

            print()
            print(
                f"[{index}] Route: {turn.route}"
            )

            print(
                f"Question: {turn.question}"
            )

            print(
                f"Answer preview: "
                f"{turn.answer[:160]}..."
            )


# --------------------------------------------------
# REGRESSION TESTS
# --------------------------------------------------

def main():

    tests_passed = 0

    memory = ConversationMemory(
        max_turns=2
    )

    # TEST 1 — Empty memory

    result = memory.resolve(
        "Explain the roofline model."
    )

    assert not result.is_followup
    assert result.preferred_route is None

    tests_passed += 1

    # Simulate a completed repository question.

    memory.remember(
        question=(
            "What did I learn about multimodal "
            "inference in Chapter 17?"
        ),
        answer=(
            "I benchmarked multimodal serving "
            "at different concurrency levels."
        ),
        route="RAG",
    )

    # TEST 2 — Reproduce our real failure.

    query = (
        "And what happened to throughput "
        "at concurrency 8?"
    )

    result = memory.resolve(query)

    assert result.is_followup
    assert result.preferred_route == "RAG"
    assert result.policy_reason == (
        "FOLLOWUP_REPOSITORY_EVIDENCE"
    )

    assert "Chapter 17" in (
        result.rewritten_query
    )

    assert "concurrency 8" in (
        result.retrieval_query
    )

    tests_passed += 1

    # TEST 3 — Causal question should NOT
    # automatically inherit RAG.

    result = memory.resolve(
        "Why did throughput decrease "
        "at concurrency 8?"
    )

    assert result.is_followup
    assert result.preferred_route is None

    tests_passed += 1

    # TEST 4 — Hypothetical, not evidence recall.

    result = memory.resolve(
        "What if concurrency increases to 16?"
    )

    assert result.is_followup
    assert result.preferred_route is None

    tests_passed += 1

    # TEST 5 — No automatic route inheritance
    # for an unrelated standalone question.

    result = memory.resolve(
        "Calculate KV cache memory for "
        "32 layers, 8 KV heads, "
        "head dimension of 128, "
        "context length 8192."
    )

    assert not result.is_followup
    assert result.preferred_route is None

    tests_passed += 1

    # TEST 6 — History message construction.

    assert len(
        memory.history_messages()
    ) == 2

    tests_passed += 1

    # TEST 7 — Bounded capacity.

    memory.remember(
        "Second question",
        "Second answer",
        "CHAT",
    )

    memory.remember(
        "Third question",
        "Third answer",
        "CHAT",
    )

    assert memory.count() == 2

    tests_passed += 1

    # TEST 8 — Clear memory.

    memory.clear()

    assert memory.count() == 0

    tests_passed += 1

    print()
    print("=" * 65)
    print("CONTEXTUAL MEMORY REGRESSION TESTS")
    print("=" * 65)

    print(
        f"Tests passed: {tests_passed}/8"
    )


if __name__ == "__main__":
    main()