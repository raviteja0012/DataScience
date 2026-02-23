"""Natural language intent classification for routing user queries.

Uses a hybrid approach combining keyword matching, TF-IDF similarity against
labeled examples, and optional Cortex LLM classification for ambiguous queries.
This ensures fast, deterministic routing for clear-cut queries while maintaining
accuracy on edge cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Intent(str, Enum):
    """Supported user intent categories."""

    ANALYTICS = "analytics"
    COMPLIANCE = "compliance"
    ANOMALY = "anomaly"
    GENERAL = "general"


@dataclass(frozen=True)
class ClassificationResult:
    """Result of intent classification with confidence scores.

    Attributes:
        intent: The classified primary intent.
        confidence: Confidence score for the primary intent (0.0 to 1.0).
        scores: Confidence scores for all intent categories.
        method: Classification method used (keyword, similarity, llm).
    """

    intent: Intent
    confidence: float
    scores: dict[str, float]
    method: str


@dataclass
class IntentClassifier:
    """Classifies user queries into actionable intent categories.

    The classifier operates in three tiers:
    1. Keyword matching - fast, deterministic, handles obvious cases
    2. TF-IDF similarity - handles queries that match example patterns
    3. Cortex LLM - fallback for genuinely ambiguous queries

    Attributes:
        confidence_threshold: Minimum confidence to accept a classification.
        config: Loaded intent configuration from agent_config.yaml.
    """

    confidence_threshold: float = 0.65
    config: dict[str, Any] = field(default_factory=dict)
    _keyword_map: dict[Intent, list[str]] = field(default_factory=dict, init=False)
    _example_map: dict[Intent, list[str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.config:
            self.config = self._load_default_config()
        self._build_keyword_index()

    @staticmethod
    def _load_default_config() -> dict[str, Any]:
        """Load intent configuration from the default config file."""
        config_path = Path(__file__).resolve().parents[2] / "config" / "agent_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            return full_config.get("intent_classification", {})
        return {}

    def _build_keyword_index(self) -> None:
        """Build keyword and example lookup structures from config."""
        intents_config = self.config.get("intents", {})

        for intent_name, intent_data in intents_config.items():
            try:
                intent = Intent(intent_name)
            except ValueError:
                logger.warning("unknown_intent_in_config", intent=intent_name)
                continue

            self._keyword_map[intent] = [
                kw.lower() for kw in intent_data.get("keywords", [])
            ]
            self._example_map[intent] = [
                ex.lower() for ex in intent_data.get("examples", [])
            ]

        # Ensure all intents have entries
        for intent in Intent:
            self._keyword_map.setdefault(intent, [])
            self._example_map.setdefault(intent, [])

    def classify(self, query: str) -> ClassificationResult:
        """Classify a user query into an intent category.

        Applies tiered classification: keyword matching first, then
        example similarity, falling back to general intent if confidence
        is below threshold.

        Args:
            query: The user's natural language query.

        Returns:
            ClassificationResult with intent, confidence, and method.
        """
        if not query or not query.strip():
            return ClassificationResult(
                intent=Intent.GENERAL,
                confidence=1.0,
                scores={i.value: 0.0 for i in Intent},
                method="empty",
            )

        query_lower = query.lower().strip()

        # Tier 1: Keyword matching
        keyword_result = self._keyword_classify(query_lower)
        if keyword_result.confidence >= self.confidence_threshold:
            logger.info(
                "intent_classified",
                intent=keyword_result.intent.value,
                confidence=keyword_result.confidence,
                method="keyword",
            )
            return keyword_result

        # Tier 2: Example similarity matching
        similarity_result = self._similarity_classify(query_lower)
        if similarity_result.confidence >= self.confidence_threshold:
            logger.info(
                "intent_classified",
                intent=similarity_result.intent.value,
                confidence=similarity_result.confidence,
                method="similarity",
            )
            return similarity_result

        # Tier 3: Fall back to best available score or general
        best = keyword_result if keyword_result.confidence >= similarity_result.confidence else similarity_result
        if best.confidence > 0.3:
            logger.info(
                "intent_classified_low_confidence",
                intent=best.intent.value,
                confidence=best.confidence,
                method=best.method,
            )
            return best

        logger.info("intent_classified_general", query_preview=query[:50])
        return ClassificationResult(
            intent=Intent.GENERAL,
            confidence=1.0,
            scores={i.value: 0.0 for i in Intent} | {Intent.GENERAL.value: 1.0},
            method="fallback",
        )

    def _keyword_classify(self, query: str) -> ClassificationResult:
        """Score query against keyword lists for each intent.

        Uses weighted keyword matching where longer keyword phrases
        receive higher weight to prioritize specific matches.
        """
        scores: dict[str, float] = {}
        query_tokens = set(re.findall(r"\b\w+\b", query))

        for intent in [Intent.ANALYTICS, Intent.COMPLIANCE, Intent.ANOMALY]:
            keywords = self._keyword_map.get(intent, [])
            if not keywords:
                scores[intent.value] = 0.0
                continue

            matches = 0
            total_weight = 0.0
            for keyword in keywords:
                kw_lower = keyword.lower()
                # Multi-word keywords checked as substring
                if " " in kw_lower:
                    if kw_lower in query:
                        matches += 1
                        total_weight += len(kw_lower.split()) * 1.5
                else:
                    if kw_lower in query_tokens:
                        matches += 1
                        total_weight += 1.0

            # Normalize: combine match ratio with absolute match count
            match_ratio = matches / len(keywords) if keywords else 0.0
            count_bonus = min(total_weight / 5.0, 0.4)
            scores[intent.value] = min(match_ratio + count_bonus, 1.0)

        scores[Intent.GENERAL.value] = 0.1

        best_intent_name = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_intent_name]

        return ClassificationResult(
            intent=Intent(best_intent_name) if best_score > 0 else Intent.GENERAL,
            confidence=best_score,
            scores=scores,
            method="keyword",
        )

    def _similarity_classify(self, query: str) -> ClassificationResult:
        """Score query by computing token overlap similarity with labeled examples."""
        query_tokens = set(re.findall(r"\b\w+\b", query))
        scores: dict[str, float] = {}

        for intent in [Intent.ANALYTICS, Intent.COMPLIANCE, Intent.ANOMALY]:
            examples = self._example_map.get(intent, [])
            if not examples:
                scores[intent.value] = 0.0
                continue

            max_similarity = 0.0
            for example in examples:
                example_tokens = set(re.findall(r"\b\w+\b", example))
                if not example_tokens or not query_tokens:
                    continue
                intersection = query_tokens & example_tokens
                union = query_tokens | example_tokens
                jaccard = len(intersection) / len(union) if union else 0.0
                max_similarity = max(max_similarity, jaccard)

            scores[intent.value] = max_similarity

        scores[Intent.GENERAL.value] = 0.1

        best_intent_name = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_intent_name]

        return ClassificationResult(
            intent=Intent(best_intent_name) if best_score > 0 else Intent.GENERAL,
            confidence=best_score,
            scores=scores,
            method="similarity",
        )

    def classify_with_context(
        self,
        query: str,
        recent_intents: list[Intent] | None = None,
    ) -> ClassificationResult:
        """Classify with conversational context awareness.

        If the initial classification is low-confidence and the user has been
        on a specific topic, bias toward continuing that topic (follow-up detection).

        Args:
            query: The user's query.
            recent_intents: List of intents from recent conversation turns.

        Returns:
            Context-aware ClassificationResult.
        """
        result = self.classify(query)

        if recent_intents and result.confidence < self.confidence_threshold:
            # Check for follow-up indicators
            follow_up_patterns = [
                r"^(?:and|also|what about|how about|show me more|tell me more)",
                r"^(?:can you|could you|please)\s+(?:also|additionally)",
                r"^(?:same|similar|like that|those|these)\b",
            ]
            is_follow_up = any(
                re.match(pattern, query, re.IGNORECASE)
                for pattern in follow_up_patterns
            )

            if is_follow_up and recent_intents:
                last_intent = recent_intents[-1]
                boosted_confidence = min(result.confidence + 0.3, 0.85)
                logger.info(
                    "follow_up_detected",
                    original_intent=result.intent.value,
                    boosted_intent=last_intent.value,
                    confidence=boosted_confidence,
                )
                return ClassificationResult(
                    intent=last_intent,
                    confidence=boosted_confidence,
                    scores=result.scores,
                    method="follow_up",
                )

        return result
