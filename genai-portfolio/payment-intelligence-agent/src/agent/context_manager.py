"""Conversation context manager for maintaining state across dialogue turns.

Tracks conversation history, extracted entities, active intent threads,
and session metadata to enable multi-turn interactions where the agent
can resolve references like "those merchants" or "the same time period."
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .intent_classifier import Intent
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation history.

    Attributes:
        role: Either 'user' or 'assistant'.
        content: The message text.
        intent: Classified intent for this turn (user turns only).
        metadata: Additional structured data (SQL executed, chart config, etc.).
        timestamp: Unix timestamp of when this turn occurred.
    """

    role: str
    content: str
    intent: Intent | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class EntityMemory:
    """Tracks entities extracted from the conversation for reference resolution.

    Enables the agent to resolve pronouns and references like
    "those merchants", "the same period", "that metric" by maintaining
    a rolling window of extracted entities.
    """

    merchants: list[str] = field(default_factory=list)
    time_ranges: list[dict[str, str]] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    payment_methods: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    def update_from_query(self, query: str, metadata: dict[str, Any] | None = None) -> None:
        """Extract and store entities from a user query.

        Performs lightweight entity extraction using pattern matching.
        For production use, this would be augmented with NER from Cortex.

        Args:
            query: The user's natural language query.
            metadata: Optional structured metadata from query processing.
        """
        import re

        query_lower = query.lower()

        # Extract time references
        time_patterns = [
            r"(?:last|past|previous)\s+(\d+\s+(?:day|week|month|quarter|year)s?)",
            r"(?:in|during|for)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?",
            r"(today|yesterday|this week|this month|this quarter|this year|ytd|mtd|qtd)",
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                time_ref = match if isinstance(match, str) else " ".join(m for m in match if m)
                if time_ref and time_ref not in [tr.get("reference") for tr in self.time_ranges]:
                    self.time_ranges.append({"reference": time_ref})
                    if len(self.time_ranges) > 5:
                        self.time_ranges.pop(0)

        # Extract payment methods
        methods = ["visa", "mastercard", "amex", "discover", "ach", "wire", "debit", "credit"]
        for method in methods:
            if method in query_lower and method not in self.payment_methods:
                self.payment_methods.append(method)

        # Store metadata entities
        if metadata:
            if "merchants" in metadata:
                self.merchants = metadata["merchants"][-10:]
            if "metrics" in metadata:
                self.metrics = metadata["metrics"][-5:]

    def get_context_summary(self) -> dict[str, Any]:
        """Return a summary of tracked entities for prompt construction."""
        summary: dict[str, Any] = {}
        if self.merchants:
            summary["recent_merchants"] = self.merchants[-5:]
        if self.time_ranges:
            summary["active_time_range"] = self.time_ranges[-1]
        if self.metrics:
            summary["recent_metrics"] = self.metrics[-3:]
        if self.payment_methods:
            summary["payment_methods"] = self.payment_methods[-3:]
        if self.regions:
            summary["regions"] = self.regions[-3:]
        return summary


class ConversationContext:
    """Manages the full conversation state for a user session.

    Maintains an ordered history of conversation turns, entity memory,
    and session metadata. Supports context windowing to keep prompt sizes
    manageable while retaining important information.

    Attributes:
        session_id: Unique identifier for this conversation session.
        max_history: Maximum number of turns to retain.
        timeout_minutes: Session timeout in minutes.
    """

    def __init__(
        self,
        session_id: str | None = None,
        max_history: int = 20,
        timeout_minutes: int = 30,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.max_history = max_history
        self.timeout_minutes = timeout_minutes
        self._history: list[ConversationTurn] = []
        self._entities = EntityMemory()
        self._created_at = time.time()
        self._last_activity = time.time()

        logger.info("session_created", session_id=self.session_id)

    @property
    def is_expired(self) -> bool:
        """Check if the session has exceeded its timeout window."""
        elapsed = (time.time() - self._last_activity) / 60.0
        return elapsed > self.timeout_minutes

    @property
    def turn_count(self) -> int:
        """Number of turns in the conversation history."""
        return len(self._history)

    @property
    def entities(self) -> EntityMemory:
        """Access the entity memory for this session."""
        return self._entities

    def add_user_turn(
        self,
        content: str,
        intent: Intent | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a user message in the conversation history.

        Args:
            content: The user's message text.
            intent: Classified intent for this message.
            metadata: Additional structured data.
        """
        self._last_activity = time.time()
        turn = ConversationTurn(
            role="user",
            content=content,
            intent=intent,
            metadata=metadata or {},
        )
        self._history.append(turn)
        self._entities.update_from_query(content, metadata)
        self._trim_history()

        logger.debug(
            "user_turn_added",
            session_id=self.session_id,
            turn_count=self.turn_count,
            intent=intent.value if intent else None,
        )

    def add_assistant_turn(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an assistant response in the conversation history.

        Args:
            content: The assistant's response text.
            metadata: Structured data about the response (SQL used, chart config, etc.).
        """
        self._last_activity = time.time()
        turn = ConversationTurn(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        self._history.append(turn)
        self._trim_history()

    def get_recent_intents(self, n: int = 5) -> list[Intent]:
        """Get the intents from the last N user turns.

        Args:
            n: Number of recent turns to examine.

        Returns:
            List of intents from recent user messages.
        """
        user_turns = [t for t in self._history if t.role == "user" and t.intent]
        return [t.intent for t in user_turns[-n:]]  # type: ignore[misc]

    def get_history_for_prompt(self, max_turns: int | None = None) -> list[dict[str, str]]:
        """Format conversation history for inclusion in an LLM prompt.

        Returns the most recent turns formatted as role/content pairs,
        suitable for constructing a multi-turn prompt.

        Args:
            max_turns: Override for maximum turns to include.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        limit = max_turns or min(self.max_history, 10)
        recent = self._history[-limit:]
        return [
            {"role": turn.role, "content": turn.content}
            for turn in recent
        ]

    def get_context_window(self) -> dict[str, Any]:
        """Build a comprehensive context window for the agent.

        Combines conversation history, entity memory, and session metadata
        into a structured context object for prompt construction.

        Returns:
            Dictionary containing all context information.
        """
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "history": self.get_history_for_prompt(),
            "entities": self._entities.get_context_summary(),
            "recent_intents": [i.value for i in self.get_recent_intents()],
        }

    def clear(self) -> None:
        """Reset the conversation history and entity memory."""
        self._history.clear()
        self._entities = EntityMemory()
        self._last_activity = time.time()
        logger.info("session_cleared", session_id=self.session_id)

    def _trim_history(self) -> None:
        """Trim history to max_history, keeping the most recent turns."""
        if len(self._history) > self.max_history:
            excess = len(self._history) - self.max_history
            self._history = self._history[excess:]
