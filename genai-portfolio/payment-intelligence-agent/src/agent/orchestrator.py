"""Cortex Agent orchestrator - the central coordination layer.

Routes classified user intents to the appropriate subsystem (analytics,
compliance RAG, or anomaly detection), manages conversation state, and
assembles rich responses. In production, this integrates with Snowflake
Cortex Agent APIs; in demo mode, it orchestrates local subsystems with
synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .intent_classifier import IntentClassifier, Intent, ClassificationResult
from .context_manager import ConversationContext
from .response_formatter import ResponseFormatter, FormattedResponse
from ..analytics.nl_to_sql import NLToSQLTranslator
from ..analytics.query_validator import QueryValidator
from ..analytics.result_analyzer import ResultAnalyzer
from ..rag.retriever import RAGRetriever
from ..anomaly.detector import AnomalyDetector
from ..utils.logger import get_logger
from ..utils.security import InputSanitizer, PIIMasker

logger = get_logger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the agent orchestrator.

    Attributes:
        demo_mode: Whether to operate with synthetic data.
        show_sql: Whether to include SQL in analytics responses.
        max_display_rows: Maximum rows to display in table responses.
        confidence_threshold: Minimum intent confidence for routing.
    """

    demo_mode: bool = True
    show_sql: bool = True
    max_display_rows: int = 100
    confidence_threshold: float = 0.65


class PaymentAgentOrchestrator:
    """Central orchestrator that routes queries to specialized subsystems.

    Implements the agentic loop: classify intent, route to the appropriate
    handler, format the response, and update conversation context. Supports
    both live Snowflake Cortex and demo mode operation.

    Usage:
        orchestrator = PaymentAgentOrchestrator()
        response = orchestrator.process_query("Show me top 10 merchants by volume")
    """

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.config = config or OrchestratorConfig()
        self._context = ConversationContext()
        self._classifier = IntentClassifier(
            confidence_threshold=self.config.confidence_threshold,
        )
        self._formatter = ResponseFormatter(
            max_display_rows=self.config.max_display_rows,
            show_sql=self.config.show_sql,
        )
        self._sanitizer = InputSanitizer()
        self._pii_masker = PIIMasker()

        # Subsystem initialization
        self._nl_to_sql = NLToSQLTranslator()
        self._query_validator = QueryValidator()
        self._result_analyzer = ResultAnalyzer()
        self._rag_retriever: RAGRetriever | None = None
        self._anomaly_detector = AnomalyDetector()

        logger.info(
            "orchestrator_initialized",
            demo_mode=self.config.demo_mode,
        )

    @property
    def context(self) -> ConversationContext:
        """Access the conversation context manager."""
        return self._context

    def initialize_rag(self, documents_path: str | None = None) -> None:
        """Initialize the RAG retriever with PCI compliance documents.

        Args:
            documents_path: Path to directory containing PCI documents.
                           Uses default sample docs if not specified.
        """
        try:
            self._rag_retriever = RAGRetriever()
            self._rag_retriever.initialize(documents_path)
            logger.info("rag_initialized", documents_path=documents_path)
        except Exception as exc:
            logger.error("rag_initialization_failed", error=str(exc))
            self._rag_retriever = None

    def process_query(self, user_query: str) -> FormattedResponse:
        """Process a user query through the full agent pipeline.

        Pipeline stages:
        1. Input sanitization and PII masking
        2. Intent classification with context awareness
        3. Route to appropriate subsystem handler
        4. Format response with visualizations
        5. Update conversation context

        Args:
            user_query: The user's natural language query.

        Returns:
            FormattedResponse with text, optional chart/table, and metadata.
        """
        # Stage 1: Sanitize input
        try:
            sanitized_query = self._sanitizer.sanitize(user_query)
        except ValueError as exc:
            logger.warning("input_sanitization_failed", error=str(exc))
            return self._formatter.format_error_response(
                str(exc),
                suggestion="Please rephrase your query without special characters.",
            )

        # Stage 2: Classify intent with conversation context
        recent_intents = self._context.get_recent_intents()
        classification = self._classifier.classify_with_context(
            sanitized_query,
            recent_intents=recent_intents,
        )

        logger.info(
            "query_classified",
            intent=classification.intent.value,
            confidence=classification.confidence,
            method=classification.method,
        )

        # Record user turn
        self._context.add_user_turn(
            content=sanitized_query,
            intent=classification.intent,
            metadata={"classification": classification.scores},
        )

        # Stage 3: Route to handler
        try:
            response = self._route_to_handler(sanitized_query, classification)
        except Exception as exc:
            logger.error("handler_error", intent=classification.intent.value, error=str(exc))
            response = self._formatter.format_error_response(
                f"Error processing your {classification.intent.value} query: {exc}",
                suggestion="Try rephrasing your question or switching to a different mode.",
            )

        # Stage 4: Update context with response
        self._context.add_assistant_turn(
            content=response.text,
            metadata={
                "response_type": response.response_type.value,
                "sql": response.sql,
                "has_chart": response.chart is not None,
            },
        )

        return response

    def _route_to_handler(
        self,
        query: str,
        classification: ClassificationResult,
    ) -> FormattedResponse:
        """Route the classified query to the appropriate subsystem.

        Args:
            query: Sanitized user query.
            classification: Intent classification result.

        Returns:
            FormattedResponse from the appropriate handler.
        """
        intent = classification.intent

        if intent == Intent.ANALYTICS:
            return self._handle_analytics(query)
        elif intent == Intent.COMPLIANCE:
            return self._handle_compliance(query)
        elif intent == Intent.ANOMALY:
            return self._handle_anomaly(query)
        else:
            return self._handle_general(query)

    def _handle_analytics(self, query: str) -> FormattedResponse:
        """Handle analytics (NL-to-SQL) queries.

        Translates the query to SQL, validates it, executes it (or simulates
        execution in demo mode), and formats the results.
        """
        # Generate SQL from natural language
        context = self._context.entities.get_context_summary()
        sql_result = self._nl_to_sql.translate(query, context=context)
        generated_sql = sql_result.get("sql", "")
        explanation = sql_result.get("explanation", "")

        # Validate the generated SQL
        validation = self._query_validator.validate(generated_sql)
        if not validation.is_safe:
            return self._formatter.format_error_response(
                f"The generated query did not pass safety validation: {validation.reason}",
                suggestion="Try rephrasing with more specific criteria.",
            )

        # Execute query (demo mode uses synthetic results)
        if self.config.demo_mode:
            result_df = self._nl_to_sql.generate_demo_results(query, generated_sql)
        else:
            from ..utils.snowflake_client import SnowflakeClient
            client = SnowflakeClient()
            result_df = client.execute_query(generated_sql)

        # Analyze results for insights
        summary = self._result_analyzer.summarize(result_df, query)

        # Mask PII in results
        if not result_df.empty:
            for col in result_df.columns:
                if col.lower() in {"card_number", "ssn", "account_number"}:
                    result_df[col] = result_df[col].astype(str).apply(self._pii_masker.mask_text)

        return self._formatter.format_analytics_response(
            data=result_df,
            query=query,
            sql=generated_sql,
            summary=f"{summary}\n\n*{explanation}*" if explanation else summary,
        )

    def _handle_compliance(self, query: str) -> FormattedResponse:
        """Handle PCI compliance RAG queries."""
        if self._rag_retriever is None:
            self.initialize_rag()

        if self._rag_retriever is None:
            return self._formatter.format_error_response(
                "The compliance knowledge base is not available.",
                suggestion="Ensure PCI documents are loaded in the config directory.",
            )

        rag_result = self._rag_retriever.query(query)

        return self._formatter.format_compliance_response(
            answer=rag_result.get("answer", "No relevant information found."),
            sources=rag_result.get("sources", []),
            requirement_ids=rag_result.get("requirement_ids", []),
        )

    def _handle_anomaly(self, query: str) -> FormattedResponse:
        """Handle anomaly detection queries."""
        detection_result = self._anomaly_detector.analyze(query, demo_mode=self.config.demo_mode)

        return self._formatter.format_anomaly_response(
            anomalies=detection_result.get("anomalies", pd.DataFrame()),
            summary=detection_result.get("summary", "Analysis complete."),
            alerts=detection_result.get("alerts", []),
        )

    def _handle_general(self, query: str) -> FormattedResponse:
        """Handle general queries that don't map to a specific subsystem."""
        return FormattedResponse(
            text=(
                "I can help you with three types of payment intelligence queries:\n\n"
                "**Analytics** - Ask about transaction data, revenue, merchants, "
                "settlements, and chargebacks. Example: *'Show me top 10 merchants "
                "by transaction volume last month'*\n\n"
                "**Compliance** - Ask about PCI DSS requirements, cardholder data "
                "protection, and security policies. Example: *'What are the requirements "
                "for storing cardholder data?'*\n\n"
                "**Anomaly Detection** - Ask about suspicious patterns, fraud indicators, "
                "and transaction anomalies. Example: *'Are there any anomalies in recent "
                "transactions?'*\n\n"
                "How can I help you today?"
            ),
        )

    def reset_session(self) -> None:
        """Reset the conversation context for a new session."""
        self._context.clear()
        logger.info("session_reset")

    def get_session_summary(self) -> dict[str, Any]:
        """Get a summary of the current session state."""
        return {
            "session_id": self._context.session_id,
            "turn_count": self._context.turn_count,
            "recent_intents": [i.value for i in self._context.get_recent_intents()],
            "entities": self._context.entities.get_context_summary(),
            "is_expired": self._context.is_expired,
        }
