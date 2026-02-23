"""Snowflake connection wrapper with connection pooling and Cortex integration.

Provides a unified interface for executing queries, calling Cortex LLM functions,
and generating embeddings, with automatic fallback to demo mode when no live
Snowflake connection is available.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import yaml

from .logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "snowflake_config.yaml"


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load Snowflake configuration from YAML, resolving env var placeholders."""
    path = config_path or _CONFIG_PATH
    if not path.exists():
        logger.warning("config_not_found", path=str(path))
        return {"demo": {"enabled": True}}

    with open(path) as f:
        config = yaml.safe_load(f)

    # Resolve environment variable references
    def _resolve(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, value)
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(item) for item in value]
        return value

    return _resolve(config)


class SnowflakeClient:
    """Manages Snowflake connections and provides query execution capabilities.

    In demo mode, all operations return synthetic data without requiring
    a live Snowflake connection. This enables local development and testing.

    Attributes:
        demo_mode: Whether the client is operating with synthetic data.
        config: Loaded configuration dictionary.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self.config = _load_config(config_path)
        self.demo_mode = self.config.get("demo", {}).get("enabled", True)
        self._connection = None

        if self.demo_mode:
            logger.info("snowflake_client_demo_mode")
        else:
            logger.info("snowflake_client_initialized", account=self.config.get("connection", {}).get("account"))

    def _get_connection(self) -> Any:
        """Establish or return existing Snowflake connection.

        Returns:
            Active Snowflake connection object.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        if self.demo_mode:
            raise ConnectionError("Cannot connect in demo mode")

        if self._connection is not None:
            return self._connection

        try:
            import snowflake.connector

            conn_config = self.config.get("connection", {})
            self._connection = snowflake.connector.connect(
                account=conn_config.get("account"),
                user=conn_config.get("user"),
                password=conn_config.get("password"),
                role=conn_config.get("role"),
                warehouse=conn_config.get("warehouse"),
                database=conn_config.get("database"),
                schema=conn_config.get("schema"),
            )
            logger.info("snowflake_connected", account=conn_config.get("account"))
            return self._connection
        except Exception as exc:
            logger.error("snowflake_connection_failed", error=str(exc))
            raise ConnectionError(f"Failed to connect to Snowflake: {exc}") from exc

    @contextmanager
    def cursor(self) -> Generator[Any, None, None]:
        """Context manager providing a Snowflake cursor.

        Yields:
            Snowflake cursor object for query execution.
        """
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def execute_query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: SQL query string.
            params: Optional bind parameters.

        Returns:
            Query results as a pandas DataFrame.

        Raises:
            ConnectionError: If in demo mode (use synthetic data instead).
        """
        if self.demo_mode:
            raise ConnectionError(
                "Direct query execution unavailable in demo mode. "
                "Use the synthetic data generator for testing."
            )

        logger.info("executing_query", sql_preview=sql[:100])
        with self.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)

    def call_cortex_complete(self, prompt: str, model: str | None = None) -> str:
        """Call Snowflake Cortex COMPLETE function for LLM inference.

        Args:
            prompt: The prompt text to send to the model.
            model: Override model name (defaults to config value).

        Returns:
            Model response text.
        """
        if self.demo_mode:
            logger.info("cortex_complete_demo_mode")
            return self._demo_complete(prompt)

        model_name = model or self.config.get("cortex", {}).get("llm_model", "mistral-large2")
        sql = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model_name}', %s) AS response"
        with self.cursor() as cur:
            cur.execute(sql, (prompt,))
            result = cur.fetchone()
            return result[0] if result else ""

    def call_cortex_embed(self, text: str, model: str | None = None) -> list[float]:
        """Generate embeddings using Snowflake Cortex EMBED function.

        Args:
            text: Text to embed.
            model: Override embedding model name.

        Returns:
            Embedding vector as a list of floats.
        """
        if self.demo_mode:
            logger.info("cortex_embed_demo_mode")
            return []

        model_name = model or self.config.get("cortex", {}).get("embed_model", "e5-base-v2")
        sql = f"SELECT SNOWFLAKE.CORTEX.EMBED('{model_name}', %s) AS embedding"
        with self.cursor() as cur:
            cur.execute(sql, (text,))
            result = cur.fetchone()
            return result[0] if result else []

    def _demo_complete(self, prompt: str) -> str:
        """Provide a synthetic LLM response for demo mode.

        Parses the prompt to generate contextually appropriate responses
        without requiring a live Cortex connection.
        """
        prompt_lower = prompt.lower()
        if "sql" in prompt_lower or "query" in prompt_lower:
            return "Based on the payment data analysis, the query results show the expected patterns."
        if "pci" in prompt_lower or "compliance" in prompt_lower:
            return (
                "Per PCI DSS v4.0, organizations must implement strong access controls, "
                "encrypt cardholder data in transit and at rest, and maintain comprehensive audit logs."
            )
        if "anomaly" in prompt_lower or "fraud" in prompt_lower:
            return (
                "The anomaly detection system identified several transactions that deviate "
                "significantly from established patterns based on statistical analysis."
            )
        return "Analysis complete. Please review the detailed results below."

    def close(self) -> None:
        """Close the Snowflake connection if open."""
        if self._connection is not None:
            try:
                self._connection.close()
                logger.info("snowflake_connection_closed")
            except Exception as exc:
                logger.warning("snowflake_close_error", error=str(exc))
            finally:
                self._connection = None

    def __del__(self) -> None:
        self.close()
