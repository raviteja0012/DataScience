"""Multi-database connector abstraction.

Provides a unified interface for connecting to Oracle, SQL Server, and
Snowflake backends.  In demo/test mode the connector returns an in-memory
SQLite engine so that examples and tests can run without real database
infrastructure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__, fmt="text")


class DatabasePlatform(Enum):
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    SNOWFLAKE = "snowflake"
    SQLITE = "sqlite"


@dataclass
class ConnectionConfig:
    """Normalised connection parameters parsed from a source-system YAML."""

    platform: DatabasePlatform
    host: str = ""
    port: int = 0
    database: str = ""
    schema: str = ""
    username: str = ""
    password: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "ConnectionConfig":
        """Build a ``ConnectionConfig`` from the ``connection`` section of a
        source-system YAML file."""
        system_cfg = cfg.get("system", {})
        conn_cfg = cfg.get("connection", {})
        platform = DatabasePlatform(system_cfg.get("platform", "sqlite"))

        def _resolve(val: Any) -> str:
            """Resolve ``${ENV_VAR}`` references."""
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                return os.environ.get(val[2:-1], "")
            return str(val) if val is not None else ""

        return cls(
            platform=platform,
            host=_resolve(conn_cfg.get("host", "")),
            port=int(conn_cfg.get("port", 0)),
            database=_resolve(conn_cfg.get("database", conn_cfg.get("service_name", ""))),
            schema=_resolve(conn_cfg.get("schema", "")),
            username=_resolve(conn_cfg.get("username", "")),
            password=_resolve(conn_cfg.get("password", "")),
            extra={k: v for k, v in conn_cfg.items()
                   if k not in ("driver", "host", "port", "database", "service_name",
                                "schema", "username", "password", "account", "warehouse",
                                "role", "pool_size", "pool_timeout", "driver_extra")},
        )


class DatabaseConnector:
    """Thin wrapper providing a uniform ``get_engine`` / ``get_connection`` API.

    When *demo_mode* is ``True`` (the default when no real credentials are
    present), all calls return an in-memory SQLite engine so that the pipeline
    can be exercised end-to-end without external databases.
    """

    def __init__(self, config: ConnectionConfig, demo_mode: bool = True) -> None:
        self._config = config
        self._demo_mode = demo_mode
        self._engine: Optional[Any] = None

    @property
    def platform(self) -> DatabasePlatform:
        return self._config.platform

    @property
    def schema(self) -> str:
        return self._config.schema

    def _build_url(self) -> str:
        """Return the SQLAlchemy connection URL for the configured platform."""
        c = self._config
        if c.platform == DatabasePlatform.ORACLE:
            return (f"oracle+cx_oracle://{c.username}:{c.password}"
                    f"@{c.host}:{c.port}/?service_name={c.database}")
        if c.platform == DatabasePlatform.SQLSERVER:
            driver = c.extra.get("driver_extra", "ODBC Driver 17 for SQL Server")
            return (f"mssql+pyodbc://{c.username}:{c.password}"
                    f"@{c.host}:{c.port}/{c.database}"
                    f"?driver={driver}")
        if c.platform == DatabasePlatform.SNOWFLAKE:
            return (f"snowflake://{c.username}:{c.password}"
                    f"@{c.extra.get('account', '')}/{c.database}/{c.schema}"
                    f"?warehouse={c.extra.get('warehouse', '')}"
                    f"&role={c.extra.get('role', '')}")
        return "sqlite://"

    def get_engine(self) -> Any:
        """Return a SQLAlchemy ``Engine``.  In demo mode returns SQLite."""
        if self._engine is not None:
            return self._engine

        try:
            from sqlalchemy import create_engine
        except ImportError:
            logger.warning("SQLAlchemy not installed; falling back to stub engine.")
            return None

        if self._demo_mode:
            self._engine = create_engine("sqlite:///:memory:", echo=False)
        else:
            self._engine = create_engine(
                self._build_url(),
                pool_size=5,
                pool_pre_ping=True,
            )
        logger.info(
            "Database engine created (platform=%s, demo=%s)",
            self._config.platform.value,
            self._demo_mode,
        )
        return self._engine

    def test_connection(self) -> bool:
        """Return ``True`` if a connection can be established."""
        engine = self.get_engine()
        if engine is None:
            return False
        try:
            with engine.connect() as conn:
                conn.execute(type(conn).connection.property.fget(conn) and
                             engine.dialect.do_ping(conn.connection) or
                             conn.exec_driver_sql("SELECT 1"))
            return True
        except Exception as exc:
            logger.error("Connection test failed: %s", exc)
            return False

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
