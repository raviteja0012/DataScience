"""Configuration package for biller integration simulator."""

import os
from pathlib import Path

CONFIG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

BILLER_CONFIG_PATH = CONFIG_DIR / "biller_config.yaml"
SCHEMA_MAPPING_PATH = CONFIG_DIR / "schema_mapping.yaml"
SETTLEMENT_RULES_PATH = CONFIG_DIR / "settlement_rules.yaml"


def get_config_path(filename: str) -> Path:
    """Resolve a config file path relative to the config directory."""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return path
