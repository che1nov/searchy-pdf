"""Configuration loading utilities for the PDF search service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from YAML."""

    directories: list[Path]
    host: str = "127.0.0.1"
    port: int = 8000
    index_file: Path = Path("index.pkl")


def load_config(config_path: Path) -> AppConfig:
    """Load and validate configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    directories_raw = raw.get("directories")
    if not isinstance(directories_raw, list) or not directories_raw:
        raise ValueError("'directories' must be a non-empty list in config.yml")

    directories: list[Path] = []
    for value in directories_raw:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Each directory in 'directories' must be a non-empty string")

        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        directories.append(candidate)

    host = raw.get("host", "127.0.0.1")
    port = raw.get("port", 8000)
    index_file_raw = raw.get("index_file", "index.pkl")

    if not isinstance(host, str) or not host:
        raise ValueError("'host' must be a non-empty string")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError("'port' must be an integer between 1 and 65535")
    if not isinstance(index_file_raw, str) or not index_file_raw:
        raise ValueError("'index_file' must be a non-empty string")

    index_file = Path(index_file_raw)
    if not index_file.is_absolute():
        index_file = (config_path.parent / index_file).resolve()

    return AppConfig(
        directories=directories,
        host=host,
        port=port,
        index_file=index_file,
    )
