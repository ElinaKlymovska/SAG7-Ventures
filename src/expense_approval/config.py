from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "expense_approval.sqlite3"


@dataclass(frozen=True)
class Settings:
    database_url: str
    openai_api_key: str | None
    openai_model: str = "gpt-5-mini"
    ai_timeout_seconds: float = 10.0
    ai_send_document_images: bool = False


def _value(name: str, secrets: Mapping[str, Any] | None, default: Any = None) -> Any:
    if secrets is not None:
        try:
            value = secrets[name]
        except (KeyError, TypeError):
            value = None
        if value not in (None, ""):
            return value
    return os.getenv(name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def load_settings(secrets: Mapping[str, Any] | None = None) -> Settings:
    database_url = str(
        _value("DATABASE_URL", secrets, f"sqlite:///{DEFAULT_DATABASE_PATH}")
    )
    api_key = str(_value("OPENAI_API_KEY", secrets, "")).strip() or None
    model = str(_value("OPENAI_MODEL", secrets, "gpt-5-mini")).strip()
    timeout = float(_value("AI_TIMEOUT_SECONDS", secrets, 10.0))
    send_images = _as_bool(_value("AI_SEND_DOCUMENT_IMAGES", secrets, False))
    return Settings(
        database_url=database_url,
        openai_api_key=api_key,
        openai_model=model,
        ai_timeout_seconds=timeout,
        ai_send_document_images=send_images,
    )
