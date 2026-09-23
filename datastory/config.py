"""Конфигурация приложения: значения читаются из .env и переменных окружения."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "datastory-ai"

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if not settings.chroma_dir.is_absolute():
        settings.chroma_dir = PROJECT_ROOT / settings.chroma_dir
    _configure_langsmith(settings)
    return settings


def _configure_langsmith(settings: Settings) -> None:
    """LangSmith читает настройки из переменных окружения — выставляем их из .env."""
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
