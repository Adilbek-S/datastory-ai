"""Конфигурация приложения: значения читаются из .env и переменных окружения."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"  # распознавание таблиц на изображениях
    # Параметры генерации структурных ответов. Значения по умолчанию выбраны экспериментом (EVALS.md, раздел «Выбор гиперпараметров LLM»)
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 1024
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"  # auto | openai | offline

    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    workspace_dir: Path = Field(default=PROJECT_ROOT / "data" / "workspace")
    skills_dir: Path = Field(default=PROJECT_ROOT / ".claude" / "skills")  # Skill методики анализа (datastory-analysis)

    # MCP: аналитический сервер запускается дочерним процессом (stdio)
    mcp_server_module: str = "datastory.mcp_server.server"
    mcp_startup_timeout: float = 30.0
    mcp_call_timeout: float = 60.0

    # Официальные примеры LangSmith годами учили задавать LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT
    # (текущее имя — LANGSMITH_*); принимаем оба варианта, чтобы ключ, заданный по любому туториалу, не терялся молча.
    langsmith_tracing: bool = Field(default=False, validation_alias=AliasChoices("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"))
    langsmith_api_key: str = Field(default="", validation_alias=AliasChoices("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"))
    langsmith_project: str = Field(default="datastory-ai", validation_alias=AliasChoices("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"))
    langsmith_endpoint: str = Field(default="", validation_alias=AliasChoices("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"))  # пусто — облако LangSmith по умолчанию; для EU и self-hosted укажите свой URL
    langsmith_workspace_id: str = ""  # нужен только для ключей, привязанных к нескольким рабочим пространствам

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for name in ("chroma_dir", "workspace_dir", "skills_dir"):
        path = getattr(settings, name)
        if not path.is_absolute():
            setattr(settings, name, PROJECT_ROOT / path)
    _configure_tracing(settings)
    return settings


def _configure_tracing(settings: Settings) -> None:
    """LangSmith читает настройки из переменных окружения: выставляем их из .env и сообщаем, включена ли трассировка."""
    from datastory.observability import configure_tracing  # локальный импорт: observability импортирует этот модуль

    configure_tracing(settings)
