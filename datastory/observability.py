"""Трассировка LangSmith: конфигурация из переменных окружения и небольшие помощники для span-ов.

Правила:
- трассировка включается, только если задано LANGSMITH_TRACING=true И LANGSMITH_API_KEY. Без ключа она остаётся
  выключенной, а причина сообщается в логе и в интерфейсе: успешная интеграция не имитируется;
- когда трассировка выключена, все помощники ничего не отправляют и не требуют сети;
- в LangSmith уходят промпты и ответы LLM, агрегированные числа, фрагменты документов из RAG и сводки MCP-вызовов.
  Строки таблицы и изображения не отправляются (изображение при распознавании исключено из трассировки).
"""
from __future__ import annotations

import functools
import inspect
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from langsmith import trace, tracing_context

from datastory.config import Settings

logger = logging.getLogger("datastory.tracing")

DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
ENV_NAMES = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT", "LANGSMITH_WORKSPACE_ID")
MAX_TEXT = 300  # длина значений в метаданных и входах span-ов


@dataclass(frozen=True)
class TracingStatus:
    enabled: bool  # трассы реально отправляются в LangSmith
    requested: bool  # LANGSMITH_TRACING=true
    project: str
    endpoint: str
    reason: str  # пояснение для пользователя

    @property
    def label(self) -> str:
        return "Включена" if self.enabled else ("Нет ключа" if self.requested else "Выключена")


def tracing_status(settings: Settings) -> TracingStatus:
    has_key = bool(settings.langsmith_api_key.strip())
    endpoint = settings.langsmith_endpoint.strip() or DEFAULT_ENDPOINT
    if settings.langsmith_tracing and has_key:
        return TracingStatus(True, True, settings.langsmith_project, endpoint, f"Трассы отправляются в проект «{settings.langsmith_project}» ({endpoint}).")
    if settings.langsmith_tracing:
        return TracingStatus(
            False, True, settings.langsmith_project, endpoint,
            "LANGSMITH_TRACING=true, но LANGSMITH_API_KEY не задан: трассировка НЕ включена, трассы не отправляются.",
        )
    return TracingStatus(False, False, settings.langsmith_project, endpoint, "Трассировка выключена (LANGSMITH_TRACING не равно true).")


def configure_tracing(settings: Settings) -> TracingStatus:
    """Переносит настройки в переменные окружения, которые читает LangSmith, и сообщает итог.

    Без ключа трассировка принудительно выключается: иначе LangChain пытался бы отправлять трассы и сыпал бы ошибками 401.
    """
    status = tracing_status(settings)
    if status.enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = status.endpoint
        if settings.langsmith_workspace_id.strip():
            os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id.strip()
        logger.info("LangSmith: %s", status.reason)
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        if status.requested:
            logger.warning("LangSmith: %s", status.reason)
    return status


def trace_config(
    run_name: str, *, thread_id: str | None = None, dataset_id: str | None = None, tags: tuple[str, ...] = (), metadata: dict | None = None
) -> dict[str, Any]:
    """Конфигурация вызова графа/цепочки LangChain: имя корневого run, теги и метаданные (thread_id группирует трассы сессии)."""
    config: dict[str, Any] = {"run_name": run_name, "tags": ["datastory", *tags], "metadata": {**(metadata or {})}}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
        config["metadata"]["thread_id"] = thread_id
    if dataset_id:
        config["metadata"]["dataset_id"] = dataset_id
    return config


def clip(value: Any, limit: int = MAX_TEXT) -> Any:
    return value if not isinstance(value, str) or len(value) <= limit else value[:limit] + "…"


class Span:
    """Обёртка над RunTree: метаданные и результат span-а. Без включённой трассировки ничего не отправляется."""

    def __init__(self, run):
        self._run = run

    def metadata(self, values: dict) -> None:
        self._run.add_metadata({k: clip(v) for k, v in values.items()})

    def outputs(self, values: dict) -> None:
        self._run.end(outputs=values)


@contextmanager
def span(
    name: str, *, run_type: str = "chain", inputs: dict | None = None, tags: tuple[str, ...] = (), metadata: dict | None = None
) -> Iterator[Span]:
    """Именованный span во вложенной трассе. Исключение в теле фиксируется в span как ошибка и пробрасывается дальше."""
    with trace(name, run_type=run_type, inputs=inputs or {}, tags=["datastory", *tags], metadata=metadata or {}) as run:  # type: ignore[arg-type]
        yield Span(run)


def milliseconds_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def untraced():
    """Контекст, внутри которого вложенные вызовы LangChain не трассируются (например, распознавание изображения)."""
    return tracing_context(enabled=False)


def traced(
    name: str, *, run_type: str = "chain", tags: tuple[str, ...] = (),
    inputs: Callable[[dict], dict] | None = None, outputs: Callable[[Any], dict] | None = None,
):
    """Декоратор: оборачивает функцию или метод в span. inputs получает аргументы вызова без self, outputs — результат."""

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
            with span(name, run_type=run_type, tags=tags, inputs=inputs(arguments) if inputs else {}) as sp:
                result = fn(*args, **kwargs)
                if outputs:
                    sp.outputs(outputs(result))
                return result

        return wrapper

    return decorate
