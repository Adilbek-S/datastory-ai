"""Доступ интерфейса к воркфлоу анализа.

AnalysisRunner держит граф и чекпоинтер и кэшируется на процесс Streamlit (st.cache_resource): перезапуск скрипта
не пересоздаёт граф, поэтому сессия анализа (thread_id в st.session_state) продолжается с точки подтверждения.
"""
from __future__ import annotations

import streamlit as st

from datastory.chat.graph import ChatService
from datastory.config import get_settings
from datastory.errors import LLMUnavailableError
from datastory.llm.client import get_llm
from datastory.rag.embeddings import EmbeddingError
from datastory.ui.kb import get_kb
from datastory.ui.mcp import get_analytics_client
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.runner import AnalysisRunner


def _kb_or_none():
    try:
        return get_kb()
    except EmbeddingError:
        return None  # без базы знаний воркфлоу отметит отсутствие документации


@st.cache_resource(show_spinner=False)
def _cached_runner(openai_model: str, has_key: bool, server_module: str, workspace_dir: str, chroma_dir: str) -> AnalysisRunner:
    # Аргументы — ключ кэша: смена модели, ключа или каталогов создаёт новый воркфлоу.
    try:
        llm = get_llm(get_settings())
    except LLMUnavailableError:
        llm = None
    return AnalysisRunner(WorkflowDeps(client=get_analytics_client, llm=lambda: llm, kb=_kb_or_none))


def get_runner() -> AnalysisRunner:
    s = get_settings()
    return _cached_runner(s.openai_model, s.has_openai_key, s.mcp_server_module, str(s.workspace_dir), str(s.chroma_dir))


@st.cache_resource(show_spinner=False)
def _cached_chat(_runner: AnalysisRunner, runner_id: int) -> ChatService:
    # runner_id — ключ кэша: новый воркфлоу (смена настроек) получает новый чат; сам runner не хешируется
    return ChatService(_runner.deps)  # те же зависимости, что у воркфлоу: MCP, LLM, база знаний, Skill


def get_chat_service() -> ChatService:
    runner = get_runner()
    return _cached_chat(runner, id(runner))
