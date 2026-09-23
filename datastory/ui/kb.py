"""Общий доступ интерфейса к базе знаний (один экземпляр на процесс Streamlit)."""
from __future__ import annotations

import streamlit as st

from datastory.config import get_settings
from datastory.rag.embeddings import EmbeddingError
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.models import DatasetProfile


@st.cache_resource(show_spinner=False)
def _cached_kb(provider_mode: str, has_key: bool, chroma_dir: str) -> KnowledgeBase:
    # Аргументы нужны только как ключ кэша: при смене режима или каталога создаётся новый экземпляр.
    return KnowledgeBase()


def get_kb() -> KnowledgeBase:
    """Бросает EmbeddingError, если провайдер выбран неверно (например, openai без ключа)."""
    settings = get_settings()
    return _cached_kb(settings.embedding_provider, settings.has_openai_key, str(settings.chroma_dir))


def index_profile_safely(profile: DatasetProfile) -> tuple[bool, str]:
    """Индексирует описание датасета; ошибки превращаются в сообщение, а не в падение страницы."""
    try:
        result = get_kb().index_dataset_profile(profile)
    except EmbeddingError as exc:
        return False, f"Датасет сохранён, но его описание не проиндексировано: {exc.user_message}"
    except Exception as exc:  # noqa: BLE001 — сбой индекса не должен ломать сохранение датасета
        return False, f"Датасет сохранён, но индексация не удалась: {exc}"
    return True, result.message
