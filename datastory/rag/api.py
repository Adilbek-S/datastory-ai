"""Функции верхнего уровня RAG Pipeline.

    index_dataset_profile   — описание датасета и колонок в индекс
    index_document          — PDF в индекс (с защитой от повторной индексации)
    search_dataset_metadata — поиск датасета/колонок по смыслу
    search_business_context — поиск фрагментов документов (Top-3) с источниками
    get_source_reference    — источник по chunk_id

Последний необязательный аргумент kb — экземпляр KnowledgeBase (по умолчанию общий из настроек).
"""
from __future__ import annotations

from functools import lru_cache

from datastory.models import DatasetProfile
from datastory.rag.knowledge_base import DEFAULT_TOP_K, KnowledgeBase
from datastory.rag.models import (
    ColumnResolution,
    ContextSearchResult,
    DatasetIndexResult,
    DocumentIndexResult,
    MetadataSearchResult,
    SourceReference,
)


@lru_cache
def get_default_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


def index_dataset_profile(profile: DatasetProfile, kb: KnowledgeBase | None = None) -> DatasetIndexResult:
    return (kb or get_default_knowledge_base()).index_dataset_profile(profile)


def index_document(
    data: bytes, filename: str, dataset_id: str | None = None, kb: KnowledgeBase | None = None
) -> DocumentIndexResult:
    return (kb or get_default_knowledge_base()).index_document(data, filename, dataset_id)


def search_dataset_metadata(
    query: str, top_k: int = DEFAULT_TOP_K, *, dataset_id: str | None = None, entry_type: str | None = None,
    kb: KnowledgeBase | None = None,
) -> MetadataSearchResult:
    return (kb or get_default_knowledge_base()).search_dataset_metadata(
        query, top_k, dataset_id=dataset_id, entry_type=entry_type
    )


def resolve_column(
    query: str, *, dataset_id: str | None = None, kb: KnowledgeBase | None = None
) -> ColumnResolution:
    return (kb or get_default_knowledge_base()).resolve_column(query, dataset_id=dataset_id)


def search_business_context(
    query: str, top_k: int = DEFAULT_TOP_K, *, dataset_id: str | None = None, document_id: str | None = None,
    kb: KnowledgeBase | None = None,
) -> ContextSearchResult:
    return (kb or get_default_knowledge_base()).search_business_context(
        query, top_k, dataset_id=dataset_id, document_id=document_id
    )


def get_source_reference(chunk_id: str, kb: KnowledgeBase | None = None) -> SourceReference:
    return (kb or get_default_knowledge_base()).get_source_reference(chunk_id)
