"""Этап retrieve_context: бизнес-определения показателей и сопроводительный контекст из RAG.

RAG возвращает только фрагменты с источниками; числа он не считает. Фрагмент признаётся определением показателя,
только если он релевантен запросу и называет сам показатель: иначе система отмечает отсутствие документации.
"""
from __future__ import annotations

import logging
from typing import Callable

from pydantic import BaseModel, Field

from datastory.rag.knowledge_base import KnowledgeBase
from datastory.rag.models import SourceReference
from datastory.workflow.catalog import BY_ID, doc_query, doc_terms

logger = logging.getLogger("datastory.workflow")

CONTEXT_QUERY = "событие в периоде, обновление, изменение показателей, контекст и ограничения интерпретации"
MAX_QUOTE = 500
MAX_EVENTS = 2


class ContextFragment(BaseModel):
    source: SourceReference
    text: str
    score: float = 0.0

    @property
    def chunk_id(self) -> str:
        return self.source.chunk_id

    @property
    def citation(self) -> str:
        return self.source.citation


class BusinessContext(BaseModel):
    query_hits: list[ContextFragment] = Field(default_factory=list, description="Фрагменты, найденные по запросу пользователя (Top-3)")
    definitions: dict[str, ContextFragment] = Field(default_factory=dict, description="metric -> определение из документации")
    events: list[ContextFragment] = Field(default_factory=list, description="Сопроводительный контекст (события, ограничения)")
    provider: str = ""
    error: str | None = None  # база знаний недоступна: все показатели считаются недокументированными

    def fragment(self, chunk_id: str) -> ContextFragment | None:
        found = [*self.definitions.values(), *self.events]
        return next((f for f in found if f.chunk_id == chunk_id), None)


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def find_definition(
    search: Callable[[str], object], query: str, terms: tuple[str, ...], require_all: bool = False
) -> ContextFragment | None:
    """Первый релевантный фрагмент, который называет показатель; иначе None (документации нет).

    require_all: фрагмент обязан назвать все термины (у отношения — и числитель, и знаменатель).
    """
    result = search(query)
    check = all if require_all else any
    for hit in result.hits:
        if hit.relevant and check(_norm(term) in _norm(hit.text) for term in terms):
            return ContextFragment(source=hit.source, text=hit.text[:MAX_QUOTE], score=hit.score)
    return None


def retrieve_query_hits(query: str, search: Callable[[str], object], only_relevant: bool = True) -> list[ContextFragment]:
    """Фрагменты документов по запросу пользователя (для планировщика). Ошибка поиска — пустой список.

    only_relevant=False — все найденные фрагменты по убыванию близости (Top-k как есть), без порога релевантности.
    """
    try:
        result = search(query)
    except Exception as exc:  # noqa: BLE001 — база знаний недоступна: планируем без неё
        logger.warning("Поиск по запросу не удался: %s", exc)
        return []
    return [ContextFragment(source=h.source, text=h.text[:MAX_QUOTE], score=h.score) for h in result.hits if h.relevant or not only_relevant]


def retrieve_business_context(
    analyses: list[str], search: Callable[[str], object], metrics: dict[str, str] | None = None
) -> BusinessContext:
    """search(query) -> ContextSearchResult. Ошибки поиска не роняют анализ: контекст помечается недоступным."""
    context = BusinessContext()
    try:
        for analysis in analyses:
            kind = BY_ID[analysis]
            metric = (metrics or {}).get(analysis) or kind.metric  # у универсальных анализов показатель зависит от набора данных
            if metric in context.definitions:
                continue
            found = find_definition(search, doc_query(kind, metric), doc_terms(kind, metric))
            if found:
                context.definitions[metric] = found
        result = search(CONTEXT_QUERY)
        context.provider = result.provider
        taken = {f.chunk_id for f in context.definitions.values()}
        context.events = [
            ContextFragment(source=h.source, text=h.text[:MAX_QUOTE], score=h.score)
            for h in result.hits if h.relevant and h.source.chunk_id not in taken
        ][:MAX_EVENTS]
    except Exception as exc:  # noqa: BLE001 — сбой RAG (ключ, индекс) не должен останавливать анализ
        logger.warning("RAG недоступен: %s", exc)
        message = getattr(exc, "user_message", None) or str(exc)
        return BusinessContext(error=f"База знаний недоступна: {message}")
    return context


def make_searcher(kb: KnowledgeBase | None, dataset_id: str, top_k: int | None = None) -> Callable[[str], object]:
    from datastory.rag.api import search_business_context

    if top_k is None:
        return lambda query: search_business_context(query, dataset_id=dataset_id, kb=kb)
    return lambda query: search_business_context(query, top_k, dataset_id=dataset_id, kb=kb)
