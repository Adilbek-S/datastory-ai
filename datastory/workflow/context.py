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
from datastory.workflow.catalog import BY_ID

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
    definitions: dict[str, ContextFragment] = Field(default_factory=dict, description="metric -> определение из документации")
    events: list[ContextFragment] = Field(default_factory=list, description="Сопроводительный контекст (события, ограничения)")
    provider: str = ""
    error: str | None = None  # база знаний недоступна: все показатели считаются недокументированными

    def fragment(self, chunk_id: str) -> ContextFragment | None:
        found = [*self.definitions.values(), *self.events]
        return next((f for f in found if f.chunk_id == chunk_id), None)


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def find_definition(search: Callable[[str], object], query: str, terms: tuple[str, ...]) -> ContextFragment | None:
    """Первый релевантный фрагмент, который называет показатель; иначе None (документации нет)."""
    result = search(query)
    for hit in result.hits:
        if hit.relevant and any(_norm(term) in _norm(hit.text) for term in terms):
            return ContextFragment(source=hit.source, text=hit.text[:MAX_QUOTE], score=hit.score)
    return None


def retrieve_business_context(
    analyses: list[str], search: Callable[[str], object]
) -> BusinessContext:
    """search(query) -> ContextSearchResult. Ошибки поиска не роняют анализ: контекст помечается недоступным."""
    context = BusinessContext()
    try:
        for analysis in analyses:
            kind = BY_ID[analysis]
            if kind.metric in context.definitions:
                continue
            found = find_definition(search, kind.doc_query, kind.doc_terms)
            if found:
                context.definitions[kind.metric] = found
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


def make_searcher(kb: KnowledgeBase | None, dataset_id: str) -> Callable[[str], object]:
    from datastory.rag.api import search_business_context

    return lambda query: search_business_context(query, dataset_id=dataset_id, kb=kb)
