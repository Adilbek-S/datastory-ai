"""Происхождение утверждений: факт из документа, результат вычисления, предположение модели.

Правило: утверждение можно назвать «фактом из документа», только если к нему приложен источник
и дословная цитата, которая действительно есть в найденном фрагменте. Иначе оно понижается
до предположения (validate_evidence). Так модель не сможет выдать догадку за подтверждённый факт.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from datastory.models import EvidenceType
from datastory.rag.models import ContextHit, ContextSearchResult, SourceReference

EVIDENCE_LABELS = {
    EvidenceType.DOCUMENT_FACT: "Факт из документа",
    EvidenceType.COMPUTED: "Результат вычисления",
    EvidenceType.ASSUMPTION: "Предположение",
}

# Правила для системного промпта LLM (вставляются вместе с format_context_for_prompt).
GROUNDING_RULES = (
    "Каждое утверждение помечай одним из типов:\n"
    "- document_fact: только если оно прямо сказано во фрагментах ниже; приведи id фрагмента и дословную цитату;\n"
    "- computed_result: значение получено вычислением по данным; укажи формулу или способ расчёта;\n"
    "- model_assumption: всё остальное (гипотезы, причины, объяснения). Не выдавай предположение за факт.\n"
    "Если во фрагментах нет ответа, так и скажи. Наличие события во времени не доказывает причинную связь."
)


# Причинные формулировки: «вызвано», «из-за», «привело к», «caused by»…
CAUSAL_CLAIM = re.compile(
    r"вызван|вызвал|из-за|причин|привел|привёл|следстви|по вине|обусловлен|результат[еа]?\s+обновлен"
    r"|caused|because|due to|led to|result of",
    re.IGNORECASE,
)


class EvidenceItem(BaseModel):
    type: EvidenceType
    statement: str
    sources: list[SourceReference] = Field(default_factory=list)
    quote: str | None = None  # дословная цитата — обязательна для факта из документа
    computation: str | None = None  # формула или способ расчёта — обязателен для вычисления
    note: str = ""

    @property
    def label(self) -> str:
        return EVIDENCE_LABELS[self.type]


def _normalize(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("«", '"').replace("»", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _downgrade(item: EvidenceItem, reason: str) -> EvidenceItem:
    return item.model_copy(
        update={"type": EvidenceType.ASSUMPTION, "note": f"{item.note} Понижено до предположения: {reason}".strip()}
    )


def validate_evidence(items: list[EvidenceItem], context: dict[str, str]) -> list[EvidenceItem]:
    """Проверяет типы утверждений.

    context: {chunk_id: текст фрагмента}, найденный поиском (см. context_from_result).
    Факт из документа без источника или с цитатой, которой нет в источнике, становится предположением.
    Вычисление без указанной формулы или способа расчёта тоже.
    """
    checked: list[EvidenceItem] = []
    for item in items:
        if item.type is EvidenceType.DOCUMENT_FACT:
            quote = _normalize(item.quote or "")
            if not item.sources:
                item = _downgrade(item, "не указан источник")
            elif not quote:
                item = _downgrade(item, "нет дословной цитаты")
            elif not any(quote in _normalize(context.get(src.chunk_id, "")) for src in item.sources):
                item = _downgrade(item, "цитата не найдена в указанном фрагменте")
            elif CAUSAL_CLAIM.search(item.statement) and not CAUSAL_CLAIM.search(quote):
                # Документ упомянул событие, но не утверждал, что оно причина: это вывод модели.
                item = _downgrade(item, "утверждение о причине не содержится в цитате")
        elif item.type is EvidenceType.COMPUTED and not (item.computation or "").strip():
            item = _downgrade(item, "не указан способ вычисления")
        checked.append(item)
    return checked


def context_from_result(result: ContextSearchResult) -> dict[str, str]:
    return {hit.source.chunk_id: hit.text for hit in result.hits}


def format_context_for_prompt(hits: list[ContextHit]) -> str:
    """Фрагменты для промпта: у каждого id и источник, чтобы модель могла на них сослаться."""
    if not hits:
        return "Фрагменты документов не найдены."
    return "\n\n".join(f"[{h.source.chunk_id}] ({h.source.citation})\n{h.text}" for h in hits)


def render_evidence(items: list[EvidenceItem]) -> str:
    """Текст с явными пометками типа утверждения."""
    lines = []
    for item in items:
        basis = ""
        if item.type is EvidenceType.DOCUMENT_FACT and item.sources:
            basis = " — " + "; ".join(s.citation for s in item.sources)
        elif item.type is EvidenceType.COMPUTED and item.computation:
            basis = f" — {item.computation}"
        lines.append(f"[{item.label}] {item.statement}{basis}")
    return "\n".join(lines)
