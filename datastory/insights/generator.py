"""Генерация выводов по графикам: LLM со структурным ответом или детерминированные правила.

LLM пишет только текст (название, краткий вывод, оговорку) и называет id числовых доказательств. Сами доказательства,
источник данных, источник контекста и обязательные ограничения интерпретации добавляет код: модель не может подменить
числа или источник. Полученный вывод затем проверяет verifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from datastory.analytics.models import MetricResult
from datastory.insights.facts import group_labels
from datastory.insights.verifier import CONTEXT_NOT_IN_SOURCE, EVIDENCE_REQUIRED, UNKNOWN_EVIDENCE, Violation
from datastory.llm.client import StructuredLLM
from datastory.models import EvidenceType, Insight, NumericEvidence
from datastory.rag.evidence import EvidenceItem, validate_evidence
from datastory.workflow.context import ContextFragment
from datastory.workflow.models import MetricDefinition, PlannedChart

DROP_THRESHOLD_PP = 2.0  # существенное падение доли в процентных пунктах
DROP_THRESHOLD_REL = 0.10  # существенное падение суммы: доля от максимума ряда
MAX_TITLE = 120

NO_DEFINITION = "Определение показателя в документации не найдено; использовано правило расчёта, подтверждённое пользователем: {formula}."
CAUSE_UNKNOWN = "Причина изменения по данным не установлена: расчёт фиксирует только сам факт изменения значений."
EVENT_NOT_CAUSE = "Событие из документации совпадает по времени с изменением, но это не доказывает причинную связь."


class InsightDraft(BaseModel):
    """Структурный ответ LLM: текст и ссылки на доказательства, но не сами числа как данные."""

    title: str = Field(description="Название вывода, до 100 символов")
    summary: str = Field(description="Краткий вывод, 1–3 предложения; числа только из числовых доказательств")
    severity: Literal["info", "warning", "positive"]
    evidence_ids: list[str] = Field(description="id числовых доказательств, на которые опирается вывод (хотя бы один)")
    limitation: str | None = Field(description="Ограничение интерпретации или null")
    context_chunk_id: str | None = Field(description="chunk_id фрагмента документа, если в выводе использован контекст, иначе null")
    context_quote: str | None = Field(description="Дословная цитата из этого фрагмента или null")


@dataclass
class ChartInsightInput:
    chart: PlannedChart
    result: MetricResult
    facts: list[NumericEvidence]
    definition: MetricDefinition | None
    events: list[ContextFragment]
    filename: str
    definition_fragment: ContextFragment | None = None
    feedback: list[str] = field(default_factory=list)

    @property
    def role(self) -> str:
        return self.chart.mapping.role

    @property
    def fragments(self) -> dict[str, ContextFragment]:
        found = [*self.events, *([self.definition_fragment] if self.definition_fragment else [])]
        return {f.chunk_id: f for f in found}

    def labels(self) -> list[str]:
        return group_labels(self.result)

    def mask(self) -> list[str]:
        """Подписи групп, названия и формулы: цифры в них не считаются числами вывода."""
        labels = group_labels(self.result)
        years = [y for label in [*labels, self.filename] for y in re.findall(r"\d{4}", label)]  # «в марте 2026 года» — год из подписи периода или имени файла
        names = [self.chart.title, self.result.label, self.result.formula, self.filename, *labels, *self.result.group_by, *years]
        return [n for n in names if n]


def data_source(inp: ChartInsightInput) -> str:
    by = ", ".join(inp.result.group_by)
    return f"MCP calculate_metrics: {inp.result.label} = {inp.result.formula}; группировка по {by}; датасет {inp.filename}"


def _sentence(text: str) -> str:
    return text if text.endswith(".") else text + "."  # «п.п.» уже заканчивается точкой


def _by_id(facts: list[NumericEvidence]) -> dict[str, NumericEvidence]:
    return {f.id: f for f in facts}


def system_limitations(inp: ChartInsightInput, event_used: bool) -> list[str]:
    """Ограничения, которые обязательны независимо от текста модели."""
    limits = []
    if inp.definition is not None and not inp.definition.documented:
        limits.append(NO_DEFINITION.format(formula=inp.definition.formula))
    facts = _by_id(inp.facts)
    changed = inp.role == "time" and any(k in facts and facts[k].value != 0 for k in ("change", "drop", "rise"))
    if changed:
        limits.append(CAUSE_UNKNOWN)
    if event_used:
        limits.append(EVENT_NOT_CAUSE)
    return limits


def _context_source(inp: ChartInsightInput, event: ContextFragment | None) -> str | None:
    cited = [f.citation for f in (inp.definition_fragment, event) if f is not None]
    return "; ".join(dict.fromkeys(cited)) or None


# --------------------------------------------------------------------------- качество данных
def generate_insights(summary) -> list[Insight]:
    """Замечания о качестве данных (вычисление по сводке). summary — DatasetSummary или DatasetProfile."""
    insights: list[Insight] = []
    if summary.duplicate_row_count:
        insights.append(
            Insight(title="Найдены дубликаты", text=f"В данных {summary.duplicate_row_count} повторяющихся строк.", severity="warning")
        )
    if summary.missing_cell_count:
        insights.append(
            Insight(title="Есть пропуски", text=f"Всего пропущенных ячеек: {summary.missing_cell_count}.", severity="warning")
        )
    if not insights:
        insights.append(Insight(title="Данные чистые", text="Пропусков и дубликатов не найдено.", severity="positive"))
    return insights


# --------------------------------------------------------------------------- правила (без LLM)
def material_drop(inp: ChartInsightInput, fact: NumericEvidence | None) -> bool:
    if fact is None or fact.value >= 0:
        return False
    if inp.result.unit == "%":
        return abs(fact.value) >= DROP_THRESHOLD_PP
    top = _by_id(inp.facts).get("max")
    return top is not None and top.value != 0 and abs(fact.value) / abs(top.value) >= DROP_THRESHOLD_REL


def rules_time_series(inp: ChartInsightInput) -> tuple[str, str, list[str]]:
    facts, label = _by_id(inp.facts), inp.result.label
    first, last = facts.get("first"), facts.get("last")
    if first is None:
        return "info", f"Результат расчёта «{label}» не содержит значений по периодам.", []
    if last is None:
        return "info", f"«{label}» есть только за один период ({first.group}): {first.formatted}. Для динамики нужно не менее двух периодов.", ["first"]

    change = facts["change"]
    sentences = [f"«{label}» в периоде {first.group} — {first.formatted}, в периоде {last.group} — {last.formatted}."]
    cited = ["first", "last", "change"]
    if change.value == 0:
        sentences.append("За весь ряд значение не изменилось.")
    else:
        verb = "выросло" if change.value > 0 else "снизилось"
        pct = facts.get("change_pct")
        tail = f" ({pct.formatted})" if pct else ""
        sentences.append(_sentence(f"За весь ряд значение {verb} на {change.formatted}{tail}"))
        if pct:
            cited.append("change_pct")
    sentences.append(
        f"Минимум — {facts['min'].formatted} (период {facts['min'].group}), максимум — {facts['max'].formatted} (период {facts['max'].group})."
    )
    cited += ["min", "max"]
    severity = "info"
    drop = facts.get("drop")
    if material_drop(inp, drop):
        severity = "warning"
        sentences.append(_sentence(f"Наибольшее падение между соседними периодами: {drop.group}, на {drop.formatted}"))
        cited.append("drop")
    return severity, " ".join(sentences), cited


def rules_category(inp: ChartInsightInput) -> tuple[str, str, list[str]]:
    facts, label = _by_id(inp.facts), inp.result.label
    shares = sorted((f for f in inp.facts if f.kind == "share"), key=lambda f: f.value, reverse=True)
    total = facts.get("overall")
    if not shares or total is None:
        return "info", f"Для «{label}» нет значений по категориям.", []

    def value_of(share: NumericEvidence) -> NumericEvidence:
        return facts[share.id.replace("_share", "_value")]  # значение той же категории

    top, bottom = shares[0], shares[-1]
    sentences = [f"«{label}» по категориям: наибольшая доля у «{top.group}» — {top.formatted} ({value_of(top).formatted})."]
    cited = [total.id, top.id, value_of(top).id]
    if bottom is not top:
        sentences.append(f"Наименьшая — у «{bottom.group}»: {bottom.formatted} ({value_of(bottom).formatted}).")
        cited += [bottom.id, value_of(bottom).id]
    sentences.append(f"Итог по всем категориям — {total.formatted}.")
    return "info", " ".join(sentences), cited


def rules_insight(inp: ChartInsightInput) -> Insight:
    severity, text, cited = rules_time_series(inp) if inp.role == "time" else rules_category(inp)
    facts = _by_id(inp.facts)
    limits = system_limitations(inp, event_used=False)
    return Insight(
        title=inp.chart.title[:MAX_TITLE], text=text, severity=severity, evidence_type=EvidenceType.COMPUTED,
        chart_id=inp.chart.chart_id, chart_title=inp.chart.title, evidence=[facts[i] for i in cited if i in facts],
        data_source=data_source(inp), context_source=_context_source(inp, None),
        limitation=" ".join(limits) or None, generated_by="rules",
    )


# --------------------------------------------------------------------------- LLM
def insight_prompt(inp: ChartInsightInput) -> str:
    definition = inp.definition
    if definition is not None and definition.documented and inp.definition_fragment:
        defined = f"найдено в документации [{inp.definition_fragment.chunk_id}] ({inp.definition_fragment.citation}): «{inp.definition_fragment.text}»"
    else:
        defined = "в документации НЕ найдено; используйте только формулу расчёта, подтверждённую пользователем"
    labels = ", ".join(inp.labels())
    scope = (
        f"Периоды на оси X: {labels}. Не упоминай периоды, которых нет в этом списке."
        if inp.role == "time"
        else f"Значения по категориям ({labels}) посчитаны по всем строкам датасета без фильтра по периоду: не называй месяцы и даты."
    )
    events = "\n".join(f"[{f.chunk_id}] ({f.citation}): «{f.text}»" for f in inp.events) or "нет"
    facts = "\n".join(f"{f.id} | {f.label} | {f.formatted}" for f in inp.facts)
    parts = [
        f"График: «{inp.chart.title}» ({inp.chart.chart_type}); ось X — «{inp.chart.x_column}»; показатель «{inp.result.label}», единица: {inp.result.unit}.",
        f"Формула расчёта (выполнена инструментом calculate_metrics): {inp.result.formula}.",
        scope,
        f"Бизнес-определение показателя: {defined}.",
        f"Сопроводительный контекст из документов:\n{events}",
        f"Числовые доказательства (id | описание | значение) — единственный допустимый источник чисел:\n{facts}",
    ]
    if inp.feedback:
        parts.append("Предыдущий ответ отклонён проверкой, исправьте:\n" + "\n".join(f"- {v}" for v in inp.feedback))
    return "\n\n".join(parts)


def quote_is_verbatim(fragment: ContextFragment, quote: str) -> bool:
    """Цитата из документа допустима, только если она дословно есть во фрагменте (правило A7)."""
    checked = validate_evidence(
        [EvidenceItem(type=EvidenceType.DOCUMENT_FACT, statement=quote, sources=[fragment.source], quote=quote)],
        {fragment.chunk_id: fragment.text},
    )[0]
    return checked.type is EvidenceType.DOCUMENT_FACT


def assemble(draft: InsightDraft, inp: ChartInsightInput) -> tuple[Insight, list[Violation]]:
    """Собирает Insight из ответа LLM: доказательства и источники берутся из кода, а не из текста модели."""
    issues: list[Violation] = []
    facts = _by_id(inp.facts)
    evidence = []
    for fact_id in dict.fromkeys(draft.evidence_ids):
        if fact_id in facts:
            evidence.append(facts[fact_id])
        else:
            issues.append(Violation(UNKNOWN_EVIDENCE, f"доказательство {fact_id!r} не входит в набор фактов графика"))
    if not evidence and not any(v.rule == UNKNOWN_EVIDENCE for v in issues):
        issues.append(Violation(EVIDENCE_REQUIRED, "не указано ни одного числового доказательства"))

    event = None
    if draft.context_chunk_id:
        fragment = inp.fragments.get(draft.context_chunk_id)
        quote = (draft.context_quote or "").strip()
        if fragment is None:
            issues.append(Violation(CONTEXT_NOT_IN_SOURCE, f"фрагмент {draft.context_chunk_id!r} не был найден в базе знаний"))
        else:
            if quote_is_verbatim(fragment, quote):
                event = fragment
            else:
                issues.append(Violation(CONTEXT_NOT_IN_SOURCE, "цитата не найдена дословно в указанном фрагменте документа"))

    # событие «совпадает по времени» имеет смысл только для динамики: у распределения по категориям изменения нет
    event_used = event is not None and event is not inp.definition_fragment and inp.role == "time"
    own = [draft.limitation.strip()] if draft.limitation and draft.limitation.strip() else []
    limits = [*own, *system_limitations(inp, event_used)]
    insight = Insight(
        title=draft.title.strip()[:MAX_TITLE], text=draft.summary.strip(), severity=draft.severity,
        evidence_type=EvidenceType.COMPUTED, chart_id=inp.chart.chart_id, chart_title=inp.chart.title, evidence=evidence,
        data_source=data_source(inp), context_source=_context_source(inp, event if event_used else None),
        limitation=" ".join(dict.fromkeys(limits)) or None, generated_by="llm",
    )
    return insight, issues


def llm_insight(llm: StructuredLLM, inp: ChartInsightInput, system: str) -> tuple[Insight, list[Violation]]:
    """Бросает LLMError при сбое модели (вызывающий код переходит на правила)."""
    return assemble(llm.generate(InsightDraft, system=system, user=insight_prompt(inp)), inp)
