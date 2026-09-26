"""Сбор доказательств для ответа в чате: результаты MCP, определения и события из RAG; ответы по правилам.

Числа берутся только из результатов calculate_metrics: готовых с дашборда или, если нужного расчёта там нет, из нового
вызова MCP. Сравнения (разности, доли) считает Python, а не LLM. Бизнес-контекст всегда приходит из RAG с источниками.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from datastory.analytics.formatting import format_metric_value
from datastory.analytics.metrics import METRICS
from datastory.analytics.models import MetricResult
from datastory.chat.models import ChatAnswer, ChatIntent
from datastory.insights.facts import build_facts, group_labels
from datastory.insights.generator import (
    CAUSE_UNKNOWN,
    EVENT_NOT_CAUSE,
    NO_DEFINITION,
    ChartInsightInput,
    rules_time_series,
)
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.models import NumericEvidence
from datastory.workflow.catalog import ANALYSES, category_columns, time_columns
from datastory.workflow.context import CONTEXT_QUERY, ContextFragment, find_definition
from datastory.workflow.models import AnalysisResult, ColumnMapping, PlannedChart

TIME_INTENTS = ("metric_change", "extreme_period", "notable_changes", "cause_question")
MAX_EVENTS = 3
MAX_DEFINITION_SENTENCES = 2
# предложение о событии: называет событие и относит его ко времени («В марте 2026 года проводилось обновление…»)
EVENT_WORD = re.compile(r"событи|обновлен|проводил|инцидент|сбо[йя]|миграц|релиз", re.IGNORECASE)
WHEN_WORD = re.compile(r"\b20\d\d\b|январ|феврал|март|апрел|\bма[йяе]\b|июн|июл|август|сентябр|октябр|ноябр|декабр", re.IGNORECASE)
SENTENCE = re.compile(r"(?<=[.!?])\s+")
NOTABLE_PP = 2.0  # для процентов: существенное изменение в п.п.
NOTABLE_REL = 0.10  # для сумм и количеств: доля от максимума ряда
EXTRA_TERMS = {
    "successful_count": ("successful", "успешно завершённых"),
    "failed_count": ("failed", "неуспешных транзакц"),
    "average_transaction_amount": ("average transaction amount", "средняя сумма транзакции"),
}


@dataclass
class ChatDeps:
    """Что нужно сбору доказательств: MCP-клиент, поиск по базе знаний и результат текущего дашборда."""

    client: AnalyticsMcpClient
    search: Callable[[str], object]  # запрос -> ContextSearchResult
    result: AnalysisResult


@dataclass
class Bundle:
    intent: ChatIntent
    metrics: list[str] = field(default_factory=list)
    inputs: dict[str, ChartInsightInput] = field(default_factory=dict)
    facts: list[NumericEvidence] = field(default_factory=list)  # id вида «success_rate.max»
    definitions: dict[str, ContextFragment | None] = field(default_factory=dict)
    events: list[ContextFragment] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    clarification: str | None = None  # уточняющий вопрос или пояснение: LLM не нужен
    labels: list[str] = field(default_factory=list)
    rag_error: str | None = None

    @property
    def fragments(self) -> dict[str, ContextFragment]:
        found = [*(f for f in self.definitions.values() if f), *self.events]
        return {f.chunk_id: f for f in found}

    def mask(self) -> list[str]:
        names = [x for i in self.inputs.values() for x in i.mask()]
        quotes = [s for f in self.fragments.values() for s in sentences(f.text)]  # цитаты из документа — не «числа вывода»
        cites = [f.citation for f in self.fragments.values()]  # «стр. 1, раздел «5. …»» — номера страниц и разделов
        formulas = [x for m in self.metrics for x in (METRICS[m].formula, METRICS[m].label)]
        years = [y for label in self.labels for y in re.findall(r"20\d\d", label)]  # «в марте 2026 года» — год из подписи периода
        return [*names, *quotes, *cites, *formulas, *self.labels, *years]


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(" ".join(text.split())) if s.strip()]


def definition_terms(metric: str) -> tuple[str, ...]:
    kind = next((a for a in ANALYSES if a.metric == metric), None)
    return kind.doc_terms if kind else EXTRA_TERMS[metric]


def definition_query(metric: str) -> str:
    kind = next((a for a in ANALYSES if a.metric == metric), None)
    return kind.doc_query if kind else f"{METRICS[metric].label} определение формула"


# --------------------------------------------------------------------------- расчёты (дашборд или новый вызов MCP)
def dashboard_metrics(result: AnalysisResult, role: str) -> list[str]:
    plan = result.plan
    charts = [c for c in (plan.charts if plan else []) if c.mapping.role == role]
    return list(dict.fromkeys(c.metric for c in charts))


def pick_dimension(result: AnalysisResult, role: str) -> str | None:
    for chart in (result.plan.charts if result.plan else []):
        if chart.mapping.role == role and chart.x_column:
            return chart.x_column
    found = time_columns(result.summary) if role == "time" else category_columns(result.summary)
    return found[0] if found else None


def get_result(deps: ChatDeps, metric: str, group_by: list[str], bundle: Bundle) -> MetricResult:
    """Результат с дашборда, если он есть; иначе новый расчёт на MCP-сервере."""
    for existing in deps.result.metrics:
        if existing.metric == metric and existing.group_by == group_by and not existing.filters:
            bundle.data_sources.append(f"результат дашборда: {existing.label} = {existing.formula}; группировка по {', '.join(group_by)}")
            return existing
    fresh = deps.client.calculate_metrics(deps.result.dataset_id, metric, group_by)
    bundle.tools.append("MCP calculate_metrics")
    bundle.data_sources.append(f"MCP calculate_metrics: {fresh.label} = {fresh.formula}; группировка по {', '.join(group_by)}")
    return fresh


def _metric_input(deps: ChatDeps, metric: str, role: str, dimension: str, bundle: Bundle) -> ChartInsightInput:
    result = get_result(deps, metric, [dimension], bundle)
    chart = PlannedChart(
        chart_id=f"chat-{metric}", analysis="count_dynamics" if role == "time" else "channel_distribution", title=result.label, metric=metric,
        chart_type="line" if role == "time" else "bar", x_column=dimension,
        mapping=ColumnMapping(role=role, column=dimension, candidates=[dimension], status="confident"),
    )
    plan = deps.result.plan
    return ChartInsightInput(
        chart=chart, result=result, facts=build_facts(result, role), definition=plan.metric_definition(metric) if plan else None,
        events=[], filename=deps.result.summary.filename,
    )


def _add(bundle: Bundle, inp: ChartInsightInput) -> None:
    metric = inp.chart.metric
    bundle.metrics.append(metric)
    bundle.inputs[metric] = inp
    bundle.facts += [f.model_copy(update={"id": f"{metric}.{f.id}"}) for f in inp.facts]
    bundle.labels += inp.labels()


def _search_events(deps: ChatDeps, question: str, bundle: Bundle) -> None:
    """RAG: сопроводительный контекст (события, ограничения) с источниками."""
    bundle.tools.append("RAG search_business_context")
    try:
        seen: dict[str, ContextFragment] = {}
        for query in (f"{question} {CONTEXT_QUERY}", question):
            for hit in deps.search(query).hits:
                if hit.relevant and hit.source.chunk_id not in seen:
                    seen[hit.source.chunk_id] = ContextFragment(source=hit.source, text=hit.text, score=hit.score)
        bundle.events = sorted(seen.values(), key=lambda f: f.score, reverse=True)[:MAX_EVENTS]
    except Exception as exc:  # noqa: BLE001 — недоступная база знаний = «документации нет», а не сбой чата
        bundle.rag_error = getattr(exc, "user_message", None) or str(exc)


def _search_definitions(deps: ChatDeps, metrics: list[str], bundle: Bundle) -> None:
    bundle.tools.append("RAG search_business_context")
    for metric in metrics:
        try:
            bundle.definitions[metric] = find_definition(deps.search, definition_query(metric), definition_terms(metric))
        except Exception as exc:  # noqa: BLE001
            bundle.definitions[metric] = None
            bundle.rag_error = getattr(exc, "user_message", None) or str(exc)


def gather(intent: ChatIntent, metric: str | None, question: str, deps: ChatDeps) -> Bundle:
    """Доказательства под тип вопроса. Бросает McpConnectionError / McpToolError, если сервер недоступен."""
    result = deps.result
    bundle = Bundle(intent=intent, labels=[str(v) for r in result.metrics for v in group_labels(r)])
    available = result.summary.available_metrics

    if intent == "metric_definition":
        targets = [metric] if metric else list(dict.fromkeys([*dashboard_metrics(result, "time"), *dashboard_metrics(result, "category")]))
        if not targets:
            bundle.clarification = "Уточните, определение какого показателя нужно: " + ", ".join(METRICS[m].label for m in available) + "."
            return bundle
        bundle.metrics = targets
        _search_definitions(deps, targets, bundle)
        return bundle

    if intent == "context_events":
        _search_events(deps, question, bundle)
        return bundle

    if intent == "top_category":
        target = metric or next(iter(dashboard_metrics(result, "category")), None) or ("transaction_count" if "transaction_count" in available else None)
        dimension = pick_dimension(result, "category")
        if target is None or dimension is None:
            bundle.clarification = "В датасете нет категориальной колонки (например, канала) или показателя количества операций для такого сравнения."
            return bundle
        _add(bundle, _metric_input(deps, target, "category", dimension, bundle), )
        return bundle

    # вопросы о динамике: metric_change, extreme_period, notable_changes, cause_question
    targets = [metric] if metric else dashboard_metrics(result, "time")
    dimension = pick_dimension(result, "time")
    if dimension is None:
        bundle.clarification = "В датасете нет колонки с датой или периодом, поэтому динамику и максимум по месяцам показать нельзя."
        return bundle
    if not targets:
        bundle.clarification = "Уточните показатель: " + ", ".join(METRICS[m].label for m in available) + "."
        return bundle
    for target in targets:
        _add(bundle, _metric_input(deps, target, "time", dimension, bundle))
    if intent == "cause_question":
        _search_events(deps, question, bundle)
    return bundle


# --------------------------------------------------------------------------- ограничения и ответы по правилам
def _changes(inp: ChartInsightInput) -> bool:
    facts = {f.id: f for f in inp.facts}
    return any(k in facts and facts[k].value != 0 for k in ("change", "drop", "rise"))


def chat_limitations(bundle: Bundle, event_used: bool) -> list[str]:
    limits: list[str] = []
    if bundle.intent == "metric_definition":
        for metric, fragment in bundle.definitions.items():
            if fragment is None:
                limits.append(NO_DEFINITION.format(formula=METRICS[metric].formula))
    if bundle.intent == "cause_question" or (
        bundle.intent in ("metric_change", "notable_changes") and any(_changes(i) for i in bundle.inputs.values())
    ):
        limits.append(CAUSE_UNKNOWN)
    if event_used and bundle.inputs:
        limits.append(EVENT_NOT_CAUSE)
    if bundle.rag_error:
        limits.append(f"База знаний недоступна ({bundle.rag_error}): бизнес-контекст не использован.")
    return limits


def is_notable(inp: ChartInsightInput) -> list[str]:
    """id доказательств существенных изменений ряда: большое падение или рост, либо общее изменение."""
    facts = {f.id: f for f in inp.facts}
    top = facts.get("max")
    found = []
    for key in ("drop", "rise", "change"):
        fact = facts.get(key)
        if fact is None or fact.value == 0:
            continue
        big = abs(fact.value) >= NOTABLE_PP if inp.result.unit == "%" else bool(top and top.value and abs(fact.value) / abs(top.value) >= NOTABLE_REL)
        if big:
            found.append(key)
    return found


def _by_id(bundle: Bundle) -> dict[str, NumericEvidence]:
    return {f.id: f for f in bundle.facts}


def _fmt(fact: NumericEvidence) -> str:
    return fact.formatted


def _event_sentences(fragment: ContextFragment) -> list[str]:
    return [x for x in sentences(fragment.text) if EVENT_WORD.search(x) and WHEN_WORD.search(x)]


def _end(text: str) -> str:
    return text if text.endswith(".") else text + "."


def _relevant_sentences(fragment: ContextFragment, words: tuple[str, ...]) -> list[str]:
    return [s for s in sentences(fragment.text) if any(_norm(w) in _norm(s) for w in words)]


def rules_answer(question: str, bundle: Bundle) -> tuple[str, list[str], list[ContextFragment]]:
    """Ответ по правилам: (текст, id использованных доказательств, использованные фрагменты документов)."""
    facts, intent = _by_id(bundle), bundle.intent
    parts: list[str] = []
    cited: list[str] = []
    used: list[ContextFragment] = []

    if intent == "metric_definition":
        for metric in bundle.metrics:
            fragment, definition = bundle.definitions.get(metric), METRICS[metric]
            if fragment:
                quotes = _relevant_sentences(fragment, definition_terms(metric))[:MAX_DEFINITION_SENTENCES] or sentences(fragment.text)[:1]
                parts.append(f"«{definition.label}» — по документации ({fragment.citation}): «{' '.join(quotes)}» Формула расчёта в системе: {definition.formula}.")
                used.append(fragment)
            else:
                parts.append(f"Определение показателя «{definition.label}» в документации не найдено. В анализе используется формула: {definition.formula}.")
        return " ".join(parts), cited, used

    if intent == "context_events":
        for fragment in bundle.events:
            picked = _event_sentences(fragment)
            if picked:
                parts.append(f"В документации ({fragment.citation}) упоминается: «{' '.join(picked[:2])}»")
                used.append(fragment)
        if not parts:
            return "В сопроводительной документации события не найдены: документ не загружен или не содержит описания событий.", cited, used
        return " ".join(parts) + " Совпадение по времени не доказывает причинную связь.", cited, used

    if intent == "top_category":
        metric = bundle.metrics[0]
        inp = bundle.inputs[metric]
        shares = sorted((f for f in inp.facts if f.kind == "share"), key=lambda f: f.value, reverse=True)
        if not shares:
            return f"Для «{inp.result.label}» нет значений по категориям.", cited, used
        ranking = []
        for share in shares:
            value = next(f for f in inp.facts if f.id == share.id.replace("_share", "_value"))
            ranking.append(f"«{share.group}» — {value.formatted} ({share.formatted})")
            cited += [f"{metric}.{value.id}", f"{metric}.{share.id}"]
        top = shares[0]
        top_value = next(f for f in inp.facts if f.id == top.id.replace("_share", "_value"))
        parts.append(
            f"Наибольшее значение «{inp.result.label}» у «{top.group}»: {top_value.formatted} ({top.formatted} от итога). "
            f"По всем категориям: {'; '.join(ranking)}."
        )
        return " ".join(parts), cited, used

    for metric in bundle.metrics:  # вопросы о динамике
        inp = bundle.inputs[metric]
        local = {f.id: f for f in inp.facts}
        label = inp.result.label
        if intent == "extreme_period":
            if "max" not in local:
                parts.append(f"«{label}» есть только за один период: сравнивать максимум и минимум не с чем.")
                continue
            hi, lo = local["max"], local["min"]
            parts.append(f"«{label}»: максимум — {hi.formatted} (период {hi.group}), минимум — {lo.formatted} (период {lo.group}).")
            cited += [f"{metric}.max", f"{metric}.min"]
        elif intent == "notable_changes":
            keys = is_notable(inp)
            if not keys:
                if "min" in local:
                    parts.append(f"«{label}»: заметных изменений нет, значения держатся в диапазоне {local['min'].formatted} — {local['max'].formatted}.")
                    cited += [f"{metric}.min", f"{metric}.max"]
                continue
            phrases = []
            for key in keys:
                fact = local[key]
                what = "падение" if fact.value < 0 else "рост"
                where = f"между периодами {fact.group}" if key != "change" else f"за весь ряд ({fact.group})"
                phrases.append(f"{what} {where} на {fact.formatted}")
                cited.append(f"{metric}.{key}")
            parts.append(_end(f"«{label}»: " + "; ".join(phrases)))
        else:  # metric_change, cause_question
            _, text, ids = rules_time_series(inp)
            parts.append(text)
            cited += [f"{metric}.{i}" for i in ids]
    if intent == "cause_question":
        head = "Причина изменения по данным не определяется: расчёты показывают только сам факт изменения."
        events = []
        for fragment in bundle.events:
            picked = _event_sentences(fragment)
            if picked:
                events.append(f"В документации ({fragment.citation}) отмечено: «{picked[0]}» — это совпадает по времени, но остаётся гипотезой.")
                used.append(fragment)
        parts = [head, *parts, *events[:1]]
    if not parts:
        parts.append("Существенных изменений показателей в дашборде не найдено.")
    return " ".join(parts), cited, used


def answer_from_rules(question: str, bundle: Bundle, classified_by: str = "rules") -> ChatAnswer:
    text, cited, used = rules_answer(question, bundle)
    facts = _by_id(bundle)
    event_used = bool(used) and bundle.intent in ("cause_question", "notable_changes", "metric_change")
    return ChatAnswer(
        question=question, intent=bundle.intent, answer=text, metrics=bundle.metrics,
        evidence=[facts[i] for i in dict.fromkeys(cited) if i in facts], data_sources=list(dict.fromkeys(bundle.data_sources)),
        context_source="; ".join(dict.fromkeys(f.citation for f in used)) or None,
        limitation=" ".join(chat_limitations(bundle, event_used)) or None, tools=list(dict.fromkeys(bundle.tools)),
        generated_by="rules", classified_by=classified_by,
    )
