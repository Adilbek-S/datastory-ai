"""План по запросу пользователя для наборов без колонок платёжной системы (например, продаж).

LLM предлагает шаги (показатель, группировка, фильтры, тип графика), а здесь они проверяются по данным и метаданным:
показатель обязан быть среди доступных, колонки и значения фильтров — существовать, тип графика — соответствовать
правилам визуализации. Всё, чего нет (EBITDA, прибыль, чужая колонка), не выдумывается, а попадает в plan.unsupported
и исключается из плана. Если группировка неоднозначна, шаг помечается для уточнения у пользователя.
"""
from __future__ import annotations

import re
from typing import Callable

from datastory.analytics.metrics import SUM_PREFIX, get_metric
from datastory.analytics.models import DatasetSummary, FilterCondition
from datastory.workflow.catalog import PIE_MAX_CATEGORIES, category_columns, time_columns, unique_count
from datastory.workflow.context import BusinessContext, ContextFragment, find_definition
from datastory.workflow.models import (
    MAX_CHARTS,
    AnalysisPlan,
    AnalysisStep,
    ColumnMapping,
    ExcludedItem,
    FilterDraft,
    PlanDraft,
    StepDraft,
)
from datastory.workflow.planning import _clarifications, _metric_definition

MAX_GROUP_COLUMNS = 2
PLACEHOLDER_METRICS = {"", "unsupported", "none", "null", "n/a", "нет", "-"}
MAX_TITLE = 120
PERIOD_VALUE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")
DYNAMICS_WORDS = re.compile(r"рост|динамик|тренд|изменени|менял|растёт|растет|снижени|падени", re.IGNORECASE)
VAGUE_GROUPING = re.compile(r"по\s+(групп|разрез|срез|сегмент)", re.IGNORECASE)


def metric_definition_lookup(search: Callable[[str], object]) -> Callable[[str], ContextFragment | None]:
    """Поиск определения показателя в документах: фрагмент должен назвать ВСЕ колонки показателя (числитель и знаменатель)."""

    def lookup(metric: str) -> ContextFragment | None:
        definition = get_metric(metric)
        terms = tuple(column.casefold() for column in definition.columns)
        return find_definition(search, f"{definition.label} определение формула {definition.formula}", terms, require_all=True)

    return lookup


def _known_values(summary: DatasetSummary, column: str) -> list[str] | None:
    """Все значения категории, если профиль их перечислил; иначе None (проверить нельзя)."""
    col = next((c for c in summary.columns if c.name == column), None)
    top = ((col.statistics or {}).get("top_values") or {}) if col else {}
    return list(top) if col and top and len(top) >= col.unique_count else None


def _period_range(summary: DatasetSummary, column: str) -> tuple[str, str]:
    """Границы дат колонки с точностью до месяца («2026-01», «2026-06») по профилю; пусто, если профиль их не хранит."""
    col = next((c for c in summary.columns if c.name == column), None)
    stats = (col.statistics or {}) if col else {}
    low, high = str(stats.get("min") or ""), str(stats.get("max") or "")
    return (low[:7], high[:7]) if low and high else ("", "")


def _match_metric(name: str, available: list[str]) -> str | None:
    lowered = {m.casefold(): m for m in available}
    return lowered.get((name or "").strip().casefold())


def _match_column(name: str, columns: list[str]) -> str | None:
    lowered = {c.casefold(): c for c in columns}
    return lowered.get((name or "").strip().casefold())


def _filter(draft: FilterDraft, summary: DatasetSummary) -> tuple[FilterCondition | None, str]:
    """(условие, причина отказа). Значения сверяются с данными: регистр приводится к написанию в данных."""
    column = _match_column(draft.column, summary.dimensions)
    if column is None:
        return None, f"фильтр по колонке «{draft.column}»: такой колонки-измерения нет"
    if not draft.values:
        return None, f"фильтр по колонке «{column}» без значений"
    known = _known_values(summary, column)
    values = []
    for raw in draft.values:
        if summary.column_types[column] == "datetime":
            if not PERIOD_VALUE.match(raw.strip()):
                return None, f"значение «{raw}» не похоже на период (ожидается ГГГГ-ММ или ГГГГ-ММ-ДД)"
            low, high = _period_range(summary, column)
            if low and not (low <= raw.strip()[:7] <= high):
                return None, f"период «{raw}» вне диапазона данных ({low} — {high}): прогнозы и будущие периоды не строятся"
            values.append(raw.strip())
        elif known is not None:
            found = next((k for k in known if k.casefold() == raw.strip().casefold()), None)
            if found is None:
                return None, f"значения «{raw}» нет в колонке «{column}» (есть: {', '.join(known)})"
            values.append(found)
        else:
            values.append(raw.strip())
    single = draft.op in ("eq", "ne")
    return FilterCondition(column=column, op=draft.op, value=values[0] if single else values), ""


def fit_custom_chart(requested: str, group_by: list[str], metric: str, summary: DatasetSummary, filters: list[FilterCondition]) -> tuple[str, str]:
    """Тип графика по правилам: время — линия (V1, V4); категории — столбцы, круг только для долей небольшого числа категорий (V2, V3)."""
    times = set(time_columns(summary))
    if group_by and group_by[0] in times:
        return "line", ("" if requested == "line" else f"динамика по «{group_by[0]}» показывается линейным графиком, а не «{requested}»")
    categories = unique_count(summary, group_by[0])
    for condition in filters:
        if condition.column == group_by[0] and condition.op in ("eq", "in"):
            categories = len(condition.value) if isinstance(condition.value, list) else 1
    additive = metric.startswith(SUM_PREFIX)
    if requested == "pie" and len(group_by) == 1 and additive and categories <= PIE_MAX_CATEGORIES:
        return "pie", ""
    reason = "" if requested == "bar" else "сравнение категорий показывается столбчатым графиком"
    return "bar", reason


def term_conflict(term: str, metric: str, search: Callable[[str], object] | None) -> str:
    """Не подменил ли планировщик показатель, названный пользователем, другим.

    Если документы описывают названный термин, а описание не упоминает колонки выбранного показателя, это подмена
    (например, «Return Rate» описан в документе, но посчитан как доля отмен). Нет документа с термином — проверить нечем.
    """
    term = (term or "").strip()
    definition = get_metric(metric)
    names = [c.casefold() for c in definition.columns] + [definition.label.casefold()]
    if not term or search is None or any(name in term.casefold() for name in names):
        return ""
    try:
        hits = search(term).hits
    except Exception:  # noqa: BLE001 — база знаний недоступна: проверить нечем
        return ""
    described = next((h for h in hits if term.casefold() in h.text.casefold()), None)  # достаточно, что документ называет термин
    if described is None or all(c.casefold() in described.text.casefold() for c in definition.columns):
        return ""
    return (
        f"показатель «{term}» описан в документах ({described.source.citation}), но не рассчитывается по этим данным: "
        f"выбранный показатель «{definition.label}» — другой"
    )


def finalize_custom_plan(
    dataset_id: str, draft: PlanDraft, summary: DatasetSummary, context: BusinessContext,
    lookup: Callable[[str], ContextFragment | None], request: str = "", search: Callable[[str], object] | None = None,
) -> AnalysisPlan:
    available = summary.available_metrics
    times, categories = time_columns(summary), category_columns(summary)
    dimensions = [*times, *categories]
    steps: list[AnalysisStep] = []
    excluded: list[ExcludedItem] = []
    notes: list[str] = []
    seen: set[tuple] = set()

    for index, item in enumerate(draft.steps, 1):
        if (item.metric or "").strip().casefold() in PLACEHOLDER_METRICS:
            continue  # шаг-заглушка: модель ничего не выбрала (запрошенное перечислено в unsupported)
        title = (item.title or "").strip()[:MAX_TITLE] or f"Шаг {index}"
        metric = _match_metric(item.metric, available)
        if metric is None:
            excluded.append(ExcludedItem(analysis="custom", label=title, reason=f"показатель «{item.metric}» отсутствует среди показателей набора данных"))
            continue
        if len(steps) >= MAX_CHARTS:
            notes.append(f"В плане не больше {MAX_CHARTS} графиков: шаг «{title}» отброшен.")
            continue

        group_by, problem = [], ""
        for name in item.group_by:
            column = _match_column(name, dimensions)
            if column is None:
                problem = f"колонка группировки «{name}» не найдена среди измерений набора данных ({', '.join(dimensions)})"
                break
            if column not in group_by:
                group_by.append(column)
        filters: list[FilterCondition] = []
        for spec in item.filters:
            condition, reason = _filter(spec, summary)
            if condition is None:
                problem = problem or reason
            else:
                filters.append(condition)
        if problem:
            excluded.append(ExcludedItem(analysis="custom", label=title, reason=problem))
            continue

        conflict = term_conflict(item.requested_term, metric, search)
        if conflict:
            excluded.append(ExcludedItem(analysis="custom", label=title, reason=conflict))
            continue

        single_intent = len(draft.steps) == 1  # слова о динамике относятся к шагу, только если шаг один
        if single_intent and group_by and times and not any(c in times for c in group_by) and DYNAMICS_WORDS.search(request):
            group_by = [times[0], *group_by][:MAX_GROUP_COLUMNS]  # рост и динамика требуют оси времени
            notes.append(f"«{title}»: в запросе речь о динамике, поэтому добавлена группировка по «{times[0]}».")
        if single_intent and group_by and not any(c in times for c in group_by) and VAGUE_GROUPING.search(request):
            group_by = []  # «по группам»: какая колонка имеется в виду, решает пользователь
            notes.append(f"«{title}»: группировка названа неопределённо — выберите колонку.")
        group_by.sort(key=lambda column: column not in times)  # дата — ось X
        if len(group_by) > MAX_GROUP_COLUMNS or (len(group_by) == 2 and group_by[0] not in times):
            notes.append(f"«{title}»: график строится по одной колонке группировки («{group_by[0]}»); две категории на одном графике не поддерживаются.")
            group_by = group_by[:1]
        group_by = group_by[:MAX_GROUP_COLUMNS]

        vague = bool(VAGUE_GROUPING.search(request)) or draft.ambiguous
        if not group_by and len(times) == 1 and not vague:
            group_by = times[:]  # группировка не названа: показываем динамику
        signature = (metric, tuple(group_by), repr([(f.column, f.op, f.value) for f in filters]))
        if signature in seen:
            continue
        seen.add(signature)

        if not group_by:  # неоднозначный запрос: колонку выберет пользователь на карточке плана
            mapping = ColumnMapping(
                role="category", column=None, candidates=dimensions, status="ambiguous",
                basis="запрос допускает несколько группировок: " + ", ".join(dimensions),
            )
            steps.append(AnalysisStep(
                chart_id=f"custom-{len(steps) + 1}", analysis="custom", title=title, metric=metric, chart_type="bar", x_column=None,
                mapping=mapping, rationale=item.rationale.strip(), group_by=[], filters=filters,
            ))
            continue

        chart_type, note = fit_custom_chart(item.chart_type, group_by, metric, summary, filters)
        if note:
            notes.append(f"График «{title}»: {note}.")
        role = "time" if group_by[0] in times else "category"
        steps.append(AnalysisStep(
            chart_id=f"custom-{len(steps) + 1}", analysis="custom", title=title, metric=metric, chart_type=chart_type, x_column=group_by[0],
            mapping=ColumnMapping(role=role, column=group_by[0], candidates=[group_by[0]], status="confident", basis="запрос пользователя"),
            rationale=item.rationale.strip(), group_by=group_by, filters=filters,
        ))

    definitions = BusinessContext(provider=context.provider, error=context.error, events=context.events)
    for metric in dict.fromkeys(s.metric for s in steps):
        try:
            fragment = lookup(metric)
        except Exception as exc:  # noqa: BLE001 — недоступная база знаний = «определения нет»
            fragment = None
            definitions.error = getattr(exc, "user_message", None) or str(exc)
        if fragment is not None:
            definitions.definitions[metric] = fragment
    metrics = [_metric_definition(m, definitions) for m in dict.fromkeys(s.metric for s in steps)]
    unsupported = list(dict.fromkeys([*(u.strip() for u in draft.unsupported if u.strip()), *(f"{e.label}: {e.reason}" for e in excluded)]))
    if definitions.error:
        notes.append(definitions.error)
    if not steps and not unsupported:  # модель не предложила ничего и ничего не отвергла: молчаливого пустого плана быть не должно
        unsupported = ["не удалось составить план по запросу: назовите показатель и группировку точнее"]
    return AnalysisPlan(
        dataset_id=dataset_id, goal=draft.goal.strip(), planner="llm", charts=steps, metrics=metrics, excluded=excluded,
        clarifications=_clarifications(steps), documentation_gaps=[f"{m.label}: определение не найдено в документации" for m in metrics if not m.documented],
        notes=notes, unsupported=unsupported, available_analyses=[],
    )
