"""Детерминированная часть планирования: черновик без LLM, проверка и нормализация плана, применение решения пользователя.

LLM только предлагает план (PlanDraft). Здесь план приводится к правилам: недоступные показатели исключаются,
типы графиков подгоняются под правила визуализации, неоднозначные колонки превращаются в вопросы пользователю.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datastory.analytics.metrics import METRICS
from datastory.analytics.models import DatasetSummary
from datastory.workflow.catalog import ALL_IDS, ANALYSES, BY_ID, AnalysisKind, fit_chart_type, metric_columns, unique_count
from datastory.workflow.context import BusinessContext
from datastory.workflow.models import (
    MAX_CHARTS,
    AnalysisCandidate,
    AnalysisPlan,
    ApprovalDecision,
    ChartDraft,
    Clarification,
    ColumnMapping,
    ExcludedItem,
    MetricDefinition,
    PlanDraft,
    PlannedChart,
)

MAX_TITLE = 120
Resolver = Callable[[str], object]  # фраза -> ColumnResolution (RAG) или None


def rules_draft(candidates: list[AnalysisCandidate], selected: list[str]) -> PlanDraft:
    """План без LLM: выбранные анализы с типами графиков по умолчанию."""
    by_id = {c.id: c for c in candidates}
    charts = [
        ChartDraft(
            analysis=kind.id, title=kind.label, chart_type=kind.default_chart,
            # без модели предложить колонку некому: при нескольких кандидатах выбор остаётся за пользователем
            x_column=by_id[kind.id].dimension_candidates[0] if len(by_id[kind.id].dimension_candidates) == 1 else "",
            rationale="Стандартный график для этого показателя.",
        )
        for kind in ANALYSES
        if kind.id in selected and by_id[kind.id].available
    ]
    return PlanDraft(goal="Динамика и структура операций", charts=charts)


def resolve_mapping(
    kind: AnalysisKind, candidates: list[str], proposed: str | None, choice: str | None, resolver: Resolver | None
) -> ColumnMapping:
    """Колонка для оси X. Одна подходящая — берём; несколько — только при согласии модели и RAG или по выбору пользователя."""
    if choice and choice in candidates:
        return ColumnMapping(role=kind.role, column=choice, candidates=candidates, status="confirmed", basis="выбор пользователя")
    if len(candidates) == 1:
        return ColumnMapping(
            role=kind.role, column=candidates[0], candidates=candidates, status="confident", basis="единственная подходящая колонка"
        )
    if proposed in candidates and resolver is not None:
        resolution = resolver(kind.role_phrase)
        chosen = getattr(getattr(resolution, "chosen", None), "column_name", None)
        if getattr(resolution, "status", None) == "confident" and chosen == proposed:
            return ColumnMapping(
                role=kind.role, column=proposed, candidates=candidates, status="confident",
                basis=f"модель и поиск по описаниям колонок (RAG) указали «{proposed}»",
            )
    hint = f" Модель предложила «{proposed}», но подтверждения нет." if proposed in candidates else ""
    return ColumnMapping(
        role=kind.role, column=None, candidates=candidates, status="ambiguous",
        basis=f"подходящих колонок несколько: {', '.join(candidates)}.{hint}",
    )


def _metric_definition(metric: str, context: BusinessContext) -> MetricDefinition:
    definition = METRICS[metric]
    fragment = context.definitions.get(metric)
    return MetricDefinition(
        metric=metric, label=definition.label, unit=definition.unit, formula=definition.formula,
        source_columns=metric_columns(metric), documented=fragment is not None,
        definition_source=fragment.citation if fragment else None, definition_quote=fragment.text if fragment else None,
    )


def _clarifications(charts: list[PlannedChart]) -> list[Clarification]:
    return [
        Clarification(
            chart_id=c.chart_id, candidates=c.mapping.candidates,
            question=f"Какая колонка используется для оси X графика «{c.title}»?",
        )
        for c in charts if c.mapping.status == "ambiguous"
    ]


def finalize_plan(
    dataset_id: str,
    draft: PlanDraft,
    planner: str,
    summary: DatasetSummary,
    candidates: list[AnalysisCandidate],
    selected: list[str],
    context: BusinessContext,
    column_choices: dict[str, str] | None = None,
    resolver: Resolver | None = None,
) -> AnalysisPlan:
    """Приводит черновик (LLM или правил) к допустимому плану."""
    choices = column_choices or {}
    by_id = {c.id: c for c in candidates}
    proposals: dict[str, ChartDraft] = {}
    notes: list[str] = []
    for item in draft.charts:
        if item.analysis in proposals:
            notes.append(f"Повторное предложение анализа «{BY_ID[item.analysis].label}» отброшено.")
        else:
            proposals[item.analysis] = item

    charts: list[PlannedChart] = []
    excluded: list[ExcludedItem] = []
    for kind in ANALYSES:  # порядок каталога, а не порядок ответа модели
        if kind.id not in selected:
            continue
        candidate = by_id[kind.id]
        if not candidate.available:
            excluded.append(ExcludedItem(analysis=kind.id, label=kind.label, reason="; ".join(candidate.reasons)))
            continue
        proposal = proposals.get(kind.id)
        mapping = resolve_mapping(
            kind, candidate.dimension_candidates, proposal.x_column if proposal else None, choices.get(kind.id), resolver
        )
        categories = None
        if kind.role == "category":  # для неоднозначной колонки берём худший случай
            counts = [unique_count(summary, c) for c in ([mapping.column] if mapping.column else mapping.candidates)]
            categories = max(counts, default=None)
        chart_type, note = fit_chart_type(kind, proposal.chart_type if proposal else None, categories)
        title = ((proposal.title if proposal else "") or kind.label).strip()[:MAX_TITLE]
        if note:
            notes.append(f"График «{title}»: {note}.")
        charts.append(
            PlannedChart(
                chart_id=kind.id, analysis=kind.id, title=title, metric=kind.metric, chart_type=chart_type,
                x_column=mapping.column, mapping=mapping, rationale=(proposal.rationale if proposal else "").strip(),
            )
        )
    if len(charts) > MAX_CHARTS:
        notes.append(f"В плане не больше {MAX_CHARTS} графиков: лишние отброшены.")
        charts = charts[:MAX_CHARTS]

    metrics = [_metric_definition(m, context) for m in dict.fromkeys(c.metric for c in charts)]
    gaps = [f"{m.label}: определение не найдено в документации" for m in metrics if not m.documented]
    if context.error:
        notes.append(context.error)
    return AnalysisPlan(
        dataset_id=dataset_id, goal=draft.goal.strip(), planner=planner if planner in ("llm", "rules") else "rules",
        charts=charts, metrics=metrics, excluded=excluded, clarifications=_clarifications(charts),
        documentation_gaps=gaps, notes=notes, available_analyses=[c.id for c in candidates if c.available],
    )


# --------------------------------------------------------------------------- решение пользователя
@dataclass
class DecisionOutcome:
    plan: AnalysisPlan  # при error/unresolved — план с применёнными выборами, но без отбора графиков
    error: str | None = None  # решение некорректно: спросить снова
    unresolved: list[str] | None = None  # графики, у которых колонка так и не выбрана


def apply_choices(plan: AnalysisPlan, choices: dict[str, str], summary: DatasetSummary) -> tuple[AnalysisPlan, list[str]]:
    """Подставляет выбор пользователя в неоднозначные колонки. Возвращает (план, ошибки)."""
    errors: list[str] = []
    charts = []
    for chart in plan.charts:
        choice = choices.get(chart.chart_id)
        if choice is None or chart.mapping.status != "ambiguous":
            charts.append(chart)
        elif choice not in chart.mapping.candidates:
            errors.append(f"Колонка «{choice}» не подходит для графика «{chart.title}»: доступно {', '.join(chart.mapping.candidates)}.")
            charts.append(chart)
        else:
            kind = BY_ID[chart.analysis]
            mapping = chart.mapping.model_copy(update={"column": choice, "status": "confirmed", "basis": "выбор пользователя"})
            chart_type, _ = fit_chart_type(
                kind, chart.chart_type, unique_count(summary, choice) if kind.role == "category" else None
            )
            charts.append(chart.model_copy(update={"x_column": choice, "mapping": mapping, "chart_type": chart_type}))
    return plan.model_copy(update={"charts": charts, "clarifications": _clarifications(charts)}), errors


def apply_decision(plan: AnalysisPlan, decision: ApprovalDecision, summary: DatasetSummary) -> DecisionOutcome:
    """Действие «approve»: выбор колонок, отбор графиков, подтверждение правил расчёта без документации."""
    plan, errors = apply_choices(plan, decision.column_choices, summary)
    if errors:
        return DecisionOutcome(plan, error=" ".join(errors))

    approved = [c for c in plan.charts if c.chart_id in set(decision.approved_chart_ids)]
    if not approved:
        return DecisionOutcome(plan, error="Не выбрано ни одного графика. Отметьте хотя бы один или отмените анализ.")
    pending = [c.chart_id for c in approved if c.mapping.status == "ambiguous"]
    if pending:
        return DecisionOutcome(plan, unresolved=pending)

    confirmed = set(decision.confirmed_metrics)
    kept, excluded = [], list(plan.excluded)
    for chart in approved:
        definition = plan.metric_definition(chart.metric)
        if definition and not definition.documented and chart.metric not in confirmed:
            excluded.append(
                ExcludedItem(
                    analysis=chart.analysis, label=chart.title,
                    reason=f"нет документации по показателю «{definition.label}», а правило расчёта ({definition.formula}) не подтверждено пользователем",
                )
            )
        else:
            kept.append(chart)
    if not kept:
        return DecisionOutcome(
            plan,
            error="Ни один график нельзя построить: у показателей нет документации, и правило расчёта не подтверждено. Подтвердите формулу или выберите другие показатели.",
        )

    used = {c.metric for c in kept}
    metrics = [
        m.model_copy(update={"formula_confirmed_by_user": m.metric in confirmed}) for m in plan.metrics if m.metric in used
    ]
    approved_plan = plan.model_copy(
        update={
            "charts": kept, "metrics": metrics, "excluded": excluded, "clarifications": [],
            "documentation_gaps": [f"{m.label}: определение не найдено в документации" for m in metrics if not m.documented],
        }
    )
    return DecisionOutcome(approved_plan)


def normalize_selection(requested: list[str] | None, default: list[str]) -> list[str]:
    """Оставляет только известные анализы; пустой выбор означает «выбор по умолчанию»."""
    chosen = [a for a in (requested or []) if a in ALL_IDS]
    return chosen or default
