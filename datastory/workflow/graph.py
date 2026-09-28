"""Основной LangGraph workflow DataStory AI.

    analyze_intent → retrieve_context → build_analysis_plan → validate_analysis_plan → human_approval
        → execute_analytics → generate_insights → verify_insights → prepare_dashboard

Условные переходы:
- нет ни одного допустимого графика                → конец (пустой план с объяснением причин);
- пользователь отклонил план и изменил выбор       → build_analysis_plan (новый план, новое подтверждение);
- неоднозначная колонка или некорректный ответ     → снова human_approval с пояснением;
- вывод не прошёл проверку достоверности           → generate_insights (одна повторная попытка), затем правила.

Все расчёты выполняет MCP-сервер: узлы вызывают инструменты через клиент и не импортируют расчётный код. Числа в выводах
берутся только из результатов calculate_metrics. Подтверждение пользователя — настоящий LangGraph interrupt: состояние
хранится в чекпоинтере под thread_id сессии анализа, поэтому перезапуск скрипта Streamlit не теряет и не повторяет шаги.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

import datastory.models
from datastory.analytics import models as analytics_models
from datastory.rag import models as rag_models
from datastory.workflow import context as workflow_context
from datastory.workflow import models as workflow_models

from datastory.analytics.engine import basic_kpis, metric_kpis
from datastory.analytics.models import DatasetSummary, MetricResult
from datastory.errors import LLMError
from datastory.insights.facts import build_facts
from datastory.insights.generator import ChartInsightInput, generate_insights as data_quality_notes, llm_insight, material_drop, rules_insight
from datastory.insights.verifier import Violation, check_insight
from datastory.llm.client import StructuredLLM
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.mcp_client.connection import McpToolError
from datastory.models import ChartSpec, Insight
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.skills import Skill, SkillError, load_skill
from datastory.workflow.catalog import evaluate_candidates, is_generic_mode
from datastory.workflow.custom_plan import finalize_custom_plan, metric_definition_lookup
from datastory.workflow.context import BusinessContext, make_searcher, retrieve_business_context, retrieve_query_hits
from datastory.workflow.models import (
    AnalysisCandidate,
    AnalysisIntent,
    AnalysisPlan,
    AnalysisResult,
    ApprovalDecision,
    ApprovalRequest,
    AUTO_GOAL,
    ContextSource,
    DashboardSection,
    DashboardSpec,
    InsightCheck,
    IntentDraft,
    MethodologyInfo,
    PlanDraft,
)
from datastory.workflow.planning import apply_decision, finalize_plan, normalize_selection, rules_draft
from datastory.observability import clip, span
from datastory.workflow.methodology import SKILL_NAME, STAGE_TITLES, methodology_info
from datastory.workflow.prompts import custom_plan_prompt, insight_system, intent_prompt, intent_system, plan_prompt, plan_system

logger = logging.getLogger("datastory.workflow")

MAX_REVISIONS = 5  # сколько раз пользователь может отклонить план и запросить новый
MAX_INSIGHT_ATTEMPTS = 2  # первая попытка LLM и одна повторная с замечаниями проверки
EMPTY_PLAN_REMINDER = (
    "\n\nПредыдущий ответ был пустым. Верни хотя бы один шаг в steps или перечисли запрошенное в unsupported."
)
WRONG_FIELD_REMINDER = (
    "\n\nПредыдущий ответ описал график в поле charts — это неправильное поле для этого сценария, оно не используется и было проигнорировано. "
    "Опиши тот же самый график в поле steps (title, metric — id показателя из списка, group_by, filters, chart_type), а charts оставь пустым []."
)
LLM_OFF = "Языковая модель недоступна (не задан OPENAI_API_KEY): план и выводы составлены по детерминированным правилам."


class WorkflowState(TypedDict, total=False):
    dataset_id: str
    user_request: str
    summary: DatasetSummary
    candidates: list[AnalysisCandidate]
    intent: AnalysisIntent
    selected_analyses: list[str]  # выбор пользователя после «изменить выбор»
    context: BusinessContext
    plan_draft: PlanDraft
    custom_mode: bool  # план составляет LLM по запросу (набор без колонок платёжной системы), а не каталог анализов
    planner: str
    plan: AnalysisPlan
    column_choices: dict[str, str]  # chart_id -> колонка, выбранная пользователем
    approval: ApprovalDecision | None
    approval_round: int
    notice: str
    revision: int
    status: str  # planned | empty | approved | cancelled | completed
    metric_results: dict[str, MetricResult]
    chart_specs: dict[str, ChartSpec]
    insights: dict[str, Insight]
    insight_issues: dict[str, list[str]]  # нарушения, найденные при сборке вывода
    insight_feedback: dict[str, list[str]]  # замечания для повторной генерации
    insight_history: dict[str, list[str]]
    insight_attempt: int
    insight_checks: dict[str, InsightCheck]
    warnings: list[str]
    methodology: MethodologyInfo  # какие этапы получили инструкции Skill
    dashboard: DashboardSpec
    result: AnalysisResult


@dataclass(frozen=True)
class PlannerRag:
    """Как планировщик по запросу использует RAG (для A/B эксперимента; по умолчанию — рабочая конфигурация)."""

    context_hits: int = 3  # сколько фрагментов передаётся LLM; 0 — контекст RAG не передаётся
    only_relevant: bool = True  # False — Top-k как есть, без порога релевантности
    validation: bool = True  # проверка плана после LLM (определения показателей, защита от подмены термина) обращается к базе знаний


@dataclass
class WorkflowDeps:
    """Зависимости узлов. Провайдеры вызываются при каждом обращении: закрытое подключение MCP заменяется новым."""

    client: Callable[[], AnalyticsMcpClient]
    llm: Callable[[], StructuredLLM | None]
    kb: Callable[[], KnowledgeBase | None]
    skill: Callable[[], Skill] = lambda: load_skill(SKILL_NAME)  # методика читается из SKILL.md при каждом обращении
    planner_rag: PlannerRag = PlannerRag()

    @classmethod
    def of(
        cls, client: AnalyticsMcpClient, llm: StructuredLLM | None = None, kb: KnowledgeBase | None = None,
        skill: Skill | None = None, planner_rag: PlannerRag | None = None,
    ) -> "WorkflowDeps":
        return cls(
            client=lambda: client, llm=lambda: llm, kb=lambda: kb, skill=(lambda: skill) if skill else (lambda: load_skill(SKILL_NAME)),
            planner_rag=planner_rag or PlannerRag(),
        )


def _merge(existing: list[str] | None, *new: str) -> list[str]:
    return list(dict.fromkeys([*(existing or []), *new]))


def _state_types() -> list[tuple[str, str]]:
    """Модели и перечисления, которые попадают в чекпоинт (сериализатор LangGraph требует явный список)."""
    found = []
    for module in (datastory.models, analytics_models, workflow_models, workflow_context, rag_models):
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == module.__name__ and issubclass(obj, (BaseModel, Enum)):
                found.append((module.__name__, name))
    return found


def make_checkpointer() -> MemorySaver:
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_state_types()))


def _with_methodology(deps: WorkflowDeps, stage: str, build_system, state: WorkflowState) -> tuple[str | None, list[str]]:
    """Системный промпт этапа с разделами Skill. Без методики LLM не вызывается: возвращается (None, предупреждения)."""
    try:
        return build_system(deps.skill()), state.get("warnings") or []
    except SkillError as exc:
        return None, _merge(state.get("warnings"), f"Методика не загружена ({exc.user_message}): этап «{STAGE_TITLES[stage]}» выполнен по правилам.")


def _record_stage(deps: WorkflowDeps, state: WorkflowState, stage: str) -> MethodologyInfo | None:
    try:
        skill = deps.skill()
    except SkillError:
        return state.get("methodology")
    stages = list(dict.fromkeys([*(state["methodology"].stages if state.get("methodology") else []), stage]))
    return methodology_info(skill, stages)


def _context_sources(context: BusinessContext, plan: AnalysisPlan, insights: list) -> list[ContextSource]:
    """Фрагменты базы знаний, найденные при анализе, и выводы, которые на них ссылаются."""
    found: list[tuple[str, str, str, str | None]] = []  # (кто, цитата, текст, показатель)
    for metric, fragment in context.definitions.items():
        definition = plan.metric_definition(metric)
        if definition is not None:  # определение показателя, которого нет в плане, пользователю не показывается
            found.append(("definition", fragment.citation, fragment.text, definition.label))
    found += [("event", f.citation, f.text, None) for f in context.events]
    sources: dict[str, ContextSource] = {}
    for kind, citation, text, label in found:
        used = [i.title for i in insights if i.context_source and citation in i.context_source]
        if citation in sources:  # один фрагмент может быть и определением, и контекстом
            sources[citation].used_in = list(dict.fromkeys([*sources[citation].used_in, *used]))
            continue
        sources[citation] = ContextSource(citation=citation, text=text, kind=kind, metric_label=label, used_in=used)
    return list(sources.values())


def build_graph(deps: WorkflowDeps, checkpointer=None):
    def analyze_intent(state: WorkflowState) -> WorkflowState:
        summary = deps.client().profile_dataset(state["dataset_id"])  # MCP profile_dataset: доступные колонки и метрики
        candidates = evaluate_candidates(summary)
        request, llm, warnings = (state.get("user_request") or "").strip(), deps.llm(), state.get("warnings")
        default_analyses = [c.id for c in candidates]
        intent = AnalysisIntent(analyses=default_analyses)
        custom = llm is not None and bool(request) and request != AUTO_GOAL and is_generic_mode(summary)
        if llm is None:
            warnings = _merge(warnings, LLM_OFF)
        elif custom:  # запрос разбирает планировщик по данным: отдельный разбор намерения по каталогу не нужен
            intent = AnalysisIntent(analyses=default_analyses, source="llm", comment="План строится по запросу пользователя.")
        elif request and request != AUTO_GOAL:  # автоматический анализ: сужать нечего, берутся все доступные анализы
            try:
                draft = llm.generate(IntentDraft, system=intent_system(candidates), user=intent_prompt(request, summary))
                intent = AnalysisIntent(
                    analyses=normalize_selection([a for a in draft.analyses if a in default_analyses], default_analyses), source="llm",
                    unsupported=draft.unsupported, comment=draft.comment,
                )
            except LLMError as exc:
                warnings = _merge(warnings, f"Запрос не удалось разобрать моделью ({exc.user_message}): показаны все доступные анализы.")
        return {
            "summary": summary, "candidates": candidates, "intent": intent, "warnings": warnings or [], "status": "started",
            "custom_mode": custom,
        }

    def retrieve_context(state: WorkflowState) -> WorkflowState:
        rag = deps.planner_rag
        searcher = make_searcher(deps.kb(), state["dataset_id"], rag.context_hits if rag.context_hits else None)
        if state.get("custom_mode"):  # определения ищутся по запросу, а затем по выбранным показателям (validate_analysis_plan)
            hits = retrieve_query_hits(state["user_request"], searcher, rag.only_relevant) if rag.context_hits else []
            context = BusinessContext(query_hits=hits)
        else:
            available = [c.id for c in state["candidates"] if c.available]
            context = retrieve_business_context(available, searcher, {c.id: c.metric for c in state["candidates"]})
        warnings = state.get("warnings")
        if context.error:
            warnings = _merge(warnings, context.error)
        return {"context": context, "warnings": warnings or []}

    def build_analysis_plan(state: WorkflowState) -> WorkflowState:
        selected = normalize_selection(state.get("selected_analyses"), state["intent"].analyses)
        llm = deps.llm()
        with span(
            "analysis_plan.generate", tags=("plan",),
            inputs={"goal": clip(state.get("user_request", "")), "selected_analyses": selected, "llm": llm is not None},
        ) as sp:
            custom = bool(state.get("custom_mode"))
            builder = (lambda skill: plan_system(skill, custom=True)) if custom else plan_system
            system, warnings = _with_methodology(deps, "plan", builder, state) if llm is not None else (None, state.get("warnings"))
            draft, planner, methodology = None, "rules", state.get("methodology")
            if llm is not None and system is not None:
                try:
                    user = (
                        custom_plan_prompt(state["summary"], state["context"], state.get("user_request", ""), deps.planner_rag.context_hits > 0) if custom
                        else plan_prompt(state["summary"], state["candidates"], selected, state["context"], state.get("user_request", ""))
                    )
                    draft = llm.generate(PlanDraft, system=system, user=user)
                    if custom and not draft.steps and not draft.unsupported:  # пустой ответ или график ушёл не в то поле: одна повторная попытка
                        reminder = WRONG_FIELD_REMINDER if draft.charts else EMPTY_PLAN_REMINDER
                        draft = llm.generate(PlanDraft, system=system, user=user + reminder)
                    planner, methodology = "llm", _record_stage(deps, state, "plan")
                except LLMError as exc:
                    warnings = _merge(warnings, f"План составлен по правилам: {exc.user_message}")
            if draft is None:
                draft, custom = rules_draft(state["candidates"], selected), False  # без модели работает каталог анализов
            sp.outputs({
                "planner": planner, "custom": custom, "goal": clip(draft.goal), "unsupported": draft.unsupported,
                "steps": [f"{c.analysis}:{c.chart_type}:{c.x_column}" for c in draft.charts]
                + [f"{s.metric}|{','.join(s.group_by)}|{len(s.filters)} filters|{s.chart_type}" for s in draft.steps],
            })
        update = {"plan_draft": draft, "planner": planner, "custom_mode": custom, "selected_analyses": selected, "warnings": warnings or []}
        return {**update, "methodology": methodology} if methodology else update

    def validate_analysis_plan(state: WorkflowState) -> WorkflowState:
        from datastory.rag.api import resolve_column

        kb, dataset_id = deps.kb(), state["dataset_id"]

        def resolver(phrase: str):
            try:
                return resolve_column(phrase, dataset_id=dataset_id, kb=kb)
            except Exception as exc:  # noqa: BLE001 — недоступный RAG означает «подтверждения нет», а не ошибку анализа
                logger.warning("Сопоставление колонки через RAG не удалось: %s", exc)
                return None

        if state.get("custom_mode"):
            searcher = make_searcher(kb, dataset_id)
            if deps.planner_rag.validation:
                lookup, guard = metric_definition_lookup(searcher), searcher
            else:  # RAG выключен и при проверке плана: определений и защиты от подмены термина нет
                lookup, guard = (lambda metric: None), None
            plan = finalize_custom_plan(
                dataset_id, state["plan_draft"], state["summary"], state["context"], lookup, state.get("user_request", ""), guard,
            )
        else:
            plan = finalize_plan(
                dataset_id, state["plan_draft"], state["planner"], state["summary"], state["candidates"],
                state["selected_analyses"], state["context"], state.get("column_choices"), resolver,
            )
        unsupported = [f"Запрошено, но не поддерживается в MVP: {item}." for item in state["intent"].unsupported]
        if unsupported:
            plan = plan.model_copy(update={"notes": [*plan.notes, *unsupported]})
        return {"plan": plan, "status": "planned" if plan.charts else "empty", "approval": None}

    def route_after_validation(state: WorkflowState) -> str:
        return "human_approval" if state["plan"].charts else END

    def human_approval(state: WorkflowState) -> WorkflowState:
        round_number = state.get("approval_round", 0) + 1
        request = ApprovalRequest(plan=state["plan"], notice=state.get("notice", ""), round=round_number)
        raw = interrupt(request.model_dump(mode="json"))  # граф останавливается; продолжит Command(resume=...)
        base: WorkflowState = {"approval_round": round_number, "notice": ""}
        try:
            decision = ApprovalDecision.model_validate(raw)
        except ValidationError:
            return {**base, "approval": None, "notice": "Не удалось разобрать ответ. Повторите выбор."}

        choices = {**state.get("column_choices", {}), **decision.column_choices}
        if decision.action == "cancel":
            return {**base, "approval": decision, "status": "cancelled"}
        if decision.action == "revise":
            revision = state.get("revision", 0) + 1
            if revision > MAX_REVISIONS:
                warnings = _merge(state.get("warnings"), f"Достигнут предел числа пересмотров плана ({MAX_REVISIONS}).")
                return {**base, "approval": decision, "status": "cancelled", "warnings": warnings}
            return {
                **base, "approval": decision, "revision": revision, "column_choices": choices,
                "selected_analyses": decision.selected_analyses or state["selected_analyses"],
            }

        outcome = apply_decision(state["plan"], decision, state["summary"])
        if outcome.error:
            return {**base, "approval": None, "plan": outcome.plan, "column_choices": choices, "notice": outcome.error}
        if outcome.unresolved:
            names = ", ".join(f"«{c.title}»" for c in outcome.plan.charts if c.chart_id in outcome.unresolved)
            return {
                **base, "approval": None, "plan": outcome.plan, "column_choices": choices,
                "notice": f"Уточните колонку для оси X: {names}.",
            }
        return {**base, "approval": decision, "plan": outcome.plan, "column_choices": choices, "status": "approved"}

    def route_after_approval(state: WorkflowState) -> str:
        decision = state.get("approval")
        if decision is None:
            return "human_approval"  # некорректный ответ или неоднозначность: спрашиваем снова
        if decision.action == "cancel" or state.get("status") == "cancelled":
            return END
        return "build_analysis_plan" if decision.action == "revise" else "execute_analytics"

    def execute_analytics(state: WorkflowState) -> WorkflowState:
        client, dataset_id = deps.client(), state["dataset_id"]
        results: dict[str, MetricResult] = {}
        specs: dict[str, ChartSpec] = {}
        warnings = state.get("warnings")
        for chart in state["plan"].charts:
            try:  # McpConnectionError не перехватывается: без сервера анализ невозможен, его повторяют целиком
                result = client.calculate_metrics(dataset_id, chart.metric, chart.grouping, chart.filters or None)
                specs[chart.chart_id] = client.create_chart_spec(result, chart.chart_type, chart.title, chart.x_column)
            except McpToolError as exc:
                warnings = _merge(warnings, f"График «{chart.title}» не построен: инструмент {exc.tool} отклонил запрос: {exc.message}")
                continue
            results[chart.chart_id] = result
            warnings = _merge(warnings, *(f"«{chart.title}»: {w}" for w in result.warnings))
        return {
            "metric_results": results, "chart_specs": specs, "warnings": warnings or [],
            "insights": {}, "insight_feedback": {}, "insight_history": {}, "insight_attempt": 0, "insight_checks": {},
        }

    def _input(state: WorkflowState, chart_id: str, feedback: list[str] | None = None) -> ChartInsightInput:
        plan, context = state["plan"], state["context"]
        chart = next(c for c in plan.charts if c.chart_id == chart_id)
        result = state["metric_results"][chart_id]
        inp = ChartInsightInput(
            chart=chart, result=result, facts=build_facts(result, chart.mapping.role),
            definition=plan.metric_definition(chart.metric), events=[], filename=state["summary"].filename,
            definition_fragment=context.definitions.get(chart.metric), feedback=feedback or [],
        )
        # событие из документации предлагается модели только там, где есть существенное падение, с которым оно могло совпасть по времени
        if material_drop(inp, next((f for f in inp.facts if f.id == "drop"), None)):
            inp.events = context.events
        return inp

    def generate_insights(state: WorkflowState) -> WorkflowState:
        llm = deps.llm()
        system, warnings = _with_methodology(deps, "insights", insight_system, state) if llm is not None else (None, state.get("warnings"))
        methodology, used_skill = state.get("methodology"), False
        insights = dict(state.get("insights", {}))
        issues = dict(state.get("insight_issues", {}))
        feedback = state.get("insight_feedback", {})
        for chart in state["plan"].charts:
            cid = chart.chart_id
            if cid not in state["metric_results"] or (cid in insights and cid not in feedback):
                continue
            inp = _input(state, cid, feedback.get(cid))
            found: list[Violation] = []
            with span(
                "insight.generate", tags=("insight",), metadata={"chart_id": cid, "metric": chart.metric, "attempt": state.get("insight_attempt", 0) + 1},
                inputs={"chart": chart.title, "facts": len(inp.facts), "retry_feedback": len(inp.feedback)},
            ) as sp:
                if llm is not None and system is not None:
                    try:
                        insights[cid], found = llm_insight(llm, inp, system)
                        used_skill = True
                    except LLMError as exc:
                        warnings = _merge(warnings, f"Вывод «{chart.title}» составлен по правилам: {exc.user_message}")
                        insights[cid] = rules_insight(inp)
                else:
                    insights[cid] = rules_insight(inp)
                sp.outputs({"generated_by": insights[cid].generated_by, "title": insights[cid].title, "evidence": len(insights[cid].evidence), "assembly_issues": len(found)})
            issues[cid] = [str(v) for v in found]
        if used_skill:
            methodology = _record_stage(deps, state, "insights")
        update = {
            "insights": insights, "insight_issues": issues, "insight_feedback": {},
            "insight_attempt": state.get("insight_attempt", 0) + 1, "warnings": warnings or [],
        }
        return {**update, "methodology": methodology} if methodology else update

    def verify_insights(state: WorkflowState) -> WorkflowState:
        attempt, warnings = state["insight_attempt"], state.get("warnings")
        insights = dict(state["insights"])
        history = {k: list(v) for k, v in state.get("insight_history", {}).items()}
        checks = dict(state.get("insight_checks", {}))
        feedback: dict[str, list[str]] = {}
        for cid, insight in state["insights"].items():
            inp = _input(state, cid)
            with span("insight.verify", tags=("insight", "verification"), metadata={"chart_id": cid, "attempt": attempt}, inputs={"generated_by": insight.generated_by}) as sp:
                found = state.get("insight_issues", {}).get(cid, []) + [str(v) for v in check_insight(insight, inp.facts, inp.mask(), inp.labels())]
                sp.outputs({"passed": not found, "violations": [clip(v, 200) for v in found]})
            if not found:
                checks[cid] = InsightCheck(chart_id=cid, passed=True, attempts=attempt, violations=history.get(cid, []))
                continue
            history[cid] = _merge(history.get(cid), *found)
            if insight.generated_by == "llm" and attempt < MAX_INSIGHT_ATTEMPTS:
                feedback[cid] = found  # вернёмся в generate_insights с замечаниями
                continue
            fallback = rules_insight(inp) if insight.generated_by == "llm" else insight
            remaining = [str(v) for v in check_insight(fallback, inp.facts, inp.mask(), inp.labels())]
            insights[cid] = fallback
            checks[cid] = InsightCheck(
                chart_id=cid, passed=not remaining, attempts=attempt, violations=history[cid], fallback=insight.generated_by == "llm"
            )
            if insight.generated_by == "llm":
                warnings = _merge(warnings, f"Вывод «{inp.chart.title}» не прошёл проверку достоверности и заменён расчётным текстом.")
            if remaining:
                warnings = _merge(warnings, f"Вывод «{inp.chart.title}» не прошёл проверку: {'; '.join(remaining)}")
        return {"insights": insights, "insight_history": history, "insight_checks": checks, "insight_feedback": feedback, "warnings": warnings or []}

    def route_after_verification(state: WorkflowState) -> str:
        return "generate_insights" if state.get("insight_feedback") else "prepare_dashboard"

    def prepare_dashboard(state: WorkflowState) -> WorkflowState:
        plan, summary = state["plan"], state["summary"]
        context: BusinessContext = state["context"]
        results, specs, insights = state["metric_results"], state["chart_specs"], state["insights"]
        charts = [c for c in plan.charts if c.chart_id in specs]
        kpis = metric_kpis({r.metric: r for r in results.values()}) or basic_kpis(summary)
        quality = [f"{i.title}: {i.text}" for i in data_quality_notes(summary) if i.severity == "warning"]
        warnings = _merge(state.get("warnings"), *plan.documentation_gaps, *quality)
        dashboard = DashboardSpec(
            title=f"Дашборд: {summary.filename}", dataset_id=state["dataset_id"], filename=summary.filename, kpis=kpis,
            sections=[
                DashboardSection(
                    chart=specs[c.chart_id], insight=insights.get(c.chart_id),
                    metric_label=results[c.chart_id].label, formula=results[c.chart_id].formula,
                )
                for c in charts
            ],
            documentation_gaps=plan.documentation_gaps, warnings=warnings,
        )
        result = AnalysisResult(
            dataset_id=state["dataset_id"], goal=state.get("user_request", ""), summary=summary, plan=plan, kpis=kpis,
            metrics=[results[c.chart_id] for c in charts], charts=[specs[c.chart_id] for c in charts],
            insights=[insights[c.chart_id] for c in charts if c.chart_id in insights],
            insight_checks=[state["insight_checks"][c.chart_id] for c in charts if c.chart_id in state["insight_checks"]],
            dashboard=dashboard, methodology=state.get("methodology"),
            context_sources=_context_sources(context, plan, [insights[c.chart_id] for c in charts if c.chart_id in insights]),
            warnings=warnings,
        )
        return {"dashboard": dashboard, "result": result, "status": "completed", "warnings": warnings}

    graph = StateGraph(WorkflowState)
    for node in (
        analyze_intent, retrieve_context, build_analysis_plan, validate_analysis_plan, human_approval,
        execute_analytics, generate_insights, verify_insights, prepare_dashboard,
    ):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "analyze_intent")
    graph.add_edge("analyze_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "build_analysis_plan")
    graph.add_edge("build_analysis_plan", "validate_analysis_plan")
    graph.add_conditional_edges("validate_analysis_plan", route_after_validation, {"human_approval": "human_approval", END: END})
    graph.add_conditional_edges(
        "human_approval", route_after_approval,
        {"human_approval": "human_approval", "build_analysis_plan": "build_analysis_plan", "execute_analytics": "execute_analytics", END: END},
    )
    graph.add_edge("execute_analytics", "generate_insights")
    graph.add_edge("generate_insights", "verify_insights")
    graph.add_conditional_edges(
        "verify_insights", route_after_verification, {"generate_insights": "generate_insights", "prepare_dashboard": "prepare_dashboard"}
    )
    graph.add_edge("prepare_dashboard", END)
    return graph.compile(checkpointer=checkpointer or make_checkpointer())
