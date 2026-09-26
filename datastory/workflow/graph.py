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
from datastory.insights.generator import ChartInsightInput, generate_insights as data_quality_notes, llm_insight, rules_insight
from datastory.insights.verifier import Violation, check_insight
from datastory.llm.client import StructuredLLM
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.mcp_client.connection import McpToolError
from datastory.models import ChartSpec, Insight
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.skills import Skill, SkillError, load_skill
from datastory.workflow.catalog import ALL_IDS, evaluate_candidates
from datastory.workflow.context import BusinessContext, make_searcher, retrieve_business_context
from datastory.workflow.models import (
    AnalysisCandidate,
    AnalysisIntent,
    AnalysisPlan,
    AnalysisResult,
    ApprovalDecision,
    ApprovalRequest,
    DashboardSection,
    DashboardSpec,
    InsightCheck,
    IntentDraft,
    MethodologyInfo,
    PlanDraft,
)
from datastory.workflow.planning import apply_decision, finalize_plan, normalize_selection, rules_draft
from datastory.workflow.methodology import SKILL_NAME, STAGE_TITLES, methodology_info
from datastory.workflow.prompts import INTENT_SYSTEM, insight_system, intent_prompt, plan_prompt, plan_system

logger = logging.getLogger("datastory.workflow")

MAX_REVISIONS = 5  # сколько раз пользователь может отклонить план и запросить новый
MAX_INSIGHT_ATTEMPTS = 2  # первая попытка LLM и одна повторная с замечаниями проверки
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


@dataclass
class WorkflowDeps:
    """Зависимости узлов. Провайдеры вызываются при каждом обращении: закрытое подключение MCP заменяется новым."""

    client: Callable[[], AnalyticsMcpClient]
    llm: Callable[[], StructuredLLM | None]
    kb: Callable[[], KnowledgeBase | None]
    skill: Callable[[], Skill] = lambda: load_skill(SKILL_NAME)  # методика читается из SKILL.md при каждом обращении

    @classmethod
    def of(
        cls, client: AnalyticsMcpClient, llm: StructuredLLM | None = None, kb: KnowledgeBase | None = None,
        skill: Skill | None = None,
    ) -> "WorkflowDeps":
        return cls(client=lambda: client, llm=lambda: llm, kb=lambda: kb, skill=(lambda: skill) if skill else (lambda: load_skill(SKILL_NAME)))


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


def build_graph(deps: WorkflowDeps, checkpointer=None):
    def analyze_intent(state: WorkflowState) -> WorkflowState:
        summary = deps.client().profile_dataset(state["dataset_id"])  # MCP profile_dataset: доступные колонки и метрики
        candidates = evaluate_candidates(summary)
        request, llm, warnings = (state.get("user_request") or "").strip(), deps.llm(), state.get("warnings")
        intent = AnalysisIntent(analyses=list(ALL_IDS))
        if llm is None:
            warnings = _merge(warnings, LLM_OFF)
        elif request:
            try:
                draft = llm.generate(IntentDraft, system=INTENT_SYSTEM, user=intent_prompt(request, summary))
                intent = AnalysisIntent(
                    analyses=normalize_selection(list(draft.analyses), list(ALL_IDS)), source="llm",
                    unsupported=draft.unsupported, comment=draft.comment,
                )
            except LLMError as exc:
                warnings = _merge(warnings, f"Запрос не удалось разобрать моделью ({exc.user_message}): показаны все доступные анализы.")
        return {"summary": summary, "candidates": candidates, "intent": intent, "warnings": warnings or [], "status": "started"}

    def retrieve_context(state: WorkflowState) -> WorkflowState:
        available = [c.id for c in state["candidates"] if c.available]
        searcher = make_searcher(deps.kb(), state["dataset_id"])
        context = retrieve_business_context(available, searcher)
        warnings = state.get("warnings")
        if context.error:
            warnings = _merge(warnings, context.error)
        return {"context": context, "warnings": warnings or []}

    def build_analysis_plan(state: WorkflowState) -> WorkflowState:
        selected = normalize_selection(state.get("selected_analyses"), state["intent"].analyses)
        llm = deps.llm()
        system, warnings = _with_methodology(deps, "plan", plan_system, state) if llm is not None else (None, state.get("warnings"))
        draft, planner, methodology = None, "rules", state.get("methodology")
        if llm is not None and system is not None:
            try:
                draft = llm.generate(
                    PlanDraft, system=system,
                    user=plan_prompt(state["summary"], state["candidates"], selected, state["context"], state.get("user_request", "")),
                )
                planner, methodology = "llm", _record_stage(deps, state, "plan")
            except LLMError as exc:
                warnings = _merge(warnings, f"План составлен по правилам: {exc.user_message}")
        if draft is None:
            draft = rules_draft(state["candidates"], selected)
        update = {"plan_draft": draft, "planner": planner, "selected_analyses": selected, "warnings": warnings or []}
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
                result = client.calculate_metrics(dataset_id, chart.metric, [chart.x_column])
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
        return ChartInsightInput(
            chart=chart, result=result, facts=build_facts(result, chart.mapping.role),
            definition=plan.metric_definition(chart.metric), events=context.events, filename=state["summary"].filename,
            definition_fragment=context.definitions.get(chart.metric), feedback=feedback or [],
        )

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
            if llm is not None and system is not None:
                try:
                    insights[cid], found = llm_insight(llm, inp, system)
                    used_skill = True
                except LLMError as exc:
                    warnings = _merge(warnings, f"Вывод «{chart.title}» составлен по правилам: {exc.user_message}")
                    insights[cid] = rules_insight(inp)
            else:
                insights[cid] = rules_insight(inp)
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
            found = state.get("insight_issues", {}).get(cid, []) + [str(v) for v in check_insight(insight, inp.facts, inp.mask(), inp.labels())]
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
            dataset_id=state["dataset_id"], summary=summary, plan=plan, kpis=kpis,
            metrics=[results[c.chart_id] for c in charts], charts=[specs[c.chart_id] for c in charts],
            insights=[insights[c.chart_id] for c in charts if c.chart_id in insights],
            insight_checks=[state["insight_checks"][c.chart_id] for c in charts if c.chart_id in state["insight_checks"]],
            dashboard=dashboard, methodology=state.get("methodology"), warnings=warnings,
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
