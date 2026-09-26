"""Общие вспомогательные объекты тестов воркфлоу: запуск, решения пользователя, сценарные ответы LLM."""
import re

from datastory.insights.generator import InsightDraft
from datastory.profiler.profiler import build_profile
from datastory.storage.store import DatasetStore
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.models import ApprovalDecision, ChartDraft, IntentDraft, PlanDraft
from datastory.workflow.runner import AnalysisRunner
from tests.helpers import ScriptedLLM


def make_runner(client, llm=None, kb=None, skill=None) -> AnalysisRunner:
    return AnalysisRunner(WorkflowDeps.of(client, llm, kb, skill))


def approve(snapshot, *, confirm_all=True, **extra) -> ApprovalDecision:
    plan = snapshot.request.plan
    confirmed = [m.metric for m in plan.metrics if not m.documented] if confirm_all else []
    return ApprovalDecision(
        action="approve", approved_chart_ids=[c.chart_id for c in plan.charts], confirmed_metrics=confirmed, **extra
    )


def save_dataset(workspace, frame, name) -> str:
    profile, typed = build_profile(frame, name)
    DatasetStore(workspace).save(typed, profile)
    return profile.dataset_id


def fact_values(user: str) -> dict[str, str]:
    """id -> значение из таблицы числовых доказательств в промпте."""
    return {m.group(1): m.group(3).strip() for m in re.finditer(r"^(\w+) \| (.*) \| (.*)$", user, re.MULTILINE)}


def good_insight(system, user, n) -> InsightDraft:
    facts = fact_values(user)
    if "cat1_share" in facts:
        return InsightDraft(
            title="Распределение по каналам", severity="info", limitation=None, context_chunk_id=None, context_quote=None,
            summary=f"Наибольшая доля — {facts['cat1_share']} ({facts['cat1_value']}), итог — {facts['overall']}.",
            evidence_ids=["cat1_share", "cat1_value", "overall"],
        )
    return InsightDraft(
        title="Динамика показателя", severity="info", limitation=None, context_chunk_id=None, context_quote=None,
        summary=f"В первом периоде значение {facts['first']}, в последнем — {facts['last']}; общее изменение {facts['change']}.",
        evidence_ids=["first", "last", "change"],
    )


def plan_all(system, user, n) -> PlanDraft:
    return PlanDraft(
        goal="Динамика и структура операций",
        charts=[
            ChartDraft(analysis="count_dynamics", title="Количество операций по месяцам", chart_type="line", x_column="Month", rationale="Динамика."),
            ChartDraft(analysis="volume_dynamics", title="Объём по месяцам", chart_type="line", x_column="Month", rationale="Динамика."),
            ChartDraft(analysis="success_dynamics", title="Успешность по месяцам", chart_type="line", x_column="Month", rationale="Динамика."),
            ChartDraft(analysis="channel_distribution", title="Каналы", chart_type="pie", x_column="Channel", rationale="Структура."),
        ],
    )


def llm_for(insight=good_insight, plan=plan_all, intent=None) -> ScriptedLLM:
    responders = {"PlanDraft": plan, "InsightDraft": insight}
    responders["IntentDraft"] = intent or (lambda s, u, n: IntentDraft(analyses=["success_dynamics"], unsupported=["прогноз"], comment="Успешность."))
    return ScriptedLLM(**responders)
