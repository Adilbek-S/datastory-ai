"""Основной LangGraph workflow на настоящем MCP-сервере: план, подтверждение (interrupt), расчёты, выводы, дашборд."""
import ast
import re
from pathlib import Path

import pandas as pd
import pytest
from langgraph.types import Command

from datastory.analytics.formatting import format_metric_value
from datastory.errors import LLMError, WorkflowError
from datastory.insights.generator import CAUSE_UNKNOWN, NO_DEFINITION, InsightDraft
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.mcp_client.connection import McpConnection, McpServerConfig, McpToolError
from datastory.models import EvidenceType
from datastory.profiler.profiler import build_profile
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.storage.store import DatasetStore
from datastory.workflow.graph import LLM_OFF, WorkflowDeps, build_graph
from datastory.workflow.models import ApprovalDecision, ChartDraft, IntentDraft, PlanDraft
from datastory.workflow.runner import AnalysisRunner
from scripts import generate_demo_data as gen
from tests.helpers import ScriptedLLM
from tests.workflow_helpers import approve, fact_values, good_insight, llm_for, make_runner, save_dataset

ROWS = gen.build_rows()
GRAPH_SOURCE = Path(__file__).resolve().parent.parent / "datastory" / "workflow" / "graph.py"
DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
NODES = {
    "analyze_intent", "retrieve_context", "build_analysis_plan", "validate_analysis_plan", "human_approval",
    "execute_analytics", "generate_insights", "verify_insights", "prepare_dashboard",
}


def py_success_rate(rows) -> float:
    return sum(r["Successful"] for r in rows) / sum(r["Transactions"] for r in rows) * 100


@pytest.fixture
def kb_with_docs():
    kb = KnowledgeBase()
    kb.index_document(DEMO_PDF, "business_metrics.pdf")
    return kb


# ================================================================== структура графа
def test_graph_has_all_nine_nodes(client):
    assert NODES <= set(build_graph(WorkflowDeps.of(client)).get_graph().nodes)


def test_workflow_never_imports_calculation_code():
    """Граф не подменяет MCP локальными расчётами: функции расчёта и работа с таблицами в нём не импортируются."""
    tree = ast.parse(GRAPH_SOURCE.read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not names & {"calculate_metric", "build_chart_spec", "build_dataset_summary", "DatasetStore"}, names
    assert not modules & {"pandas", "datastory.analytics.metrics", "datastory.analytics.charts", "datastory.storage.store"}, modules


# ================================================================== human-in-the-loop
def test_analysis_stops_for_approval_before_any_calculation(client, datasets):
    runner = make_runner(client)
    client.calls.clear()
    snapshot = runner.start(datasets["demo"])
    assert snapshot.phase == "awaiting_approval" and snapshot.request.round == 1
    assert client.calls == ["profile_dataset"]  # ни calculate_metrics, ни create_chart_spec до подтверждения
    assert snapshot.result is None


def test_plan_has_the_four_supported_charts(client, datasets):
    plan = make_runner(client).start(datasets["demo"]).request.plan
    assert [(c.analysis, c.chart_type, c.x_column) for c in plan.charts] == [
        ("count_dynamics", "line", "Month"), ("volume_dynamics", "line", "Month"),
        ("success_dynamics", "line", "Month"), ("channel_distribution", "pie", "Channel"),
    ]
    assert len(plan.charts) <= 4 and plan.planner == "rules" and plan.excluded == []
    assert {m.metric for m in plan.metrics} == {"transaction_count", "transaction_volume", "success_rate"}


def test_missing_llm_is_reported(client, datasets):
    snapshot = make_runner(client).start(datasets["demo"])
    assert LLM_OFF in snapshot.warnings


def test_full_flow_on_demo_dataset_with_business_document(client, datasets, kb_with_docs):
    runner = make_runner(client, kb=kb_with_docs)
    snapshot = runner.start(datasets["demo"])
    plan = snapshot.request.plan
    assert plan.documentation_gaps == [] and all(m.documented for m in plan.metrics)
    assert "business_metrics.pdf" in plan.metric_definition("success_rate").definition_source

    client.calls.clear()
    done = runner.resume(snapshot.thread_id, approve(snapshot, confirm_all=False))  # документация есть: формулы подтверждать не нужно
    assert done.phase == "completed"
    assert client.calls.count("calculate_metrics") == 4 and client.calls.count("create_chart_spec") == 4
    result = done.result
    assert [c.kind for c in result.charts] == ["line", "line", "line", "pie"] and all(c.data for c in result.charts)
    assert result.dashboard.title.endswith("transactions_2026.xlsx") and len(result.dashboard.sections) == 4
    assert [k.label for k in result.kpis] == ["Количество транзакций", "Success Rate", "Объём транзакций"]

    # у каждого графика есть вывод: название, краткий вывод, числа, источник данных, источник контекста, ограничение
    assert [i.chart_id for i in result.insights] == [c.chart_id for c in plan.charts]
    for insight in result.insights:
        assert insight.title and insight.text and insight.evidence and "calculate_metrics" in insight.data_source
        assert "business_metrics.pdf" in insight.context_source and insight.evidence_type is EvidenceType.COMPUTED
    assert all(c.passed for c in result.insight_checks)


def test_insight_numbers_come_from_mcp_results(client, datasets, kb_with_docs):
    runner = make_runner(client, kb=kb_with_docs)
    snapshot = runner.start(datasets["demo"])
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    success = next(i for i in result.insights if i.chart_id == "success_dynamics")
    march = py_success_rate([r for r in ROWS if r["Month"] == "2026-03"])
    february = py_success_rate([r for r in ROWS if r["Month"] == "2026-02"])
    facts = {e.id: e for e in success.evidence}
    assert facts["min"].formatted == format_metric_value(march, "%") and facts["min"].group == "2026-03"
    assert facts["drop"].value == pytest.approx(march - february) and facts["drop"].group == "2026-02 → 2026-03"
    assert facts["drop"].computation.startswith("значение(2026-03) − значение(2026-02)")
    assert format_metric_value(march, "%") in success.text
    assert CAUSE_UNKNOWN in success.limitation  # причина снижения не установлена — так и сказано
    for causal in ("вызван", "из-за", "привел", "привёл"):
        assert causal not in success.text.lower()

    distribution = next(i for i in result.insights if i.chart_id == "channel_distribution")
    mobile = sum(r["Transactions"] for r in ROWS if r["Channel"] == "Mobile")
    assert format_metric_value(mobile, "шт.") in distribution.text


def test_missing_documentation_requires_the_users_formula(client, datasets):
    runner = make_runner(client)  # база знаний пуста: определений нет
    snapshot = runner.start(datasets["demo"])
    plan = snapshot.request.plan
    assert len(plan.documentation_gaps) == 3 and not any(m.documented for m in plan.metrics)

    # без подтверждения правила расчёта система не строит график, а спрашивает снова
    client.calls.clear()
    again = runner.resume(snapshot.thread_id, approve(snapshot, confirm_all=False))
    assert again.phase == "awaiting_approval" and again.request.round == 2
    assert "не подтверждено" in again.request.notice or "Подтвердите формулу" in again.request.notice
    assert "calculate_metrics" not in client.calls

    # подтверждён только Success Rate: остальные графики исключаются с объяснением
    partial = approve(snapshot, confirm_all=False).model_copy(update={"confirmed_metrics": ["success_rate"]})
    done = runner.resume(snapshot.thread_id, partial)
    assert done.phase == "completed"
    assert [c.chart_id for c in done.result.plan.charts] == ["success_dynamics"]
    assert {e.analysis for e in done.result.plan.excluded} == {"count_dynamics", "volume_dynamics", "channel_distribution"}
    insight = done.result.insights[0]
    assert NO_DEFINITION.format(formula="SUM(Successful) / SUM(Transactions) × 100") in insight.limitation
    assert insight.context_source is None  # источника бизнес-контекста нет, и он не выдуман
    assert done.result.plan.metric_definition("success_rate").formula_confirmed_by_user
    assert any("определение не найдено" in w for w in done.result.warnings)


def test_missing_column_excludes_the_metric(client, workspace, datasets):
    frame = pd.DataFrame(
        {"Month": [f"2026-0{m}" for m in range(1, 5)] * 2, "Channel": ["a"] * 4 + ["b"] * 4,
         "Transactions": [100, 110, 120, 130] * 2, "Successful": [98, 99, 100, 101] * 2}
    )
    dataset_id = save_dataset(workspace, frame, "no_amount.csv")
    snapshot = make_runner(client).start(dataset_id)
    plan = snapshot.request.plan
    assert [c.analysis for c in plan.charts] == ["count_dynamics", "success_dynamics", "channel_distribution"]
    excluded = {e.analysis: e.reason for e in plan.excluded}
    assert list(excluded) == ["volume_dynamics"] and "нет колонки «Amount_KZT»" in excluded["volume_dynamics"]
    assert "volume_dynamics" not in plan.available_analyses


def test_dataset_without_any_supported_chart_ends_without_plan(client, datasets):
    runner = make_runner(client)
    client.calls.clear()
    snapshot = runner.start(datasets["generic"])
    assert snapshot.phase == "empty" and snapshot.plan.charts == []
    # нет колонок платёжной системы: предлагаются универсальные анализы суммы «score», но нет ни даты, ни подходящей категории
    assert {e.analysis for e in snapshot.plan.excluded} == {"measure_dynamics", "measure_comparison"}
    assert client.calls == ["profile_dataset"]


def test_ambiguous_column_mapping_asks_the_user(client, workspace):
    frame = pd.DataFrame(
        {"Month": ["2026-01", "2026-02", "2026-03", "2026-04"] * 2,
         "Created": pd.to_datetime(["2026-01-15", "2026-02-20", "2026-03-05", "2026-04-11"] * 2),
         "Channel": ["a"] * 4 + ["b"] * 4, "Transactions": [100, 110, 120, 130] * 2, "Successful": [98, 99, 100, 101] * 2}
    )
    dataset_id = save_dataset(workspace, frame, "ambiguous.csv")
    runner = make_runner(client)
    snapshot = runner.start(dataset_id)
    plan = snapshot.request.plan
    dynamics = [c for c in plan.charts if c.mapping.role == "time"]
    assert dynamics and all(c.mapping.status == "ambiguous" and c.x_column is None for c in dynamics)
    assert {q.chart_id for q in plan.clarifications} == {c.chart_id for c in dynamics}
    assert all(q.candidates == ["Month", "Created"] for q in plan.clarifications)

    # ответ без выбора колонки: расчёты не начинаются, вопрос задаётся снова
    client.calls.clear()
    asked = runner.resume(snapshot.thread_id, approve(snapshot))
    assert asked.phase == "awaiting_approval" and asked.request.round == 2 and "Уточните колонку" in asked.request.notice
    assert "calculate_metrics" not in client.calls

    choices = {c.chart_id: "Month" for c in dynamics}
    done = runner.resume(snapshot.thread_id, approve(asked, column_choices=choices))
    assert done.phase == "completed"
    assert {c.x_column for c in done.result.plan.charts if c.mapping.role == "time"} == {"Month"}
    assert all(c.mapping.status == "confirmed" for c in done.result.plan.charts if c.mapping.role == "time")


def test_choice_outside_candidates_is_rejected(client, workspace):
    frame = pd.DataFrame(
        {"Month": ["2026-01", "2026-02", "2026-03"], "Created": pd.to_datetime(["2026-01-15", "2026-02-20", "2026-03-05"]),
         "Transactions": [100, 110, 120], "Successful": [98, 99, 100]}
    )
    runner = make_runner(client)
    snapshot = runner.start(save_dataset(workspace, frame, "bad_choice.csv"))
    decision = approve(snapshot, column_choices={"count_dynamics": "Transactions"})
    again = runner.resume(snapshot.thread_id, decision)
    assert again.phase == "awaiting_approval" and "не подходит" in again.request.notice


def test_rejected_plan_can_be_changed_and_rerun(client, datasets):
    runner = make_runner(client)
    snapshot = runner.start(datasets["demo"])
    client.calls.clear()
    revised = runner.resume(
        snapshot.thread_id, ApprovalDecision(action="revise", selected_analyses=["success_dynamics", "channel_distribution"])
    )
    assert revised.phase == "awaiting_approval" and revised.request.round == 2
    assert [c.analysis for c in revised.request.plan.charts] == ["success_dynamics", "channel_distribution"]
    assert "calculate_metrics" not in client.calls  # отклонённый план ничего не считал

    done = runner.resume(revised.thread_id, approve(revised))
    assert done.phase == "completed" and [c.chart_id for c in done.result.plan.charts] == ["success_dynamics", "channel_distribution"]
    assert client.calls.count("create_chart_spec") == 2


def test_user_can_deselect_a_chart_when_approving(client, datasets):
    runner = make_runner(client)
    snapshot = runner.start(datasets["demo"])
    decision = approve(snapshot).model_copy(update={"approved_chart_ids": ["volume_dynamics"]})
    done = runner.resume(snapshot.thread_id, decision)
    assert [c.chart_id for c in done.result.plan.charts] == ["volume_dynamics"] and len(done.result.charts) == 1
    assert done.result.kpis[0].label == "Объём транзакций"


def test_empty_selection_is_asked_again(client, datasets):
    runner = make_runner(client)
    snapshot = runner.start(datasets["demo"])
    again = runner.resume(snapshot.thread_id, ApprovalDecision(action="approve", approved_chart_ids=[]))
    assert again.phase == "awaiting_approval" and "Не выбрано ни одного графика" in again.request.notice


def test_cancel_finishes_without_calculations(client, datasets):
    runner = make_runner(client)
    snapshot = runner.start(datasets["demo"])
    client.calls.clear()
    assert runner.resume(snapshot.thread_id, ApprovalDecision(action="cancel")).phase == "cancelled"
    assert client.calls == []
    with pytest.raises(WorkflowError, match="не ожидает подтверждения"):
        runner.resume(snapshot.thread_id, ApprovalDecision(action="cancel"))


def test_unparseable_answer_is_asked_again(client, datasets):
    runner = make_runner(client)
    snapshot = runner.start(datasets["demo"])
    config = runner._config(snapshot.thread_id)
    runner.graph.invoke(Command(resume={"action": "разрешить всё"}), config)
    again = runner.snapshot(snapshot.thread_id)
    assert again.phase == "awaiting_approval" and "Не удалось разобрать ответ" in again.request.notice


# ================================================================== сессии, чекпоинт и перезапуск Streamlit
def test_every_analysis_session_has_its_own_thread(client, datasets):
    runner = make_runner(client)
    first, second = runner.start(datasets["demo"]), runner.start(datasets["demo"])
    assert first.thread_id != second.thread_id
    runner.resume(first.thread_id, approve(first))
    assert runner.snapshot(first.thread_id).phase == "completed"
    assert runner.snapshot(second.thread_id).phase == "awaiting_approval"  # другая сессия не затронута


def test_reruns_read_the_checkpoint_and_do_not_repeat_steps(client, datasets):
    runner = make_runner(client)
    snapshot = runner.start(datasets["demo"])
    client.calls.clear()
    for _ in range(5):  # перерисовки Streamlit
        again = runner.snapshot(snapshot.thread_id)
        assert again.phase == "awaiting_approval" and again.request == snapshot.request
    assert client.calls == []  # чтение состояния ничего не выполняет

    done = runner.resume(snapshot.thread_id, approve(snapshot))
    calls = list(client.calls)
    for _ in range(3):
        assert runner.snapshot(snapshot.thread_id).phase == "completed"
    assert client.calls == calls and calls.count("profile_dataset") == 0  # анализ не начинался заново
    assert done.result == runner.snapshot(snapshot.thread_id).result


def test_a_rebuilt_graph_continues_from_the_saved_checkpoint(client, datasets):
    runner = make_runner(client)
    snapshot = runner.start(datasets["demo"])
    rebuilt = AnalysisRunner(WorkflowDeps.of(client), checkpointer=runner.checkpointer)  # новый граф, то же хранилище чекпоинтов
    assert rebuilt.snapshot(snapshot.thread_id).request == snapshot.request
    assert rebuilt.resume(snapshot.thread_id, approve(snapshot)).phase == "completed"


def test_unknown_session_is_reported(client):
    with pytest.raises(WorkflowError, match="не найдена"):
        make_runner(client).snapshot("analysis-unknown")


def test_old_sessions_are_forgotten(client, datasets):
    runner = AnalysisRunner(WorkflowDeps.of(client), max_threads=2)
    ids = [runner.start(datasets["demo"]).thread_id for _ in range(3)]
    with pytest.raises(WorkflowError):
        runner.snapshot(ids[0])
    assert runner.snapshot(ids[2]).phase == "awaiting_approval"


def test_checkpoint_serialization_has_no_warnings(client, datasets, caplog):
    runner = make_runner(client)
    snapshot = runner.start(datasets["demo"])
    runner.resume(snapshot.thread_id, approve(snapshot))
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]


# ================================================================== ошибки MCP
def test_unavailable_mcp_server_is_a_reported_failure(workspace, datasets):
    broken = AnalyticsMcpClient(
        McpConnection(McpServerConfig(command="python", args=("-m", "no_such_module_xyz"), name="datastory-analytics"), startup_timeout=10)
    )
    try:
        runner = make_runner(broken)
        snapshot = runner.start(datasets["demo"])
        assert snapshot.phase == "failed" and snapshot.failure.kind == "connection"
        assert "Не удалось подключиться к MCP-серверу «datastory-analytics»" in snapshot.failure.message
        assert broken.calls == ["profile_dataset"]  # молчаливой подмены локальными расчётами нет
        with pytest.raises(WorkflowError):
            runner.resume(snapshot.thread_id, ApprovalDecision(action="cancel"))
    finally:
        broken.close()


class RefusingClient:
    """Настоящий клиент, у которого один показатель MCP-сервер «отклоняет»."""

    def __init__(self, real, refused_metric):
        self.real, self.refused = real, refused_metric

    def __getattr__(self, name):
        return getattr(self.real, name)

    def calculate_metrics(self, dataset_id, metric, *args, **kwargs):
        if metric == self.refused:
            raise McpToolError("calculate_metrics", "тестовый отказ")
        return self.real.calculate_metrics(dataset_id, metric, *args, **kwargs)


def test_a_refused_chart_is_skipped_and_reported(client, datasets):
    runner = make_runner(RefusingClient(client, "transaction_volume"))
    snapshot = runner.start(datasets["demo"])
    done = runner.resume(snapshot.thread_id, approve(snapshot))
    assert done.phase == "completed"
    assert [s.chart.title for s in done.result.dashboard.sections] == [
        "Динамика количества операций", "Динамика успешности", "Распределение по каналам"
    ]
    assert any("Динамика объёма" in w and "тестовый отказ" in w for w in done.result.warnings)


# ================================================================== LLM: структурные ответы
def test_llm_builds_a_structured_plan_and_writes_insights(client, datasets, kb_with_docs):
    llm = llm_for()
    runner = make_runner(client, llm, kb_with_docs)
    snapshot = runner.start(datasets["demo"])
    plan = snapshot.request.plan
    assert plan.planner == "llm" and plan.goal == "Динамика и структура операций"
    assert [c.title for c in plan.charts][:2] == ["Количество операций по месяцам", "Объём по месяцам"]
    assert LLM_OFF not in snapshot.warnings

    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    assert all(i.generated_by == "llm" for i in result.insights) and all(c.attempts == 1 and c.passed for c in result.insight_checks)
    assert len(llm.prompts("InsightDraft")) == 4  # по одному обращению на график


def test_user_request_is_interpreted_by_the_llm(client, datasets):
    llm = llm_for()
    runner = make_runner(client, llm)
    snapshot = runner.start(datasets["demo"], "Покажи успешность и сделай прогноз")
    assert [c.analysis for c in snapshot.request.plan.charts] == ["success_dynamics"]  # прогноз в MVP не поддерживается
    assert "Покажи успешность" in llm.prompts("IntentDraft")[0][1]
    assert any("не поддерживается в MVP: прогноз" in n for n in snapshot.request.plan.notes)


def test_llm_never_receives_table_rows_or_unlisted_numbers(client, datasets):
    llm = llm_for()
    runner = make_runner(client, llm)
    snapshot = runner.start(datasets["demo"])
    runner.resume(snapshot.thread_id, approve(snapshot))
    plan_prompt = llm.prompts("PlanDraft")[0][1]
    middle_row = sorted(r["Amount_KZT"] for r in ROWS)[9]  # значение строки, не совпадающее с минимумом или максимумом
    assert "Month" in plan_prompt and str(middle_row) not in plan_prompt  # только сводка датасета, без строк таблицы
    insight_prompt = llm.prompts("InsightDraft")[2][1]
    assert "Числовые доказательства" in insight_prompt and "97,66%" in insight_prompt


def test_llm_plan_is_corrected_by_visualization_rules(client, datasets):
    def bad_plan(system, user, n):
        return PlanDraft(
            goal="x",
            charts=[
                ChartDraft(analysis="success_dynamics", title="Успешность (круговая)", chart_type="pie", x_column="Month", rationale="?"),
                ChartDraft(analysis="success_dynamics", title="Дубль", chart_type="line", x_column="Month", rationale="?"),
                ChartDraft(analysis="channel_distribution", title="Каналы", chart_type="line", x_column="Channel", rationale="?"),
            ],
        )

    plan = make_runner(client, llm_for(plan=bad_plan)).start(datasets["demo"]).request.plan
    by_id = {c.analysis: c for c in plan.charts}
    assert by_id["success_dynamics"].chart_type == "line"  # круговая диаграмма для динамики запрещена
    assert by_id["channel_distribution"].chart_type == "pie"  # линия для категорий заменена
    assert len(plan.charts) == 4 and any("круговая диаграмма не применяется к динамике" in n for n in plan.notes)
    assert any("Повторное предложение" in n for n in plan.notes)


def test_llm_cannot_add_unavailable_or_unknown_columns(client, workspace):
    frame = pd.DataFrame({"Month": ["2026-01", "2026-02", "2026-03"], "Transactions": [100, 110, 120], "Successful": [98, 99, 100]})

    def greedy(system, user, n):
        return PlanDraft(
            goal="x",
            charts=[
                ChartDraft(analysis="volume_dynamics", title="Объём", chart_type="line", x_column="Month", rationale="?"),
                ChartDraft(analysis="count_dynamics", title="Количество", chart_type="line", x_column="Nonexistent", rationale="?"),
            ],
        )

    plan = make_runner(client, llm_for(plan=greedy)).start(save_dataset(workspace, frame, "small.csv")).request.plan
    assert [c.analysis for c in plan.charts] == ["count_dynamics", "success_dynamics"]
    assert plan.charts[0].x_column == "Month"  # выдуманная колонка отброшена
    assert {e.analysis for e in plan.excluded} == {"volume_dynamics", "channel_distribution"}


def test_llm_failure_falls_back_to_rules_with_a_warning(client, datasets):
    def failing(system, user, n):
        return LLMError("Превышен лимит запросов OpenAI. Повторите позже.")

    llm = ScriptedLLM(PlanDraft=failing, InsightDraft=failing)
    runner = make_runner(client, llm)
    snapshot = runner.start(datasets["demo"])
    assert snapshot.request.plan.planner == "rules"
    assert any("План составлен по правилам: Превышен лимит" in w for w in snapshot.warnings)
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    assert result.insights and all(i.generated_by == "rules" for i in result.insights)
    assert any("составлен по правилам" in w and "Динамика" in w for w in result.warnings)


def test_invented_numbers_are_rejected_and_regenerated_with_feedback(client, datasets):
    def insight(system, user, n):
        if "Успешность по месяцам" not in user:
            return good_insight(system, user, n)
        if n <= 4:  # первая попытка по графику успешности — с выдуманным числом
            draft = good_insight(system, user, n)
            return draft.model_copy(update={"summary": draft.summary + " Успешность выросла до 99,9%."})
        return good_insight(system, user, n)

    llm = llm_for(insight=insight)
    runner = make_runner(client, llm)
    snapshot = runner.start(datasets["demo"])
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    check = next(c for c in result.insight_checks if c.chart_id == "success_dynamics")
    assert check.passed and check.attempts == 2 and not check.fallback
    assert any("[numbers-grounded]" in v and "99,9" in v for v in check.violations)
    assert any("Предыдущий ответ отклонён" in u and "99,9" in u for _, u in llm.prompts("InsightDraft"))
    text = next(i for i in result.insights if i.chart_id == "success_dynamics").text
    assert "99,9" not in text


def test_persistently_invalid_insight_is_replaced_by_a_calculated_one(client, datasets):
    def invented(system, user, n):
        draft = good_insight(system, user, n)
        return draft.model_copy(update={"summary": "Успешность достигла 123,45%, потому что мы так решили."})

    runner = make_runner(client, llm_for(insight=invented))
    snapshot = runner.start(datasets["demo"])
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    assert all(c.fallback and c.attempts == 2 and c.violations for c in result.insight_checks)
    assert all(i.generated_by == "rules" and "123,45" not in i.text for i in result.insights)
    assert any("заменён расчётным текстом" in w for w in result.warnings)


def test_causal_claim_from_the_llm_is_not_published(client, datasets, kb_with_docs):
    def causal(system, user, n):
        facts = fact_values(user)
        if "drop" not in facts:
            return good_insight(system, user, n)
        return InsightDraft(
            title="Падение успешности", severity="warning", limitation=None, context_chunk_id=None, context_quote=None,
            summary=f"Успешность упала на {facts['drop']}, потому что обновление инфраструктуры вызвало снижение.",
            evidence_ids=["drop"],
        )

    runner = make_runner(client, llm_for(insight=causal), kb_with_docs)
    snapshot = runner.start(datasets["demo"])
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    success = next(c for c in result.insight_checks if c.chart_id == "success_dynamics")
    assert success.fallback and any("[no-causal-claim]" in v for v in success.violations)
    assert all("вызвал" not in i.text for i in result.insights)


def test_llm_context_is_accepted_only_with_a_verbatim_quote(client, datasets, kb_with_docs):
    definition_search = kb_with_docs.search_business_context("контекстное событие обновление инфраструктуры")
    event = next(h for h in definition_search.hits if "плановое обновление" in h.text)

    def with_context(quote):
        def insight(system, user, n):
            draft = good_insight(system, user, n)
            if "first" not in fact_values(user):
                return draft
            return draft.model_copy(
                update={
                    "summary": draft.summary + " Изменение совпадает по времени с событием из документации.",
                    "context_chunk_id": event.source.chunk_id, "context_quote": quote,
                }
            )

        return insight

    runner = make_runner(client, llm_for(insight=with_context("плановое обновление инфраструктуры обработки транзакций")), kb_with_docs)
    snapshot = runner.start(datasets["demo"])
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    insight = next(i for i in result.insights if i.chart_id == "success_dynamics")
    assert "Контекстное событие" in insight.context_source and "не доказывает причинную связь" in insight.limitation

    fake = make_runner(client, llm_for(insight=with_context("обновление вызвало падение на 40%")), kb_with_docs)
    snapshot = fake.start(datasets["demo"])
    result = fake.resume(snapshot.thread_id, approve(snapshot)).result
    check = next(c for c in result.insight_checks if c.chart_id == "success_dynamics")
    assert check.fallback and any("[context-not-in-source]" in v for v in check.violations)


def test_document_events_are_offered_only_to_charts_with_a_material_drop(client, datasets, kb_with_docs):
    llm = llm_for()
    runner = make_runner(client, llm, kb_with_docs)
    snapshot = runner.start(datasets["demo"])
    runner.resume(snapshot.thread_id, approve(snapshot))
    prompts = [u for _, u in llm.prompts("InsightDraft")]
    assert len(prompts) == 4 and sum("плановое обновление" in u for u in prompts) == 1  # только график успешности с мартовским падением
    assert "плановое обновление" in next(u for u in prompts if "Успешность по месяцам" in u)
