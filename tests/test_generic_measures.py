"""Наборы данных без колонок платёжной системы (продажи): анализ суммы числовой колонки тем же MCP и тем же графом."""
import pandas as pd
import pytest

from datastory.analytics.formatting import format_metric_value
from datastory.analytics.metrics import SUM_PREFIX, available_metrics, get_metric, sum_metric
from datastory.chat.classify import classify_by_rules, detect_metric
from datastory.chat.graph import ChatService
from datastory.errors import AnalyticsError, LLMError
from datastory.workflow.catalog import evaluate_candidates
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.models import AUTO_GOAL, IntentDraft
from scripts import generate_sales_demo as sales
from tests.helpers import ScriptedLLM
from tests.workflow_helpers import approve, make_runner, save_dataset

ROWS = sales.build_rows()
GOAL = "Покажи динамику выручки по месяцам, сравни регионы и найди основные изменения"


def total(rows, key="Revenue_KZT") -> int:
    return sum(r[key] for r in rows)


@pytest.fixture
def sales_id(workspace, client):
    return save_dataset(workspace, pd.DataFrame(ROWS), "sales_2026.xlsx")


# ================================================================== показатель «сумма колонки»
def test_sum_metric_is_described_from_the_column_name():
    revenue = sum_metric("Revenue_KZT")
    assert (revenue.name, revenue.label, revenue.unit, revenue.formula) == ("sum:Revenue_KZT", "Revenue", "KZT", "SUM(Revenue_KZT)")
    orders = get_metric("sum:Orders")
    assert (orders.label, orders.unit, orders.columns) == ("Orders", "", ("Orders",))
    assert orders.compute({"Orders": 42}) == 42


def test_malformed_sum_metric_is_rejected():
    for name in ("sum:", "sum", "total:Revenue"):
        with pytest.raises(AnalyticsError, match="Неизвестная метрика"):
            get_metric(name)


def test_sums_are_offered_only_without_payment_columns():
    assert available_metrics(["Orders", "Revenue_KZT"]) == ["sum:Orders", "sum:Revenue_KZT"]
    assert available_metrics(["Transactions", "Successful", "Orders"]) == ["transaction_count", "successful_count", "success_rate"]
    assert available_metrics([]) == []


def test_mcp_calculates_sums_by_period_and_by_category(client, sales_id):
    by_month = client.calculate_metrics(sales_id, "sum:Revenue_KZT", ["Month"])
    assert [r.value for r in by_month.rows] == [total([r for r in ROWS if r["Month"] == m]) for m in sales.MONTHS]
    assert by_month.overall.value == total(ROWS) and by_month.unit == "KZT" and by_month.formula == "SUM(Revenue_KZT)"
    by_region = client.calculate_metrics(sales_id, "sum:Revenue_KZT", ["Region"])
    assert {r.group["Region"]: r.value for r in by_region.rows} == {reg: total([r for r in ROWS if r["Region"] == reg]) for reg in sales.REGIONS}


def test_mcp_rejects_a_sum_of_a_missing_or_text_column(client, sales_id):
    from datastory.mcp_client.connection import McpToolError

    with pytest.raises(McpToolError, match="нет колонок"):
        client.calculate_metrics(sales_id, "sum:Profit", ["Month"])
    with pytest.raises(McpToolError, match="числовой"):
        client.calculate_metrics(sales_id, "sum:Region", ["Month"])


# ================================================================== workflow на продажах
def test_sales_dataset_gets_generic_analyses_instead_of_payment_ones(client, sales_id):
    snapshot = make_runner(client).start(sales_id, AUTO_GOAL)
    plan = snapshot.request.plan
    assert [(c.analysis, c.chart_type, c.metric, c.x_column) for c in plan.charts] == [
        ("measure_dynamics", "line", "sum:Revenue_KZT", "Month"), ("measure_comparison", "bar", "sum:Revenue_KZT", "Region"),
    ]
    assert [c.title for c in plan.charts] == ["Динамика показателя: Revenue", "Сравнение по категориям: Revenue"]
    assert plan.excluded == [] and all(c.rationale for c in plan.charts)  # платёжные анализы не предлагаются и не «исключаются»
    assert not any(m.documented for m in plan.metrics) and plan.metrics[0].formula == "SUM(Revenue_KZT)"


def test_revenue_is_chosen_as_the_primary_measure(client, sales_id):
    candidates = evaluate_candidates(client.profile_dataset(sales_id))
    assert {c.metric for c in candidates} == {"sum:Revenue_KZT"}  # а не Orders


def test_sales_analysis_runs_end_to_end_with_numbers_from_mcp(client, sales_id):
    runner = make_runner(client)
    snapshot = runner.start(sales_id, GOAL)
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    assert [k.label for k in result.kpis] == ["Revenue: итог"] and result.kpis[0].value == format_metric_value(total(ROWS), "KZT")
    assert [c.kind for c in result.charts] == ["line", "bar"]
    dynamics, comparison = result.insights
    first, last = total([r for r in ROWS if r["Month"] == "2026-01"]), total([r for r in ROWS if r["Month"] == "2026-06"])
    assert format_metric_value(first, "KZT") in dynamics.text and format_metric_value(last, "KZT") in dynamics.text
    almaty = total([r for r in ROWS if r["Region"] == "Almaty"])
    assert "«Almaty»" in comparison.text and format_metric_value(almaty, "KZT") in comparison.text
    assert all(c.passed for c in result.insight_checks) and "SUM(Revenue_KZT)" in dynamics.limitation  # формула — правило пользователя
    assert "Причина изменения по данным не установлена" in dynamics.limitation


def test_llm_intent_is_limited_to_the_analyses_of_the_dataset(client, sales_id):
    def no_plan(system, user, n):
        return LLMError("план по правилам")  # план строится правилами: проверяется только разбор запроса

    llm = ScriptedLLM(
        IntentDraft=lambda s, u, n: IntentDraft(analyses=["success_dynamics", "measure_comparison"], unsupported=[], comment="Регионы."),
        PlanDraft=no_plan,
    )
    snapshot = make_runner(client, llm).start(sales_id, "Сравни регионы")
    assert [c.analysis for c in snapshot.request.plan.charts] == ["measure_comparison"]  # платёжный анализ чужого набора отброшен
    system = llm.prompts("IntentDraft")[0][0]
    assert "measure_dynamics" in system and "success_dynamics" not in system


# ================================================================== чат по продажам
def test_chat_questions_about_revenue_and_regions(client, sales_id):
    runner = make_runner(client)
    snapshot = runner.start(sales_id, GOAL)
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    chat = ChatService(WorkflowDeps.of(client, None, None))
    client.calls.clear()

    top = chat.ask("Какой регион имеет наибольшую выручку?", result)
    assert top.intent == "top_category" and top.metrics == ["sum:Revenue_KZT"] and top.answer.index("Almaty") < top.answer.index("Astana")
    change = chat.ask("Как изменилась выручка?", result)
    assert change.intent == "metric_change" and change.metrics == ["sum:Revenue_KZT"]
    peak = chat.ask("В какой месяц выручка максимальна?", result)
    assert "2026-06" in peak.answer and format_metric_value(total([r for r in ROWS if r["Month"] == "2026-06"]), "KZT") in peak.answer
    assert client.calls == []  # все расчёты уже были на дашборде


def test_chat_new_revenue_calculation_goes_to_mcp(client, sales_id):
    runner = make_runner(client)
    snapshot = runner.start(sales_id, AUTO_GOAL)
    plan = snapshot.request.plan
    result = runner.resume(snapshot.thread_id, approve(snapshot).model_copy(update={"approved_chart_ids": ["measure_comparison"]})).result
    client.calls.clear()
    answer = ChatService(WorkflowDeps.of(client, None, None)).ask("Как изменилась выручка?", result)  # ряда по месяцам на дашборде нет
    assert client.calls == ["calculate_metrics"] and "MCP calculate_metrics" in answer.tools
    assert format_metric_value(total([r for r in ROWS if r["Month"] == "2026-01"]), "KZT") in answer.answer
    assert plan.charts[0].analysis == "measure_dynamics"


def test_revenue_words_map_to_the_revenue_column():
    available = ["sum:Orders", "sum:Revenue_KZT"]
    assert detect_metric("Как изменилась выручка?", available) == "sum:Revenue_KZT"
    assert detect_metric("Сколько было заказов?", available) == "sum:Orders"
    assert detect_metric("Какая сумма Revenue_KZT по месяцам?", available) == "sum:Revenue_KZT"
    assert detect_metric("Как изменилась успешность?", available) is None
    assert classify_by_rules("Какой регион лидирует?")[0] == "top_category"


# ================================================================== демо-файл
def test_sales_demo_data_is_reproducible_and_has_the_astana_dip():
    assert sales.build_rows() == sales.build_rows() and len(ROWS) == 24
    astana = {r["Month"]: r["Revenue_KZT"] / r["Orders"] for r in ROWS if r["Region"] == "Astana"}
    assert astana["2026-04"] < 0.7 * astana["2026-03"]  # заметный провал выручки на заказ в апреле
    assert SUM_PREFIX == "sum:"
