"""LangGraph-воркфлоу поверх настоящего MCP: все аналитические шаги идут через вызовы инструментов."""
import ast
from pathlib import Path

import pandas as pd
import pytest

from datastory.analytics.formatting import format_metric_value
from datastory.analytics.models import AnalysisResult
from datastory.config import Settings
from datastory.evaluation.pipeline import evaluate_result
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.mcp_client.connection import McpConnectionError, McpServerConfig, McpConnection, McpToolError
from datastory.models import EvidenceType
from datastory.profiler.profiler import build_profile
from datastory.storage.store import DatasetStore
from datastory.workflow.graph import build_graph, run_analysis_for_dataset
from scripts import generate_demo_data as gen

ROWS = gen.build_rows()
GRAPH_SOURCE = Path(__file__).resolve().parent.parent / "datastory" / "workflow" / "graph.py"


def py_success_rate(rows) -> float:
    return sum(r["Successful"] for r in rows) / sum(r["Transactions"] for r in rows) * 100


@pytest.fixture(scope="module")
def demo_result(client, datasets) -> AnalysisResult:
    client.calls.clear()
    return run_analysis_for_dataset(datasets["demo"], client)


# ================================================================== граф действительно ходит в MCP
def test_workflow_never_imports_calculation_code():
    """Граф не подменяет MCP локальными функциями: расчётные модули в нём не импортируются."""
    tree = ast.parse(GRAPH_SOURCE.read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    forbidden = {"datastory.analytics.metrics", "datastory.analytics.charts", "datastory.analytics.summary", "datastory.storage.store", "pandas"}
    assert not imported & forbidden, imported & forbidden


def test_graph_structure(client):
    nodes = set(build_graph(client).get_graph().nodes)
    assert {"profile", "metrics", "charts", "insights"} <= nodes


def test_all_three_tools_were_called_through_the_client(client, datasets):
    client.calls.clear()
    result = run_analysis_for_dataset(datasets["demo"], client)
    assert client.calls[0] == "profile_dataset" and client.calls.count("profile_dataset") == 1
    assert client.calls.count("calculate_metrics") == len(result.metrics) == 4 + 5  # 4 итоговые + 5 с группировкой
    assert client.calls.count("create_chart_spec") == len(result.charts) == 4
    assert set(client.calls) == {"profile_dataset", "calculate_metrics", "create_chart_spec"}


def test_workflow_uses_the_running_mcp_process(client, datasets):
    starts = client.connection.start_count
    run_analysis_for_dataset(datasets["demo"], client)
    run_analysis_for_dataset(datasets["demo"], client)
    assert client.connection.start_count == starts  # повторные анализы не запускают новых процессов


# ================================================================== результат для датасета платёжной системы
def test_kpis_are_the_overall_metric_values(demo_result):
    kpis = {k.label: k.value for k in demo_result.kpis}
    assert list(kpis) == ["Количество транзакций", "Success Rate", "Объём транзакций", "Средняя сумма транзакции"]
    transactions = sum(r["Transactions"] for r in ROWS)
    assert kpis["Количество транзакций"] == format_metric_value(transactions, "шт.")
    assert kpis["Success Rate"] == format_metric_value(py_success_rate(ROWS), "%")
    assert kpis["Объём транзакций"] == format_metric_value(sum(r["Amount_KZT"] for r in ROWS), "KZT")
    assert kpis["Средняя сумма транзакции"] == format_metric_value(sum(r["Amount_KZT"] for r in ROWS) / transactions, "KZT")
    assert kpis["Success Rate"].endswith("%") and "," in kpis["Success Rate"]


def test_charts_are_created_by_the_tool_with_data_inside(demo_result):
    charts = {c.title: c for c in demo_result.charts}
    assert [c.kind for c in demo_result.charts] == ["line", "pie", "bar", "bar"]
    line = charts["Success Rate по периодам и категориям"]
    assert (line.x, line.color, line.y) == ("Month", "Channel", "success_rate") and len(line.data) == 18
    assert all(c.data and c.source_metric for c in demo_result.charts)
    pie = next(c for c in demo_result.charts if c.kind == "pie")
    assert {p["Channel"] for p in pie.data} == {"Mobile", "Web", "API"}


def test_metric_results_are_kept_for_transparency(demo_result):
    names = [m.metric for m in demo_result.metrics]
    assert names[:4] == ["transaction_count", "success_rate", "transaction_volume", "average_transaction_amount"]
    by_period = next(m for m in demo_result.metrics if m.metric == "success_rate" and m.group_by == ["Month"])
    assert [r.value for r in by_period.rows] == [py_success_rate([r for r in ROWS if r["Month"] == m]) for m in gen.MONTHS]


def test_march_drop_insight_is_computed_and_makes_no_causal_claim(demo_result):
    drop = next(i for i in demo_result.insights if i.title.startswith("Снижение Success Rate"))
    assert drop.title.endswith("2026-03") and drop.severity == "warning"
    assert drop.evidence_type is EvidenceType.COMPUTED
    march = py_success_rate([r for r in ROWS if r["Month"] == "2026-03"])
    others = py_success_rate([r for r in ROWS if r["Month"] != "2026-03"])
    assert format_metric_value(march, "%") in drop.text and format_metric_value(others, "%") in drop.text
    assert "SUM(Successful) / SUM(Transactions) × 100" in drop.text
    assert "не объясняют его причину" in drop.text
    for causal in ("вызван", "из-за обновления", "привело"):
        assert causal not in drop.text


def test_all_insights_are_marked_as_computed(demo_result):
    assert all(i.evidence_type is EvidenceType.COMPUTED for i in demo_result.insights)


def test_evaluation_passes(demo_result):
    cases = evaluate_result(demo_result)
    assert all(c.passed for c in cases), [c for c in cases if not c.passed]


# ================================================================== другие датасеты
def test_dataset_without_payment_columns_gets_basic_result(client, datasets):
    client.calls.clear()
    result = run_analysis_for_dataset(datasets["generic"], client)
    assert [k.label for k in result.kpis] == ["Строк", "Столбцов", "Числовых показателей", "Пропусков"]
    assert result.metrics == [] and result.charts == []
    assert client.calls == ["profile_dataset"]  # метрики платёжной системы не запрашиваются
    assert result.insights[0].title == "Данные чистые"


def test_stable_success_rate_gives_positive_insight(client, workspace):
    frame = pd.DataFrame(
        {"Month": ["2026-01", "2026-02", "2026-03", "2026-04"] * 2, "Channel": ["a"] * 4 + ["b"] * 4,
         "Transactions": [1000] * 8, "Successful": [970, 972, 968, 971, 975, 973, 974, 972]}
    )
    profile, typed = build_profile(frame, "stable.csv")
    DatasetStore(workspace).save(typed, profile)
    insights = run_analysis_for_dataset(profile.dataset_id, client).insights
    assert any(i.title == "Success Rate стабилен" and i.severity == "positive" for i in insights)


def test_too_few_periods_give_no_trend_insight(client, workspace):
    frame = pd.DataFrame({"Month": ["2026-01", "2026-02"], "Transactions": [100, 100], "Successful": [99, 50]})
    profile, typed = build_profile(frame, "short.csv")
    DatasetStore(workspace).save(typed, profile)
    titles = [i.title for i in run_analysis_for_dataset(profile.dataset_id, client).insights]
    assert not any("Success Rate" in t for t in titles)


def test_dataset_without_categories_still_charts_by_period(client, workspace):
    frame = pd.DataFrame({"Month": [f"2026-0{m}" for m in range(1, 5)], "Transactions": [100, 110, 120, 130], "Successful": [98, 99, 100, 101]})
    profile, typed = build_profile(frame, "period_only.csv")
    DatasetStore(workspace).save(typed, profile)
    result = run_analysis_for_dataset(profile.dataset_id, client)
    assert [c.title for c in result.charts] == ["Success Rate по периодам", "Количество транзакций по периодам"]


# ================================================================== ошибки MCP доходят до вызывающего кода
def test_unknown_dataset_is_a_tool_error(client):
    with pytest.raises(McpToolError, match="не найден"):
        run_analysis_for_dataset("ffffffffffff", client)


def test_unavailable_server_raises_a_connection_error_not_a_silent_fallback(workspace, datasets):
    broken = AnalyticsMcpClient(
        McpConnection(McpServerConfig(command="python", args=("-m", "no_such_module_xyz"), name="datastory-analytics"), startup_timeout=10)
    )
    try:
        with pytest.raises(McpConnectionError, match="Не удалось подключиться к MCP-серверу «datastory-analytics»"):
            run_analysis_for_dataset(datasets["demo"], broken)
        assert broken.calls == ["profile_dataset"]  # локального расчёта «на всякий случай» нет
    finally:
        broken.close()


def test_client_from_settings_uses_the_configured_server_module(workspace):
    broken = AnalyticsMcpClient.from_settings(Settings(workspace_dir=workspace, mcp_server_module="no_such_module_xyz", mcp_startup_timeout=10, _env_file=None))
    try:
        with pytest.raises(McpConnectionError) as exc:
            broken.profile_dataset("abcdef123456")
        assert "no_such_module_xyz" in exc.value.user_message
    finally:
        broken.close()
