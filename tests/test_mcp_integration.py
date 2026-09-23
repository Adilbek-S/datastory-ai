"""Интеграционные тесты: реальный MCP-клиент, транспорт stdio, сервер в отдельном процессе.

Каждый вызов проходит весь путь: клиент -> JSON-RPC по stdio -> процесс сервера -> расчёт -> ответ.
Ожидаемые значения считаются независимо, простым Python по исходным строкам генератора демо-данных.
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from datastory.analytics.models import DatasetSummary, MetricResult
from datastory.config import Settings
from datastory.mcp_client.analytics_client import AnalyticsMcpClient, analytics_server_config
from datastory.mcp_client.connection import McpConnection, McpConnectionError, McpServerConfig, McpToolError
from datastory.models import ChartSpec
from datastory.visualization.engine import build_figure
from scripts import generate_demo_data as gen

ROWS = gen.build_rows()
EXPECTED = json.loads((gen.DEFAULT_OUT / "expected_metrics.json").read_text(encoding="utf-8"))
FAKES = Path(__file__).resolve().parent / "fake_servers"


def py_success_rate(rows) -> float:
    return sum(r["Successful"] for r in rows) / sum(r["Transactions"] for r in rows) * 100


def py_average(rows) -> float:
    return sum(r["Amount_KZT"] for r in rows) / sum(r["Transactions"] for r in rows)


# ================================================================== настоящий MCP: набор инструментов
def test_real_mcp_server_lists_exactly_three_tools(client):
    assert client.list_tools() == ["profile_dataset", "calculate_metrics", "create_chart_spec"]


def test_server_runs_in_a_separate_process_over_stdio(client):
    config = analytics_server_config(Settings(_env_file=None))
    assert config.args[0] == "-m" and config.command == sys.executable  # дочерний процесс, а не вызов функции
    assert client.connection.is_connected and client.connection.start_count == 1


# ================================================================== Tool 1: profile_dataset
def test_profile_dataset_over_mcp(client, datasets):
    summary = client.profile_dataset(datasets["demo"])
    assert isinstance(summary, DatasetSummary)
    assert summary.row_count == 18 and summary.column_names == gen.COLUMNS
    assert summary.column_types["Month"] == "datetime" and summary.column_types["Channel"] == "categorical"
    assert summary.numeric_measures == ["Transactions", "Successful", "Failed", "Amount_KZT"]
    assert summary.dimensions == ["Month", "Channel"]
    assert len(summary.available_metrics) == 6
    stats = next(c for c in summary.columns if c.name == "Transactions").statistics
    assert stats["min"] == min(r["Transactions"] for r in ROWS) and stats["max"] == max(r["Transactions"] for r in ROWS)
    assert summary.missing_cell_count == 0 and summary.missing_by_column == {}


def test_profile_dataset_hides_personal_data_over_mcp(client, datasets):
    summary = client.profile_dataset(datasets["clients"])
    assert summary.sensitive_columns == ["email"] and "email" not in summary.dimensions
    assert "a@b.kz" not in summary.model_dump_json()


# ================================================================== Tool 2: calculate_metrics
def test_success_rate_over_mcp_is_sum_over_sum(client, datasets):
    overall = client.calculate_metrics(datasets["demo"], "success_rate")
    assert isinstance(overall, MetricResult)
    assert overall.overall.value == py_success_rate(ROWS)
    assert overall.overall.value == pytest.approx(EXPECTED["totals"]["success_rate_pct"], abs=1e-4)

    by_month = client.calculate_metrics(datasets["demo"], "success_rate", ["Month"])
    assert [r.group["Month"] for r in by_month.rows] == gen.MONTHS
    for row in by_month.rows:
        assert row.value == py_success_rate([r for r in ROWS if r["Month"] == row.group["Month"]])
        assert row.value == pytest.approx(EXPECTED["by_month"][row.group["Month"]]["success_rate_pct"], abs=1e-4)
    march = next(r for r in by_month.rows if r.group["Month"] == "2026-03")
    assert march.value == min(r.value for r in by_month.rows)  # мартовское снижение видно


def test_average_transaction_amount_over_mcp_is_sum_over_sum(client, datasets):
    by_channel = client.calculate_metrics(datasets["demo"], "average_transaction_amount", ["Channel"])
    for row in by_channel.rows:
        assert row.value == py_average([r for r in ROWS if r["Channel"] == row.group["Channel"]])
        assert row.value == pytest.approx(EXPECTED["by_channel"][row.group["Channel"]]["avg_transaction_amount_kzt"], abs=1e-3)
    overall = client.calculate_metrics(datasets["demo"], "average_transaction_amount")
    assert overall.overall.value == py_average(ROWS)
    assert overall.overall.value != sum(r.value for r in by_channel.rows) / 3  # не среднее средних


@pytest.mark.parametrize(
    ("metric", "key"),
    [("transaction_count", "Transactions"), ("successful_count", "Successful"), ("failed_count", "Failed"), ("transaction_volume", "Amount_KZT")],
)
def test_sum_metrics_over_mcp(client, datasets, metric, key):
    assert client.calculate_metrics(datasets["demo"], metric).overall.value == sum(r[key] for r in ROWS)


def test_filters_and_group_by_over_mcp(client, datasets):
    filters = [{"column": "Channel", "op": "in", "value": ["Mobile", "Web"]}, {"column": "Month", "op": "gte", "value": "2026-03"}]
    result = client.calculate_metrics(datasets["demo"], "success_rate", ["Month"], filters)
    selected = [r for r in ROWS if r["Channel"] in ("Mobile", "Web") and r["Month"] >= "2026-03"]
    assert [r.group["Month"] for r in result.rows] == ["2026-03", "2026-04", "2026-05", "2026-06"]
    assert result.overall.value == py_success_rate(selected)
    shorthand = client.calculate_metrics(datasets["demo"], "transaction_count", filters={"Channel": "API"})
    assert shorthand.overall.value == sum(r["Transactions"] for r in ROWS if r["Channel"] == "API")


# ================================================================== Tool 3: create_chart_spec
def test_chart_spec_over_mcp_and_plotly_rendering(client, datasets):
    metric = client.calculate_metrics(datasets["demo"], "success_rate", ["Month", "Channel"])
    spec = client.create_chart_spec(metric, "line", "Success Rate по каналам", "Month")
    assert isinstance(spec, ChartSpec) and spec.kind == "line" and spec.color == "Channel" and len(spec.data) == 18
    figure = build_figure(pd.DataFrame(), spec)  # рисует Streamlit/Plotly, сервер картинок не отдаёт
    assert len(figure.data) == 3 and figure.layout.title.text == "Success Rate по каналам"

    volume = client.calculate_metrics(datasets["demo"], "transaction_volume", ["Channel"])
    assert client.create_chart_spec(volume, "pie", "Доли", "Channel").kind == "pie"
    assert client.create_chart_spec(volume, "bar", "Объём", "Channel", "transaction_volume").kind == "bar"


def test_server_never_returns_images(client, datasets):
    connection = client.connection
    metric = client.calculate_metrics(datasets["demo"], "transaction_count", ["Channel"])
    for tool, arguments in (
        ("profile_dataset", {"dataset_id": datasets["demo"]}),
        ("calculate_metrics", {"dataset_id": datasets["demo"], "metric": "transaction_count", "group_by": ["Channel"]}),
        ("create_chart_spec", {"metric_result": metric.model_dump(mode="json"), "chart_type": "bar", "title": "T", "x_axis": "Channel"}),
    ):
        raw = connection._request(lambda session, t=tool, a=arguments: session.call_tool(t, a), tool)
        assert not raw.isError and {block.type for block in raw.content} == {"text"}
        assert "image" not in json.dumps(raw.structuredContent).lower()


# ================================================================== ошибки через MCP
@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"metric": "success_rate", "group_by": ["Chanel"]}, "Возможно, вы имели в виду «Channel»"),
        ({"metric": "success_rate", "group_by": ["Channel; DROP TABLE data"]}, "не найдена в датасете"),
        ({"metric": "success_rate", "group_by": ["__import__('os').system('calc')"]}, "не найдена в датасете"),
        ({"metric": "success_rate", "filters": {"Nope": 1}}, "колонка «Nope» не найдена"),
        ({"metric": "success_rate", "filters": [{"column": "Channel", "op": "exec", "value": "x"}]}, "неизвестный оператор"),
        ({"metric": "profit"}, "Неизвестная метрика"),
    ],
)
def test_validation_errors_reach_the_client_as_tool_errors(client, datasets, kwargs, fragment):
    with pytest.raises(McpToolError) as exc:
        client.calculate_metrics(datasets["demo"], **kwargs)
    assert fragment in exc.value.message and exc.value.tool == "calculate_metrics"
    assert not exc.value.message.startswith("Error executing tool")  # служебный префикс убран


def test_injection_payload_in_filter_value_is_just_data(client, datasets):
    result = client.calculate_metrics(datasets["demo"], "transaction_count", filters={"Channel": "__import__('os').system('calc')"})
    assert result.rows_matched == 0 and result.overall.value == 0


def test_sensitive_column_is_protected_over_mcp(client, datasets):
    with pytest.raises(McpToolError, match="персональные данные"):
        client.calculate_metrics(datasets["clients"], "transaction_count", ["email"])
    with pytest.raises(McpToolError, match="персональные данные"):
        client.calculate_metrics(datasets["clients"], "transaction_count", filters={"email": "a@b.kz"})


def test_unknown_dataset_and_metric_without_columns(client, datasets):
    with pytest.raises(McpToolError, match="не найден"):
        client.profile_dataset("ffffffffffff")
    with pytest.raises(McpToolError, match="нельзя посчитать"):
        client.calculate_metrics(datasets["generic"], "success_rate")


def test_chart_errors_over_mcp(client, datasets):
    ratio = client.calculate_metrics(datasets["demo"], "success_rate", ["Channel"])
    with pytest.raises(McpToolError, match="отношение"):
        client.create_chart_spec(ratio, "pie", "T", "Channel")
    with pytest.raises(McpToolError, match="x_axis"):
        client.create_chart_spec(ratio, "bar", "T", "Month")


def test_tool_errors_do_not_break_the_connection(client, datasets):
    starts = client.connection.start_count
    for _ in range(3):
        with pytest.raises(McpToolError):
            client.calculate_metrics(datasets["demo"], "success_rate", ["Nope"])
        assert client.calculate_metrics(datasets["demo"], "transaction_count").overall.value > 0
    assert client.connection.start_count == starts  # ошибка запроса не перезапускает сервер


# ================================================================== жизненный цикл на реальном сервере
def test_one_server_process_serves_many_calls_and_threads(client, datasets):
    starts = client.connection.start_count

    def job(month: str) -> float:
        rows = [r for r in ROWS if r["Month"] == month]
        value = client.calculate_metrics(datasets["demo"], "success_rate", filters={"Month": month}).overall.value
        assert value == py_success_rate(rows)
        return value

    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(job, gen.MONTHS * 2))
    assert len(values) == 12 and client.connection.start_count == starts


def test_workspace_is_passed_to_the_server_through_the_environment(datasets, tmp_path):
    other = AnalyticsMcpClient.from_settings(Settings(workspace_dir=tmp_path / "another_workspace", _env_file=None))
    try:
        with pytest.raises(McpToolError, match="не найден"):  # у другого сервера другое хранилище
            other.profile_dataset(datasets["demo"])
    finally:
        other.close()


def test_reconnect_after_disconnect_and_terminal_close(workspace, datasets):
    own = AnalyticsMcpClient.from_settings(Settings(workspace_dir=workspace, _env_file=None))
    assert own.profile_dataset(datasets["demo"]).row_count == 18 and own.connection.start_count == 1
    own.connection.disconnect()
    assert not own.connection.is_connected
    assert own.profile_dataset(datasets["demo"]).row_count == 18 and own.connection.start_count == 2

    own.close()
    own.close()  # повторное закрытие безопасно
    assert own.is_closed and not own.connection.is_connected
    with pytest.raises(McpConnectionError, match="закрыто"):
        own.profile_dataset(datasets["demo"])


# ================================================================== жизненный цикл на тестовом сервере
def fake(script: str, **kwargs) -> McpConnection:
    config = McpServerConfig(command=sys.executable, args=(str(FAKES / script),), name="Тестовый сервер")
    return McpConnection(config, **kwargs)


def test_process_is_reused_and_restarted_on_demand():
    connection = fake("lifecycle_server.py", call_timeout=20)
    first = connection.call_tool("pid")["result"]
    assert connection.call_tool("pid")["result"] == first  # тот же процесс

    connection.disconnect()
    second = connection.call_tool("pid")["result"]
    assert second != first and connection.start_count == 2  # новый процесс


def test_server_crash_during_a_call_is_reported_and_recovered():
    connection = fake("lifecycle_server.py", call_timeout=20)
    before = connection.call_tool("pid")["result"]
    with pytest.raises(McpConnectionError, match="потеряна"):
        connection.call_tool("die")  # процесс завершается посреди вызова; повтор тоже падает
    after = connection.call_tool("pid")["result"]  # но соединение восстанавливается
    assert after != before and connection.start_count >= 3


def test_call_timeout_does_not_hang_and_connection_recovers():
    connection = fake("lifecycle_server.py", call_timeout=20)
    stuck = connection.call_tool("pid")["result"]
    with pytest.raises(McpConnectionError, match="не ответил на вызов"):
        connection.call_tool("sleep", {"seconds": 30}, timeout=1)
    assert connection.call_tool("pid")["result"] != stuck  # зависший процесс заменён новым


def test_tool_level_error_from_a_fake_server_is_not_a_connection_error():
    connection = fake("lifecycle_server.py")
    with pytest.raises(McpToolError, match="ожидаемая ошибка инструмента"):
        connection.call_tool("fail")
    assert connection.call_tool("pid")["result"] > 0 and connection.start_count == 1


def test_startup_timeout_gives_a_clear_error():
    connection = fake("slow_start_server.py", startup_timeout=2)
    with pytest.raises(McpConnectionError) as exc:
        connection.ensure_started()
    assert "«Тестовый сервер»" in exc.value.user_message
    assert "не ответил вовремя" in exc.value.user_message or "не запустился" in exc.value.user_message


def test_server_that_fails_to_start_reports_stderr_details():
    config = McpServerConfig(command=sys.executable, args=("-m", "no_such_module_xyz"), name="Аналитика")
    with pytest.raises(McpConnectionError) as exc:
        McpConnection(config, startup_timeout=10).ensure_started()
    assert "Не удалось подключиться к MCP-серверу «Аналитика»" in exc.value.user_message
    assert "No module named" in exc.value.details  # диагностика сервера — для блока «Технические подробности»


def test_missing_executable_gives_a_clear_error():
    connection = McpConnection(McpServerConfig(command="no-such-binary-xyz", name="Аналитика"), startup_timeout=5)
    with pytest.raises(McpConnectionError, match="исполняемый файл сервера не найден"):
        connection.ensure_started()


def test_failed_start_can_be_retried_after_the_cause_is_fixed(tmp_path):
    connection = McpConnection(McpServerConfig(command="no-such-binary-xyz"), startup_timeout=5)
    for _ in range(2):
        with pytest.raises(McpConnectionError):
            connection.ensure_started()  # ошибка не «залипает» и не оставляет висящих потоков
    assert not connection.is_connected and connection.start_count == 0


def test_no_server_processes_are_left_behind_after_close():
    connection = fake("lifecycle_server.py")
    connection.call_tool("pid")
    state_thread = connection._state.thread
    connection.close()
    assert not state_thread.is_alive() and not connection.is_connected
    assert not [t for t in threading.enumerate() if t.name.startswith("mcp-Тестовый") and t.is_alive() and t is state_thread]
