"""Сервер MCP (FastMCP) в процессе теста: набор инструментов, схемы, результаты и обработка ошибок."""
import asyncio

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from datastory.analytics.models import DatasetSummary, MetricResult
from datastory.file_processing.loader import read_table
from datastory.mcp_server import server
from datastory.models import ChartSpec
from datastory.profiler.profiler import build_profile
from datastory.storage.store import DatasetStore
from scripts import generate_demo_data as gen

DEMO = gen.DEFAULT_OUT


@pytest.fixture
def dataset_id() -> str:
    raw = read_table((DEMO / "transactions_2026.xlsx").read_bytes(), "transactions_2026.xlsx", "transactions")
    profile, typed = build_profile(raw, "transactions_2026.xlsx", "transactions")
    DatasetStore().save(typed, profile)  # каталог хранилища задаёт conftest через WORKSPACE_DIR
    return profile.dataset_id


def call(name: str, arguments: dict):
    content, structured = asyncio.run(server.mcp.call_tool(name, arguments))
    return content, structured


def tools():
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


# ------------------------------------------------------------------ набор и схемы инструментов
def test_exactly_three_tools_in_documented_order():
    assert list(tools()) == ["profile_dataset", "calculate_metrics", "create_chart_spec"]
    assert server.mcp.name == "datastory-analytics"


def test_every_tool_is_documented_and_has_structured_output_schema():
    for tool in tools().values():
        assert tool.description and len(tool.description) > 60
        assert tool.outputSchema and tool.outputSchema["type"] == "object"


def test_required_and_optional_parameters():
    schemas = {name: tool.inputSchema for name, tool in tools().items()}
    assert schemas["profile_dataset"]["required"] == ["dataset_id"]
    assert schemas["calculate_metrics"]["required"] == ["dataset_id", "metric"]
    assert {"group_by", "filters"} <= set(schemas["calculate_metrics"]["properties"])
    assert schemas["create_chart_spec"]["required"] == ["metric_result", "chart_type", "title", "x_axis"]
    assert "y_axis" in schemas["create_chart_spec"]["properties"]


def test_metric_and_chart_type_are_enums_in_the_schema():
    schemas = {name: tool.inputSchema for name, tool in tools().items()}
    assert schemas["calculate_metrics"]["properties"]["metric"]["enum"] == [
        "transaction_count", "successful_count", "failed_count", "transaction_volume", "success_rate", "average_transaction_amount",
    ]
    assert schemas["create_chart_spec"]["properties"]["chart_type"]["enum"] == ["line", "bar", "pie"]


def test_metric_description_states_the_formulas():
    description = tools()["calculate_metrics"].inputSchema["properties"]["metric"]["description"]
    assert "SUM(Successful) / SUM(Transactions) × 100" in description
    assert "SUM(Amount_KZT) / SUM(Transactions)" in description


def test_server_has_no_image_or_binary_output_types():
    output_text = str([t.outputSchema for t in tools().values()]).lower()
    for word in ("image", "png", "svg", "base64", "binary"):
        assert word not in output_text


# ------------------------------------------------------------------ вызовы
def test_profile_dataset_returns_summary(dataset_id):
    content, structured = call("profile_dataset", {"dataset_id": dataset_id})
    summary = DatasetSummary.model_validate(structured)
    assert summary.row_count == 18 and summary.available_metrics
    assert [block.type for block in content] == ["text"]


def test_calculate_metrics_returns_metric_result(dataset_id):
    _, structured = call("calculate_metrics", {"dataset_id": dataset_id, "metric": "success_rate", "group_by": ["Month"], "filters": {"Channel": "Web"}})
    result = MetricResult.model_validate(structured)
    assert result.metric == "success_rate" and len(result.rows) == 6 and result.filters[0].column == "Channel"


def test_create_chart_spec_returns_chart_spec_without_image(dataset_id):
    _, metric = call("calculate_metrics", {"dataset_id": dataset_id, "metric": "transaction_volume", "group_by": ["Channel"]})
    content, structured = call("create_chart_spec", {"metric_result": metric, "chart_type": "pie", "title": "Доли", "x_axis": "Channel"})
    spec = ChartSpec.model_validate(structured)
    assert spec.kind == "pie" and len(spec.data) == 3
    assert {block.type for block in content} == {"text"}


# ------------------------------------------------------------------ ошибки
@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"metric": "revenue"}, "Неизвестная метрика"),
        ({"metric": "success_rate", "group_by": ["Nope"]}, "не найдена в датасете"),
        ({"metric": "success_rate", "filters": [{"column": "Channel", "op": "like", "value": "M"}]}, "неизвестный оператор"),
        ({"metric": "success_rate", "filters": [{"column": "Month", "value": "вчера"}]}, "YYYY-MM"),
    ],
)
def test_bad_requests_become_clear_tool_errors(dataset_id, arguments, fragment):
    with pytest.raises(ToolError, match=fragment):
        call("calculate_metrics", {"dataset_id": dataset_id, **arguments})


def test_unknown_or_malformed_dataset_id(dataset_id):
    for bad in ("ffffffffffff", "../../etc/passwd", "", "x" * 40):
        with pytest.raises(ToolError, match="Датасет|Некорректный идентификатор"):
            call("profile_dataset", {"dataset_id": bad})


def test_wrong_argument_types_are_rejected_by_the_schema(dataset_id):
    with pytest.raises(ToolError):
        call("calculate_metrics", {"dataset_id": dataset_id, "metric": 5})
    with pytest.raises(ToolError):
        call("calculate_metrics", {"dataset_id": dataset_id, "metric": "success_rate", "group_by": 7})


def test_chart_errors_are_reported(dataset_id):
    _, ratio = call("calculate_metrics", {"dataset_id": dataset_id, "metric": "success_rate", "group_by": ["Channel"]})
    with pytest.raises(ToolError, match="отношение"):
        call("create_chart_spec", {"metric_result": ratio, "chart_type": "pie", "title": "T", "x_axis": "Channel"})
    with pytest.raises(ToolError, match="Неизвестный тип графика"):
        call("create_chart_spec", {"metric_result": ratio, "chart_type": "3d-donut", "title": "T", "x_axis": "Channel"})
    with pytest.raises(ToolError):
        call("create_chart_spec", {"metric_result": {"not": "a result"}, "chart_type": "bar", "title": "T", "x_axis": "Channel"})


def test_unexpected_failures_do_not_leak_internal_details(dataset_id, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("секретный путь C:\\internal\\keys.txt")

    monkeypatch.setattr(server, "calculate_metric", boom)
    with pytest.raises(ToolError) as exc:
        call("calculate_metrics", {"dataset_id": dataset_id, "metric": "success_rate"})
    assert "Внутренняя ошибка сервера" in str(exc.value)
    assert "секретный" not in str(exc.value) and "keys.txt" not in str(exc.value)
