"""create_chart_spec: спецификации line / bar / pie с данными и настройками Plotly (без MCP)."""
import pandas as pd
import pytest

from datastory.analytics.charts import MAX_POINTS, build_chart_spec
from datastory.analytics.metrics import calculate_metric
from datastory.analytics.models import MetricResult, MetricRow
from datastory.errors import AnalyticsError
from datastory.file_processing.loader import read_table
from datastory.models import ChartSpec
from datastory.profiler.profiler import build_profile
from datastory.visualization.engine import build_figure
from scripts import generate_demo_data as gen

ID = "abcdef123456"


@pytest.fixture(scope="module")
def demo():
    raw = read_table((gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes(), "transactions_2026.xlsx")
    profile, typed = build_profile(raw, "transactions_2026.xlsx", dataset_id=ID)
    return typed, profile


def result(demo, metric, group_by, filters=None) -> MetricResult:
    return calculate_metric(demo[0], demo[1], ID, metric, group_by, filters)


# ================================================================== line
def test_line_chart_by_month(demo):
    spec = build_chart_spec(result(demo, "success_rate", ["Month"]), "line", "Success Rate по месяцам", "Month")
    assert isinstance(spec, ChartSpec) and spec.kind == "line"
    assert (spec.x, spec.y, spec.color) == ("Month", "success_rate", None)
    assert [p["Month"] for p in spec.data] == gen.MONTHS
    assert all(set(p) == {"Month", "success_rate"} for p in spec.data)
    assert spec.settings["markers"] is True and spec.settings["category_orders"] == {"Month": gen.MONTHS}
    assert (spec.title, spec.x_title, spec.y_title, spec.unit) == ("Success Rate по месяцам", "Month", "Success Rate, %", "%")
    assert spec.source_metric == "success_rate" and "SUM(Successful) / SUM(Transactions) × 100" in spec.description


def test_line_chart_with_series_from_second_group_column(demo):
    spec = build_chart_spec(result(demo, "success_rate", ["Month", "Channel"]), "line", "SR", "Month")
    assert spec.color == "Channel" and len(spec.data) == 18
    assert spec.settings["category_orders"]["Channel"] == ["API", "Mobile", "Web"]
    assert [(p["Month"], p["Channel"]) for p in spec.data] == sorted((p["Month"], p["Channel"]) for p in spec.data)
    march_web = next(p for p in spec.data if p["Month"] == "2026-03" and p["Channel"] == "Web")
    assert march_web["success_rate"] == pytest.approx(89.9515, abs=1e-3)


def test_x_axis_can_be_the_second_group_column(demo):
    spec = build_chart_spec(result(demo, "transaction_count", ["Month", "Channel"]), "bar", "T", "Channel")
    assert spec.x == "Channel" and spec.color == "Month"
    assert [p["Channel"] for p in spec.data[:6]] == ["API"] * 6  # сначала по оси X, затем по сериям


def test_none_values_are_kept_as_gaps():
    empty = MetricResult(
        dataset_id=ID, metric="success_rate", label="Success Rate", unit="%", formula="f", group_by=["Month"],
        rows=[MetricRow(group={"Month": "2026-01"}, value=90.0), MetricRow(group={"Month": "2026-02"}, value=None)],
    )
    spec = build_chart_spec(empty, "line", "Пропуск", "Month")
    assert [p["success_rate"] for p in spec.data] == [90.0, None]


# ================================================================== bar
def test_bar_chart(demo):
    spec = build_chart_spec(result(demo, "average_transaction_amount", ["Channel"]), "bar", "Средняя сумма", "Channel", "value")
    assert spec.kind == "bar" and spec.y == "average_transaction_amount" and spec.color is None
    assert "barmode" not in spec.settings and spec.y_title == "Средняя сумма транзакции, KZT"
    assert [p["Channel"] for p in spec.data] == ["API", "Mobile", "Web"]


def test_grouped_bars_for_two_dimensions(demo):
    spec = build_chart_spec(result(demo, "transaction_volume", ["Month", "Channel"]), "bar", "Объём", "Month")
    assert spec.settings["barmode"] == "group" and spec.color == "Channel"


# ================================================================== pie
def test_pie_of_additive_metric(demo):
    spec = build_chart_spec(result(demo, "transaction_volume", ["Channel"]), "pie", "Доли объёма", "Channel")
    assert spec.kind == "pie" and spec.color is None and "markers" not in spec.settings
    assert {p["Channel"]: p["transaction_volume"] for p in spec.data} == {
        c: sum(r["Amount_KZT"] for r in gen.build_rows() if r["Channel"] == c) for c in ("Mobile", "Web", "API")
    }


@pytest.mark.parametrize("metric", ["success_rate", "average_transaction_amount"])
def test_pie_is_rejected_for_ratios(demo, metric):
    with pytest.raises(AnalyticsError, match="отношение.*bar или line"):
        build_chart_spec(result(demo, metric, ["Channel"]), "pie", "Доли", "Channel")


def test_pie_needs_one_dimension_and_positive_values(demo):
    with pytest.raises(AnalyticsError, match="по одной колонке"):
        build_chart_spec(result(demo, "transaction_count", ["Month", "Channel"]), "pie", "T", "Month")
    zeros = MetricResult(
        dataset_id=ID, metric="transaction_count", label="n", unit="шт.", formula="f", group_by=["A"],
        rows=[MetricRow(group={"A": "x"}, value=0), MetricRow(group={"A": "y"}, value=None)],
    )
    with pytest.raises(AnalyticsError, match="нет положительных значений"):
        build_chart_spec(zeros, "pie", "T", "A")
    negative = zeros.model_copy(update={"rows": [MetricRow(group={"A": "x"}, value=5), MetricRow(group={"A": "y"}, value=-1)]})
    with pytest.raises(AnalyticsError, match="отрицательные"):
        build_chart_spec(negative, "pie", "T", "A")


def test_pie_drops_undefined_values():
    partial = MetricResult(
        dataset_id=ID, metric="transaction_count", label="n", unit="шт.", formula="f", group_by=["A"],
        rows=[MetricRow(group={"A": "x"}, value=5), MetricRow(group={"A": "y"}, value=None)],
    )
    assert [p["A"] for p in build_chart_spec(partial, "pie", "T", "A").data] == ["x"]


# ================================================================== проверка входа
def test_y_axis_accepts_value_or_metric_name_only(demo):
    r = result(demo, "success_rate", ["Month"])
    assert build_chart_spec(r, "line", "T", "Month", "value").y == build_chart_spec(r, "line", "T", "Month", "success_rate").y
    with pytest.raises(AnalyticsError, match="y_axis"):
        build_chart_spec(r, "line", "T", "Month", "Transactions")


@pytest.mark.parametrize(
    ("chart_type", "title", "x", "fragment"),
    [
        ("scatter", "T", "Month", "Неизвестный тип графика"),
        ("", "T", "Month", "Неизвестный тип графика"),
        ("line", "   ", "Month", "непустое название"),
        ("line", "x" * 201, "Month", "не более 200"),
        ("line", "T", "Channel", "x_axis: колонка «Channel» отсутствует в group_by"),
        ("line", "T", "Nope", "отсутствует в group_by"),
    ],
)
def test_invalid_chart_arguments(demo, chart_type, title, x, fragment):
    with pytest.raises(AnalyticsError, match=fragment):
        build_chart_spec(result(demo, "success_rate", ["Month"]), chart_type, title, x)


def test_chart_requires_grouped_result(demo):
    with pytest.raises(AnalyticsError, match="нужен результат calculate_metrics с group_by"):
        build_chart_spec(result(demo, "success_rate", None), "bar", "T", "Month")


def test_too_many_dimensions_and_points(demo):
    three = result(demo, "transaction_count", ["Month", "Channel", "Transactions"])
    with pytest.raises(AnalyticsError, match="одной или двум"):
        build_chart_spec(three, "bar", "T", "Month")
    many = MetricResult(
        dataset_id=ID, metric="transaction_count", label="n", unit="шт.", formula="f", group_by=["A"],
        rows=[MetricRow(group={"A": str(i)}, value=i) for i in range(MAX_POINTS + 1)],
    )
    with pytest.raises(AnalyticsError, match="Слишком много точек"):
        build_chart_spec(many, "bar", "T", "A")


def test_description_mentions_filters(demo):
    spec = build_chart_spec(result(demo, "success_rate", ["Month"], {"Channel": "Web"}), "line", "T", "Month")
    assert "Фильтры: Channel eq Web" in spec.description


# ================================================================== сериализация и отрисовка
def test_spec_is_plain_json_and_roundtrips(demo):
    spec = build_chart_spec(result(demo, "success_rate", ["Month", "Channel"]), "line", "T", "Month")
    restored = ChartSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert all(type(v) in (str, int, float, type(None)) for point in spec.data for v in point.values())


@pytest.mark.parametrize(
    ("metric", "group_by", "kind", "x", "traces", "trace_type"),
    [
        ("success_rate", ["Month"], "line", "Month", 1, "scatter"),
        ("success_rate", ["Month", "Channel"], "line", "Month", 3, "scatter"),
        ("transaction_count", ["Channel"], "bar", "Channel", 1, "bar"),
        ("transaction_count", ["Month", "Channel"], "bar", "Month", 3, "bar"),
        ("transaction_volume", ["Channel"], "pie", "Channel", 1, "pie"),
    ],
)
def test_plotly_figure_is_built_from_the_spec_alone(demo, metric, group_by, kind, x, traces, trace_type):
    spec = build_chart_spec(result(demo, metric, group_by), kind, "Заголовок", x)
    figure = build_figure(pd.DataFrame(), spec)  # без DataFrame датасета: все данные внутри спецификации
    assert len(figure.data) == traces and {t.type for t in figure.data} == {trace_type}
    assert figure.layout.title.text == "Заголовок"
    if kind == "pie":
        assert sorted(figure.data[0].labels) == ["API", "Mobile", "Web"]
        assert sum(figure.data[0].values) == sum(p["transaction_volume"] for p in spec.data)
    else:
        assert figure.layout.yaxis.title.text == spec.y_title and figure.layout.xaxis.title.text == spec.x_title
        assert sum(len(t.y) for t in figure.data) == len(spec.data)


def test_line_figure_keeps_month_order_and_values(demo):
    r = result(demo, "success_rate", ["Month"])
    figure = build_figure(pd.DataFrame(), build_chart_spec(r, "line", "T", "Month"))
    assert list(figure.data[0].x) == gen.MONTHS
    assert list(figure.data[0].y) == [row.value for row in r.rows]


def test_unknown_plotly_settings_are_ignored(demo):
    spec = build_chart_spec(result(demo, "transaction_count", ["Channel"]), "bar", "T", "Channel")
    spec.settings.update({"template": "evil", "color_discrete_map": {"x": "y"}, "labels": {"a": "b"}})
    figure = build_figure(pd.DataFrame(), spec)  # в px попадают только настройки из белого списка
    assert figure.layout.template.layout.colorway != ("evil",) and len(figure.data) == 1
