"""calculate_metrics: формулы, фильтры, группировка и строгая проверка входных данных (без MCP, чистый Python)."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datastory.analytics.metrics import METRICS, calculate_metric
from datastory.errors import AnalyticsError
from datastory.file_processing.loader import read_table
from datastory.profiler.profiler import build_profile
from scripts import generate_demo_data as gen

DEMO = gen.DEFAULT_OUT
ROWS = gen.build_rows()  # исходные строки демо-данных: независимый источник для проверки
EXPECTED = json.loads((DEMO / "expected_metrics.json").read_text(encoding="utf-8"))
ID = "abcdef123456"


# ------------------------------------------------------------------ независимые расчёты «вручную»
def py_sum(key: str, rows=ROWS) -> int:
    return sum(r[key] for r in rows)


def py_success_rate(rows) -> float:
    return sum(r["Successful"] for r in rows) / sum(r["Transactions"] for r in rows) * 100


def py_average_amount(rows) -> float:
    return sum(r["Amount_KZT"] for r in rows) / sum(r["Transactions"] for r in rows)


def select(**conditions):
    return [r for r in ROWS if all(r[k] == v for k, v in conditions.items())]


@pytest.fixture(scope="module")
def demo():
    raw = read_table((DEMO / "transactions_2026.xlsx").read_bytes(), "transactions_2026.xlsx")
    profile, typed = build_profile(raw, "transactions_2026.xlsx", dataset_id=ID)
    return typed, profile


def calc(demo, metric, group_by=None, filters=None):
    return calculate_metric(*demo[:1], demo[1], ID, metric, group_by, filters)


def custom(frame: pd.DataFrame):
    profile, typed = build_profile(frame, "custom.csv", dataset_id=ID)
    return typed, profile


def run(dataset, metric, group_by=None, filters=None):
    return calculate_metric(dataset[0], dataset[1], ID, metric, group_by, filters)


# ================================================================== формула Success Rate
def test_success_rate_is_sum_of_successful_over_sum_of_transactions(demo):
    result = calc(demo, "success_rate")
    assert result.overall.value == py_sum("Successful") / py_sum("Transactions") * 100
    assert result.overall.value == pytest.approx(EXPECTED["totals"]["success_rate_pct"], abs=1e-4)
    assert result.formula == "SUM(Successful) / SUM(Transactions) × 100" and result.unit == "%"
    assert result.overall.components == {"sum_successful": py_sum("Successful"), "sum_transactions": py_sum("Transactions")}


def test_success_rate_by_month_matches_independent_python_and_expected_json(demo):
    result = calc(demo, "success_rate", ["Month"])
    assert [r.group["Month"] for r in result.rows] == gen.MONTHS
    for row in result.rows:
        month = row.group["Month"]
        assert row.value == py_success_rate(select(Month=month))
        assert row.value == pytest.approx(EXPECTED["by_month"][month]["success_rate_pct"], abs=1e-4)


def test_success_rate_by_channel_and_month_channel(demo):
    by_channel = calc(demo, "success_rate", ["Channel"])
    for row in by_channel.rows:
        assert row.value == py_success_rate(select(Channel=row.group["Channel"]))
        assert row.value == pytest.approx(EXPECTED["by_channel"][row.group["Channel"]]["success_rate_pct"], abs=1e-4)

    cells = calc(demo, "success_rate", ["Month", "Channel"])
    assert len(cells.rows) == 18
    expected = {(e["month"], e["channel"]): e for e in EXPECTED["by_month_channel"]}
    assert {(r.group["Month"], r.group["Channel"]) for r in cells.rows} == set(expected)
    for row in cells.rows:
        key = (row.group["Month"], row.group["Channel"])
        assert row.value == pytest.approx(expected[key]["success_rate_pct"], abs=1e-4)
        assert row.value == py_success_rate(select(Month=key[0], Channel=key[1]))
    assert [(r.group["Month"], r.group["Channel"]) for r in cells.rows] == sorted(expected)  # порядок: месяц, затем канал


def test_march_is_the_lowest_success_rate(demo):
    rows = calc(demo, "success_rate", ["Month"]).rows
    assert min(rows, key=lambda r: r.value).group["Month"] == EXPECTED["anomaly"]["lowest_success_rate_month"] == gen.DROP_MONTH


def test_success_rate_is_not_the_mean_of_row_percentages():
    frame = pd.DataFrame({"Transactions": [1000, 9000], "Successful": [500, 9000], "Month": ["2026-01", "2026-01"]})
    result = run(custom(frame), "success_rate")
    mean_of_percentages = (500 / 1000 * 100 + 9000 / 9000 * 100) / 2  # 75 — так считать нельзя
    assert mean_of_percentages == 75.0
    assert result.overall.value == 95.0  # (500 + 9000) / (1000 + 9000) × 100


def test_grouped_totals_are_ratio_of_sums_not_mean_of_group_values():
    frame = pd.DataFrame(
        {"Channel": ["A", "A", "B"], "Transactions": [100, 100, 1000], "Successful": [50, 50, 1000], "Amount_KZT": [1000, 1000, 10000]}
    )
    result = run(custom(frame), "success_rate", ["Channel"])
    assert [r.value for r in result.rows] == [50.0, 100.0]
    assert result.overall.value == 1100 / 1200 * 100  # не (50 + 100) / 2 = 75
    assert result.overall.value != sum(r.value for r in result.rows) / len(result.rows)


# ================================================================== формула Average Transaction Amount
def test_average_transaction_amount_is_sum_of_amount_over_sum_of_transactions(demo):
    result = calc(demo, "average_transaction_amount")
    assert result.overall.value == py_sum("Amount_KZT") / py_sum("Transactions")
    assert result.overall.value == pytest.approx(EXPECTED["totals"]["avg_transaction_amount_kzt"], abs=1e-3)
    assert result.formula == "SUM(Amount_KZT) / SUM(Transactions)" and result.unit == "KZT"


def test_average_amount_by_month_channel_matches_independent_python_and_expected_json(demo):
    by_channel = calc(demo, "average_transaction_amount", ["Channel"])
    for row in by_channel.rows:
        channel = row.group["Channel"]
        assert row.value == py_average_amount(select(Channel=channel))
        assert row.value == pytest.approx(EXPECTED["by_channel"][channel]["avg_transaction_amount_kzt"], abs=1e-3)
    by_month = calc(demo, "average_transaction_amount", ["Month"])
    for row in by_month.rows:
        assert row.value == pytest.approx(EXPECTED["by_month"][row.group["Month"]]["avg_transaction_amount_kzt"], abs=1e-3)


def test_average_amount_is_not_the_mean_of_row_averages():
    frame = pd.DataFrame({"Transactions": [10, 90], "Amount_KZT": [1000, 900_000]})  # 100 и 10 000 на транзакцию
    result = run(custom(frame), "average_transaction_amount")
    assert (1000 / 10 + 900_000 / 90) / 2 == 5050.0  # так считать нельзя
    assert result.overall.value == 901_000 / 100 == 9010.0


# ================================================================== счётчики и объём
@pytest.mark.parametrize(
    ("metric", "key"),
    [("transaction_count", "Transactions"), ("successful_count", "Successful"), ("failed_count", "Failed"), ("transaction_volume", "Amount_KZT")],
)
def test_sum_metrics_are_exact_integers(demo, metric, key):
    result = calc(demo, metric)
    assert result.overall.value == py_sum(key) and isinstance(result.overall.value, int)
    by_month = calc(demo, metric, ["Month"])
    assert [r.value for r in by_month.rows] == [py_sum(key, select(Month=m)) for m in gen.MONTHS]
    assert sum(r.value for r in by_month.rows) == result.overall.value  # суммы групп складываются в итог


def test_successful_plus_failed_equals_transactions(demo):
    assert calc(demo, "successful_count").overall.value + calc(demo, "failed_count").overall.value == calc(demo, "transaction_count").overall.value


def test_all_six_metrics_are_supported_with_labels_and_units():
    assert list(METRICS) == [
        "transaction_count", "successful_count", "failed_count", "transaction_volume", "success_rate", "average_transaction_amount",
    ]
    assert {m.unit for m in METRICS.values()} == {"шт.", "KZT", "%"}


def test_group_rows_report_rows_used_and_matched(demo):
    result = calc(demo, "transaction_count", ["Channel"])
    assert [r.rows_used for r in result.rows] == [6, 6, 6] and result.rows_matched == 18


# ================================================================== фильтры
def test_filter_eq_ne_in_not_in(demo):
    assert calc(demo, "transaction_count", filters=[{"column": "Channel", "op": "eq", "value": "Mobile"}]).overall.value == py_sum("Transactions", select(Channel="Mobile"))
    assert calc(demo, "transaction_count", filters=[{"column": "Channel", "op": "ne", "value": "Mobile"}]).overall.value == py_sum("Transactions") - py_sum("Transactions", select(Channel="Mobile"))
    two = [r for r in ROWS if r["Channel"] in ("Mobile", "Web")]
    assert calc(demo, "success_rate", filters=[{"column": "Channel", "op": "in", "value": ["Mobile", "Web"]}]).overall.value == py_success_rate(two)
    assert calc(demo, "success_rate", filters=[{"column": "Channel", "op": "not_in", "value": ["Mobile", "Web"]}]).overall.value == py_success_rate(select(Channel="API"))


def test_shorthand_filters(demo):
    assert calc(demo, "success_rate", filters={"Channel": "API"}).overall.value == py_success_rate(select(Channel="API"))
    two = [r for r in ROWS if r["Channel"] in ("Mobile", "API")]
    result = calc(demo, "success_rate", filters={"Channel": ["Mobile", "API"]})
    assert result.overall.value == py_success_rate(two)
    assert [(c.column, c.op.value) for c in result.filters] == [("Channel", "in")]


def test_numeric_filters(demo):
    big = [r for r in ROWS if r["Transactions"] > 100_000]
    assert calc(demo, "transaction_count", filters=[{"column": "Transactions", "op": "gt", "value": 100_000}]).overall.value == py_sum("Transactions", big)
    low = [r for r in ROWS if r["Transactions"] <= 50_000]
    assert calc(demo, "transaction_count", filters=[{"column": "Transactions", "op": "lte", "value": 50_000}]).overall.value == py_sum("Transactions", low)
    inside = [r for r in ROWS if 90_000 <= r["Transactions"] <= 110_000]
    assert calc(demo, "transaction_count", filters=[{"column": "Transactions", "op": "between", "value": [90_000, 110_000]}]).overall.value == py_sum("Transactions", inside)
    assert calc(demo, "transaction_count", filters=[{"column": "Transactions", "op": "gte", "value": 10**9}]).rows_matched == 0
    assert calc(demo, "transaction_count", filters=[{"column": "Transactions", "op": "lt", "value": 50_000}]).rows_matched == len([r for r in ROWS if r["Transactions"] < 50_000])


def test_month_filters_have_month_granularity(demo):
    march = select(Month="2026-03")
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "eq", "value": "2026-03"}]).overall.value == py_success_rate(march)
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "eq", "value": "2026-03-01"}]).rows_matched == 3
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "gte", "value": "2026-05"}]).rows_matched == 6
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "gt", "value": "2026-05"}]).rows_matched == 3
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "lt", "value": "2026-02"}]).rows_matched == 3
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "lte", "value": "2026-02"}]).rows_matched == 6
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "ne", "value": "2026-03"}]).rows_matched == 15
    quarter = [r for r in ROWS if r["Month"] in ("2026-02", "2026-03", "2026-04")]
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "between", "value": ["2026-02", "2026-04"]}]).overall.value == py_success_rate(quarter)
    two = [r for r in ROWS if r["Month"] in ("2026-01", "2026-06")]
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "in", "value": ["2026-01", "2026-06"]}]).overall.value == py_success_rate(two)
    assert calc(demo, "success_rate", filters=[{"column": "Month", "op": "not_in", "value": ["2026-01", "2026-06"]}]).rows_matched == 12


def test_day_precision_dates():
    frame = pd.DataFrame({"Day": pd.date_range("2026-03-01", periods=6), "Transactions": [1, 2, 3, 4, 5, 6]})
    dataset = custom(frame)
    assert run(dataset, "transaction_count", filters=[{"column": "Day", "op": "eq", "value": "2026-03-03"}]).overall.value == 3
    assert run(dataset, "transaction_count", filters=[{"column": "Day", "op": "between", "value": ["2026-03-02", "2026-03-04"]}]).overall.value == 9
    assert run(dataset, "transaction_count", filters=[{"column": "Day", "op": "gt", "value": "2026-03-04"}]).overall.value == 11
    assert run(dataset, "transaction_count", group_by=["Day"]).rows[0].group["Day"] == "2026-03-01"  # не первое число месяца: даты целиком


def test_combined_filters_and_group_by(demo):
    rows = [r for r in ROWS if r["Channel"] == "Web" and r["Month"] >= "2026-03"]
    result = calc(demo, "success_rate", ["Month"], [{"column": "Channel", "value": "Web"}, {"column": "Month", "op": "gte", "value": "2026-03"}])
    assert [r.group["Month"] for r in result.rows] == ["2026-03", "2026-04", "2026-05", "2026-06"]
    assert result.overall.value == py_success_rate(rows)
    for row in result.rows:
        assert row.value == py_success_rate(select(Channel="Web", Month=row.group["Month"]))


def test_empty_selection_gives_zero_counts_and_undefined_ratio(demo):
    counts = calc(demo, "transaction_count", filters={"Channel": "Mobile", "Month": "2026-13-99"[:0] or "2099-01"})
    assert counts.rows_matched == 0 and counts.overall.value == 0
    ratio = calc(demo, "success_rate", filters={"Month": "2099-01"})
    assert ratio.overall.value is None
    assert any("не оставили ни одной строки" in w for w in ratio.warnings)


def test_unknown_category_value_warns_with_available_values(demo):
    result = calc(demo, "transaction_count", filters={"Channel": "Fax"})
    assert result.rows_matched == 0
    assert any("«Fax»" in w and "Mobile" in w for w in result.warnings)


def test_filter_values_are_data_never_code(demo):
    payloads = ["__import__('os').system('echo hacked')", "Mobile' or 1==1 --", "x'; DROP TABLE data; --", "Channel == 'Web'", "${7*7}"]
    for payload in payloads:
        result = calc(demo, "transaction_count", filters={"Channel": payload})
        assert result.rows_matched == 0 and result.overall.value == 0  # просто нет такого значения
        result = calc(demo, "transaction_count", filters=[{"column": "Channel", "op": "in", "value": [payload, "API"]}])
        assert result.overall.value == py_sum("Transactions", select(Channel="API"))


# ================================================================== пропуски и нули
def test_rows_with_missing_values_are_excluded_from_numerator_and_denominator():
    frame = pd.DataFrame(
        {"Channel": ["A", "A", "A", "B"], "Transactions": [100, 100, 100, 50], "Successful": [90, np.nan, 80, 50]}
    )
    result = run(custom(frame), "success_rate", ["Channel"])
    a = next(r for r in result.rows if r.group["Channel"] == "A")
    assert a.value == (90 + 80) / (100 + 100) * 100  # строка с пропуском не попала ни в числитель, ни в знаменатель
    assert a.rows_used == 2 and result.overall.value == (90 + 80 + 50) / (100 + 100 + 50) * 100
    assert any("Исключено строк с пропусками" in w and "1" in w for w in result.warnings)


def test_zero_denominator_is_undefined_not_an_error():
    frame = pd.DataFrame({"Channel": ["A", "B"], "Transactions": [0, 10], "Successful": [0, 9]})
    result = run(custom(frame), "success_rate", ["Channel"])
    assert [r.value for r in result.rows] == [None, 90.0]
    assert any("значение не определено" in w for w in result.warnings)
    assert run(custom(frame), "success_rate", filters={"Channel": "A"}).overall.value is None


def test_float_amounts_are_summed_without_accumulated_error():
    frame = pd.DataFrame({"Transactions": [1] * 10, "Amount_KZT": [0.1] * 10})
    assert run(custom(frame), "transaction_volume").overall.value == 1.0  # обычная сумма дала бы 0.9999999999999999


def test_metric_columns_are_matched_case_insensitively():
    frame = pd.DataFrame({"transactions": [10, 20], "SUCCESSFUL": [9, 18]})
    assert run(custom(frame), "success_rate").overall.value == 27 / 30 * 100


# ================================================================== порядок и форматирование групп
def test_groups_are_sorted_and_missing_group_goes_last():
    frame = pd.DataFrame({"Region": ["b", None, "a", "b"], "Transactions": [1, 2, 3, 4]})
    result = run(custom(frame), "transaction_count", ["Region"])
    assert [r.group["Region"] for r in result.rows] == ["a", "b", None]
    assert [r.value for r in result.rows] == [3, 5, 2]


def test_group_by_accepts_a_single_string(demo):
    assert calc(demo, "transaction_count", "Channel").group_by == ["Channel"]


# ================================================================== строгая проверка group_by
def test_unknown_group_by_column_lists_available_and_suggests(demo):
    with pytest.raises(AnalyticsError) as exc:
        calc(demo, "success_rate", ["Chanel"])
    message = exc.value.user_message
    assert "«Chanel»" in message and "Возможно, вы имели в виду «Channel»" in message
    assert "Доступные колонки" in message and "«Month»" in message and "«Amount_KZT»" in message


@pytest.mark.parametrize(
    ("group_by", "fragment"),
    [
        (["Channel", "Channel"], "указана дважды"),
        (["Month", "Channel", "Transactions", "Failed"], "не более 3"),
        ({"Channel": 1}, "ожидается список"),
        ([1], "должно быть строкой"),
        (["Channel; DROP TABLE x"], "не найдена"),
        (["__import__('os')"], "не найдена"),
        (["channel"], "Возможно, вы имели в виду «Channel»"),  # регистр важен: сначала уточняем у пользователя
    ],
)
def test_invalid_group_by(demo, group_by, fragment):
    with pytest.raises(AnalyticsError, match=fragment.replace("(", r"\(")):
        calc(demo, "success_rate", group_by)


def test_too_many_groups_is_rejected():
    frame = pd.DataFrame({"id": [f"row-{i}" for i in range(300)], "Transactions": range(1, 301)})
    with pytest.raises(AnalyticsError, match="Слишком много групп"):
        run(custom(frame), "transaction_count", ["id"])


def test_sensitive_columns_cannot_be_used_for_grouping_or_filtering():
    frame = pd.DataFrame({"email": ["a@b.kz", "c@d.kz", "e@f.kz"], "Transactions": [1, 2, 3]})
    dataset = custom(frame)
    assert dataset[1].column("email").is_sensitive
    with pytest.raises(AnalyticsError, match="персональные данные"):
        run(dataset, "transaction_count", ["email"])
    with pytest.raises(AnalyticsError, match="персональные данные"):
        run(dataset, "transaction_count", filters={"email": "a@b.kz"})


# ================================================================== строгая проверка filters
@pytest.mark.parametrize(
    ("filters", "fragment"),
    [
        ([{"column": "Nope", "op": "eq", "value": 1}], "колонка «Nope» не найдена"),
        ([{"column": "Channel", "op": "like", "value": "M%"}], "неизвестный оператор 'like'"),
        ([{"column": "Channel"}], "полями column"),
        ([{"column": "Channel", "value": "x", "extra": 1}], "полями column"),
        (["Channel = 'Web'"], "полями column"),
        ("Channel = 'Web'", "ожидается список условий"),
        ([{"column": 5, "value": 1}], "column должен быть строкой"),
        ({"Transactions": "много"}, "должно быть числом"),
        ({"Transactions": True}, "должно быть числом"),
        ({"Channel": 5}, "должно быть строкой"),
        ([{"column": "Channel", "op": "gt", "value": "M"}], "допустимы операторы eq, ne, in, not_in"),
        ([{"column": "Channel", "op": "in", "value": "Mobile"}], "нужен непустой список"),
        ([{"column": "Channel", "op": "in", "value": []}], "нужен непустой список"),
        ([{"column": "Transactions", "op": "between", "value": [1]}], "список из двух значений"),
        ([{"column": "Transactions", "op": "between", "value": [10, 1]}], "начало диапазона больше конца"),
        ([{"column": "Month", "op": "eq", "value": "March"}], "YYYY-MM"),
        ([{"column": "Month", "op": "eq", "value": "2026-13"}], "несуществующая дата"),
        ([{"column": "Month", "op": "eq", "value": "2026-02-30"}], "несуществующая дата"),
        ([{"column": "Month", "op": "eq", "value": 202603}], "YYYY-MM"),
        ([{"column": "Transactions", "op": "eq", "value": float("inf")}], "должно быть числом"),
    ],
)
def test_invalid_filters_are_rejected_with_a_clear_message(demo, filters, fragment):
    with pytest.raises(AnalyticsError, match=fragment):
        calc(demo, "success_rate", filters=filters)


def test_too_many_filters_and_values(demo):
    with pytest.raises(AnalyticsError, match="не более 10"):
        calc(demo, "success_rate", filters=[{"column": "Channel", "value": "Web"}] * 11)
    with pytest.raises(AnalyticsError, match="не более 100"):
        calc(demo, "success_rate", filters=[{"column": "Channel", "op": "in", "value": [str(i) for i in range(101)]}])


def test_boolean_column_filters():
    frame = pd.DataFrame({"Live": ["да", "нет", "да", "нет"], "Transactions": [1, 2, 3, 4]})
    dataset = custom(frame)
    assert run(dataset, "transaction_count", filters=[{"column": "Live", "value": True}]).overall.value == 4
    assert run(dataset, "transaction_count", filters=[{"column": "Live", "op": "ne", "value": True}]).overall.value == 6
    with pytest.raises(AnalyticsError, match="true или false"):
        run(dataset, "transaction_count", filters=[{"column": "Live", "value": "да"}])


# ================================================================== метрика и колонки
@pytest.mark.parametrize("metric", ["revenue", "SUCCESS_RATE", "", None, 5, "success_rate; DROP TABLE x"])
def test_unknown_metric(demo, metric):
    with pytest.raises(AnalyticsError, match="Неизвестная метрика.*transaction_count, successful_count"):
        calc(demo, metric)


def test_metric_needs_its_source_columns():
    frame = pd.DataFrame({"Transactions": [1, 2], "Successful": [1, 2]})
    with pytest.raises(AnalyticsError) as exc:
        run(custom(frame), "transaction_volume")
    assert "Amount_KZT" in exc.value.user_message and "Доступные колонки" in exc.value.user_message
    with pytest.raises(AnalyticsError, match="нет колонок «Failed»"):
        run(custom(frame), "failed_count")


def test_source_columns_must_be_numeric():
    frame = pd.DataFrame({"Transactions": ["много", "мало", "средне"], "Successful": [1, 2, 3]})
    with pytest.raises(AnalyticsError, match="должна быть числовой"):
        run(custom(frame), "success_rate")


# ================================================================== безопасность реализации
BARE_FORBIDDEN = {"eval", "exec", "compile", "__import__"}  # встроенные функции (re.compile и graph.compile — не они)
ATTRIBUTE_FORBIDDEN = {"query", "eval", "system", "popen", "read_sql", "read_sql_query", "to_sql"}
FORBIDDEN_IMPORTS = {"subprocess", "os", "pickle", "sqlite3", "importlib"}
LLM_IMPORTS = {"openai", "langchain", "langchain_openai", "langchain_core", "anthropic", "langsmith"}
SOURCE_ROOT = Path(__file__).resolve().parent.parent / "datastory"


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize(
    "path",
    ["analytics/metrics.py", "analytics/charts.py", "analytics/summary.py", "mcp_server/server.py", "workflow/graph.py"],
)
def test_no_dynamic_code_execution_in_tool_implementation(path):
    tree = ast.parse((SOURCE_ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in BARE_FORBIDDEN, f"{path}: вызов {node.func.id}()"
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ATTRIBUTE_FORBIDDEN, f"{path}: .{node.func.attr}()"
    assert not _imported_modules(tree) & FORBIDDEN_IMPORTS, f"{path}: запрещённый импорт"


@pytest.mark.parametrize("path", ["analytics/metrics.py", "analytics/charts.py", "analytics/summary.py", "mcp_server/server.py"])
def test_tool_implementation_does_not_use_llm_libraries(path):
    tree = ast.parse((SOURCE_ROOT / path).read_text(encoding="utf-8"))
    assert not _imported_modules(tree) & LLM_IMPORTS, f"{path}: арифметику не поручаем LLM"


def test_static_check_actually_detects_forbidden_code():
    """Проверка проверки: на заведомо плохом коде она должна срабатывать."""
    bad = ast.parse("import os\nx = eval('1+1')\ndf.query('a > 1')\n")
    calls = [n for n in ast.walk(bad) if isinstance(n, ast.Call)]
    assert any(isinstance(c.func, ast.Name) and c.func.id in BARE_FORBIDDEN for c in calls)
    assert any(isinstance(c.func, ast.Attribute) and c.func.attr in ATTRIBUTE_FORBIDDEN for c in calls)
    assert _imported_modules(bad) & FORBIDDEN_IMPORTS
