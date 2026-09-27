"""Построение golden cases: 30 синтетических запросов по демо-набору sales_2026 и эталонные числа.

Эталоны считает обычный Python (циклы и словари) по тем же строкам, из которых сгенерирован sales_2026.xlsx
(scripts/generate_sales_demo.py). Ни pandas, ни MCP, ни LLM здесь не участвуют, поэтому сравнение с MCP независимо.

    python -m datastory.evaluation.golden_builder      # перезаписать evaluation/golden/cases.json
"""
from __future__ import annotations

import sys

from datastory.evaluation.golden import (
    GOLDEN_PATH,
    ExpectedFilter,
    ExpectedNumeric,
    ExpectedRow,
    GoldenCase,
    GoldenSet,
    dump_golden,
)
from scripts import generate_sales_demo as sales

METRICS_DOC, CHANNELS_DOC = "sales_metrics.pdf", "sales_channels_regions.pdf"
REVENUE, ORDERS, CANCELLED = "sum:Revenue_KZT", "sum:Orders", "sum:Cancelled_Orders"
AOV, RATE = "ratio:Revenue_KZT/Orders", "rate:Cancelled_Orders/Orders"
SEC_REVENUE, SEC_ORDERS, SEC_AOV, SEC_RATE, SEC_OUT_OF_SCOPE = "2. Выручка", "3. Заказы и отмены", "4. Средний чек", "5. Доля отмен", "6. Показатели без данных"
SEC_CHANNELS = "1. Каналы продаж"
TIME, REGION, CHANNEL, CATEGORY = "Month", "Region", "Channel", "Category"


# --------------------------------------------------------------------------- эталонный расчёт (чистый Python)
def _columns(metric: str) -> list[str]:
    kind, _, body = metric.partition(":")
    return [body] if kind == "sum" else body.split("/")


def _value(rows: list[dict], metric: str) -> float:
    kind, _, body = metric.partition(":")
    if kind == "sum":
        return float(sum(r[body] for r in rows))
    numerator, denominator = body.split("/")
    scale = 100.0 if kind == "rate" else 1.0
    return sum(r[numerator] for r in rows) / sum(r[denominator] for r in rows) * scale  # сумма к сумме, не среднее строк


def _passes(row: dict, condition: ExpectedFilter) -> bool:
    allowed = condition.value if condition.op == "in" else [condition.value]
    return row[condition.column] in allowed


def reference(rows: list[dict], metric: str, group_by: list[str], filters: list[ExpectedFilter]) -> ExpectedNumeric:
    selected = [r for r in rows if all(_passes(r, f) for f in filters)]
    groups: dict[tuple, list[dict]] = {}
    for row in selected:
        groups.setdefault(tuple(row[c] for c in group_by), []).append(row)
    return ExpectedNumeric(
        metric=metric, group_by=group_by, filters=filters,
        rows=[ExpectedRow(group=dict(zip(group_by, key)), value=_value(part, metric)) for key, part in sorted(groups.items())],
        overall=_value(selected, metric),
    )


# --------------------------------------------------------------------------- случаи
def _flt(column: str, value) -> ExpectedFilter:
    return ExpectedFilter(column=column, op="in" if isinstance(value, list) else "eq", value=value)


def _answer(rows, case_id, group, query, metric, group_by, chart, doc=None, section=None, filters=(), notes="") -> GoldenCase:
    filters = list(filters)
    columns = list(dict.fromkeys([*_columns(metric), *group_by, *(f.column for f in filters)]))
    return GoldenCase(
        id=case_id, group=group, user_query=query, expected_behavior="answer", expected_metric=metric, expected_columns=columns,
        expected_group_by=group_by, expected_filters=filters, expected_chart_type=chart, expected_rag_document=doc,
        expected_rag_section=section, expected_numeric=reference(rows, metric, group_by, filters), notes=notes,
    )


def _refusal(case_id, query, behavior, doc=None, section=None, notes="") -> GoldenCase:
    return GoldenCase(
        id=case_id, group="invalid_ambiguous", user_query=query, expected_behavior=behavior, expected_rag_document=doc,
        expected_rag_section=section, notes=notes,
    )


def build_cases(rows: list[dict]) -> list[GoldenCase]:
    a = lambda *args, **kw: _answer(rows, *args, **kw)  # noqa: E731
    mi, ts, cc, fl, rg = "metric_interpretation", "time_series", "category_comparison", "filters", "rag_terminology"
    return [
        # 1. Интерпретация показателя: разговорное название → метрика по определению из документа
        a("mi-01", mi, "Какой средний чек по каналам?", AOV, [CHANNEL], "bar", METRICS_DOC, SEC_AOV),
        a("mi-02", mi, "Покажи средний чек по категориям товаров", AOV, [CATEGORY], "bar", METRICS_DOC, SEC_AOV),
        a("mi-03", mi, "Какая доля отмен по регионам?", RATE, [REGION], "bar", METRICS_DOC, SEC_RATE),
        a("mi-04", mi, "Сколько в среднем стоит один заказ по регионам?", AOV, [REGION], "bar", METRICS_DOC, SEC_AOV),
        a("mi-05", mi, "Какой процент заказов отменяется по каналам?", RATE, [CHANNEL], "bar", METRICS_DOC, SEC_RATE),
        # 2. Динамика во времени
        a("ts-01", ts, "Покажи динамику выручки по месяцам", REVENUE, [TIME], "line", METRICS_DOC, SEC_REVENUE),
        a("ts-02", ts, "Как менялось количество заказов от месяца к месяцу?", ORDERS, [TIME], "line", METRICS_DOC, SEC_ORDERS),
        a("ts-03", ts, "Покажи динамику отменённых заказов по месяцам", CANCELLED, [TIME], "line", METRICS_DOC, SEC_ORDERS),
        a("ts-04", ts, "Покажи динамику среднего чека по месяцам", AOV, [TIME], "line", METRICS_DOC, SEC_AOV),
        a("ts-05", ts, "Как изменялась доля отмен по месяцам?", RATE, [TIME], "line", METRICS_DOC, SEC_RATE),
        # 3. Сравнение категорий
        a("cc-01", cc, "Сравни регионы по выручке", REVENUE, [REGION], "bar", METRICS_DOC, SEC_REVENUE),
        a("cc-02", cc, "Какая категория показала наибольший рост выручки?", REVENUE, [TIME, CATEGORY], "line", METRICS_DOC, SEC_REVENUE,
          notes="Рост считается по ряду «месяц × категория»: нужна ось времени и серии."),
        a("cc-03", cc, "Какой канал приносит больше всего выручки?", REVENUE, [CHANNEL], "bar", METRICS_DOC, SEC_REVENUE),
        a("cc-04", cc, "Сравни категории по количеству заказов", ORDERS, [CATEGORY], "bar", METRICS_DOC, SEC_ORDERS),
        a("cc-05", cc, "Покажи вклад регионов в выручку", REVENUE, [REGION], "pie", METRICS_DOC, SEC_REVENUE,
          notes="Доли аддитивного показателя по четырём регионам: допустима круговая диаграмма."),
        # 4. Фильтры
        a("fl-01", fl, "Сравни выручку Web и Mobile", REVENUE, [CHANNEL], "bar", METRICS_DOC, SEC_REVENUE, [_flt(CHANNEL, ["Web", "Mobile"])]),
        a("fl-02", fl, "Покажи динамику выручки только по Almaty", REVENUE, [TIME], "line", METRICS_DOC, SEC_REVENUE, [_flt(REGION, "Almaty")]),
        a("fl-03", fl, "Какой средний чек в категории Electronics по каналам?", AOV, [CHANNEL], "bar", METRICS_DOC, SEC_AOV, [_flt(CATEGORY, "Electronics")]),
        a("fl-04", fl, "Сравни выручку регионов за март", REVENUE, [REGION], "bar", METRICS_DOC, SEC_REVENUE, [_flt(TIME, "2026-03")]),
        a("fl-05", fl, "Покажи заказы Offline по регионам", ORDERS, [REGION], "bar", METRICS_DOC, SEC_ORDERS, [_flt(CHANNEL, "Offline")]),
        # 5. Терминология, определённая в документах базы знаний
        a("rg-01", rg, "Покажи cancellation rate", RATE, [TIME], "line", METRICS_DOC, SEC_RATE,
          notes="Группировка не названа: показывается динамика по времени."),
        a("rg-02", rg, "Покажи AOV по регионам", AOV, [REGION], "bar", METRICS_DOC, SEC_AOV),
        a("rg-03", rg, "Как менялся Cancellation Rate по месяцам?", RATE, [TIME], "line", METRICS_DOC, SEC_RATE),
        a("rg-04", rg, "Покажи Revenue по категориям", REVENUE, [CATEGORY], "bar", METRICS_DOC, SEC_REVENUE),
        a("rg-05", rg, "Покажи заказы по каналам продаж", ORDERS, [CHANNEL], "bar", CHANNELS_DOC, SEC_CHANNELS),
        # 6. Недопустимые и неоднозначные запросы: ничего не выдумывать
        _refusal("iv-01", "Покажи EBITDA", "reject", notes="Показателя нет ни в данных, ни в базе знаний."),
        _refusal("iv-02", "Покажи Return Rate по категориям", "reject", METRICS_DOC, SEC_OUT_OF_SCOPE,
                 "Показатель описан в документе, но колонок для него в данных нет; доля отмен — другой показатель."),
        _refusal("iv-03", "Покажи прибыль по регионам", "reject", notes="Колонки прибыли нет, определения тоже."),
        _refusal("iv-04", "Покажи выручку по группам", "clarify", notes="«Группы» — это регион, канал или категория: нужен вопрос пользователю."),
        _refusal("iv-05", "Сделай прогноз выручки на июль", "reject", notes="Прогнозы не поддерживаются; июль вне периода данных."),
    ]


def build_golden() -> GoldenSet:
    rows = sales.build_rows()
    return GoldenSet(
        description=(
            "30 полностью синтетических golden cases по демо-набору sales_2026 (вымышленная сеть SalesDemo KZ). "
            "Эталонные числа рассчитаны обычным Python-кодом по строкам генератора; файл строится командой "
            "python -m datastory.evaluation.golden_builder."
        ),
        dataset="data/demo/sales_2026.xlsx",
        documents=[f"data/demo/sales_docs/{name}" for name in sales.DOCS],
        cases=build_cases(rows),
    )


def main() -> int:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(dump_golden(build_golden()), encoding="utf-8")
    print(GOLDEN_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
