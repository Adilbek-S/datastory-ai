"""Показатели для интерфейса: карточки KPI из сводки датасета и из результатов метрик.

Сами метрики считает MCP-сервер (datastory.analytics.metrics); здесь только оформление результатов.
"""
from __future__ import annotations

from datastory.analytics.formatting import format_metric_value, format_number
from datastory.analytics.models import DatasetSummary, MetricResult
from datastory.models import KPI

KPI_METRICS = ("transaction_count", "success_rate", "transaction_volume", "average_transaction_amount")
KPI_HINTS = {
    "transaction_count": "SUM(Transactions)",
    "success_rate": "SUM(Successful) / SUM(Transactions) × 100",
    "transaction_volume": "SUM(Amount_KZT)",
    "average_transaction_amount": "SUM(Amount_KZT) / SUM(Transactions)",
}


def basic_kpis(summary: DatasetSummary) -> list[KPI]:
    total_cells = summary.row_count * summary.column_count
    missing_pct = (summary.missing_cell_count / total_cells * 100) if total_cells else 0.0
    return [
        KPI(label="Строк", value=format_number(summary.row_count)),
        KPI(label="Столбцов", value=str(summary.column_count)),
        KPI(label="Числовых показателей", value=str(len(summary.numeric_measures)), hint="Кандидаты для расчётов"),
        KPI(label="Пропусков", value=f"{format_number(missing_pct, 1)}%", hint=f"{summary.missing_cell_count} ячеек"),
    ]


def metric_kpis(overall: dict[str, MetricResult]) -> list[KPI]:
    """Карточки по итоговым значениям метрик (значение по всем строкам, а не среднее по группам)."""
    return [
        KPI(label=r.label, value=format_metric_value(r.overall.value, r.unit), hint=KPI_HINTS.get(name, r.formula))
        for name in KPI_METRICS
        if (r := overall.get(name)) is not None
    ]
