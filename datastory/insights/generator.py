"""Insight Generator. Пока — правила по результатам расчётов; LLM-выводы (GPT-4o-mini) — следующий этап.

Все выводы здесь — результаты вычислений (EvidenceType.COMPUTED): они опираются на числа,
посчитанные MCP-сервером, и не содержат предположений о причинах.
"""
from __future__ import annotations

from datastory.analytics.formatting import format_number
from datastory.analytics.models import MetricResult
from datastory.models import EvidenceType, Insight

DROP_THRESHOLD_PP = 2.0  # снижение Success Rate относительно остальных периодов, п.п.


def generate_insights(summary) -> list[Insight]:
    """Качество данных. summary — DatasetSummary или DatasetProfile (нужны duplicate_row_count и missing_cell_count)."""
    insights: list[Insight] = []
    if summary.duplicate_row_count:
        insights.append(
            Insight(
                title="Найдены дубликаты",
                text=f"В данных {summary.duplicate_row_count} повторяющихся строк.",
                severity="warning",
            )
        )
    if summary.missing_cell_count:
        insights.append(
            Insight(
                title="Есть пропуски",
                text=f"Всего пропущенных ячеек: {summary.missing_cell_count}.",
                severity="warning",
            )
        )
    if not insights:
        insights.append(Insight(title="Данные чистые", text="Пропусков и дубликатов не найдено.", severity="positive"))
    return insights


def generate_metric_insights(success_rate_by_period: MetricResult | None) -> list[Insight]:
    """Вывод по динамике Success Rate: период с заметным снижением относительно остальных.

    Уровень остальных периодов считается как сумма к сумме, а не как среднее процентов.
    Причина снижения по данным не определяется, и вывод об этом не делается.
    """
    result = success_rate_by_period
    if result is None or result.metric != "success_rate" or len(result.group_by) != 1:
        return []
    rows = [r for r in result.rows if r.value is not None]
    if len(rows) < 3:
        return []

    period_column = result.group_by[0]
    lowest = min(rows, key=lambda r: r.value)
    others = [r for r in rows if r is not lowest]
    successful = sum(r.components["sum_successful"] for r in others)
    transactions = sum(r.components["sum_transactions"] for r in others)
    if transactions == 0:
        return []
    baseline = successful / transactions * 100
    delta = lowest.value - baseline
    period = lowest.group[period_column]

    if delta <= -DROP_THRESHOLD_PP:
        return [
            Insight(
                title=f"Снижение Success Rate: {period}",
                text=(
                    f"Success Rate в периоде {period} — {format_number(lowest.value, 2)}%, это на {format_number(abs(delta), 2)} п.п. "
                    f"ниже, чем в остальных периодах ({format_number(baseline, 2)}%). "
                    f"Расчёт: {result.formula} (суммы по периоду). Данные фиксируют снижение, но не объясняют его причину."
                ),
                severity="warning",
                evidence_type=EvidenceType.COMPUTED,
            )
        ]
    highest = max(rows, key=lambda r: r.value)
    return [
        Insight(
            title="Success Rate стабилен",
            text=(
                f"По периодам Success Rate держится в диапазоне {format_number(lowest.value, 2)}–{format_number(highest.value, 2)}%; "
                f"заметных снижений (более {DROP_THRESHOLD_PP:g} п.п.) нет. Расчёт: {result.formula}."
            ),
            severity="positive",
            evidence_type=EvidenceType.COMPUTED,
        )
    ]
