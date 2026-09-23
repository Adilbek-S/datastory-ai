"""Evaluation Pipeline. Пока — проверки структурной корректности результата анализа.

Оценка качества LLM-выводов (LLM-as-judge, LangSmith datasets) — следующий этап.
"""
from __future__ import annotations

from datastory.analytics.models import AnalysisResult
from datastory.models import EvalCase


def evaluate_result(result: AnalysisResult) -> list[EvalCase]:
    summary = result.summary
    return [
        EvalCase(
            name="summary_covers_all_columns",
            passed=len(summary.columns) == summary.column_count,
            details=f"{len(summary.columns)} из {summary.column_count}",
        ),
        EvalCase(name="has_kpis", passed=bool(result.kpis)),
        EvalCase(name="has_insights", passed=bool(result.insights)),
        EvalCase(
            name="charts_carry_their_data",
            passed=all(chart.data for chart in result.charts),
            details=f"графиков: {len(result.charts)}",
        ),
    ]
