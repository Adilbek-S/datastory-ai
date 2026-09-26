"""Evaluation Pipeline. Пока — проверки структурной корректности результата анализа.

Оценка качества LLM-выводов (LLM-as-judge, LangSmith datasets) — следующий этап.
"""
from __future__ import annotations

from datastory.models import EvalCase
from datastory.workflow.models import MAX_CHARTS, AnalysisResult


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
        EvalCase(name="at_most_four_charts", passed=len(result.charts) <= MAX_CHARTS, details=f"графиков: {len(result.charts)}"),
        EvalCase(
            name="every_insight_has_numeric_evidence",
            passed=all(i.evidence and i.data_source for i in result.insights),
            details=f"выводов: {len(result.insights)}",
        ),
        EvalCase(
            name="insights_passed_verification",
            passed=all(c.passed for c in result.insight_checks),
            details=f"проверено: {len(result.insight_checks)}",
        ),
    ]
