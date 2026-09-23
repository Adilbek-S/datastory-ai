"""Evaluation Pipeline. Пока — проверки структурной корректности результата анализа.

Оценка качества LLM-выводов (LLM-as-judge, LangSmith datasets) — следующий этап.
"""
from __future__ import annotations

from datastory.models import AnalysisResult, EvalCase


def evaluate_result(result: AnalysisResult) -> list[EvalCase]:
    profile = result.profile
    return [
        EvalCase(
            name="profile_covers_all_columns",
            passed=len(profile.columns) == profile.column_count,
            details=f"{len(profile.columns)} из {profile.column_count}",
        ),
        EvalCase(name="has_kpis", passed=bool(result.kpis)),
        EvalCase(name="has_insights", passed=bool(result.insights)),
    ]
