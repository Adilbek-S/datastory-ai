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
            passed=len(profile.column_profiles) == profile.columns,
            details=f"{len(profile.column_profiles)} из {profile.columns}",
        ),
        EvalCase(name="has_kpis", passed=bool(result.kpis)),
        EvalCase(name="has_insights", passed=bool(result.insights)),
    ]
