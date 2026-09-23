"""Краткий профиль датасета для LLM.

Строится ТОЛЬКО из DatasetProfile (никогда из DataFrame), поэтому в модель не попадает
таблица целиком. Персональные данные исключаются: у отмеченных колонок скрыты примеры,
топ-значения и статистика; значения, похожие на e-mail/телефон/карту, маскируются.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from datastory.models import ColumnKind, ColumnProfile, DatasetProfile
from datastory.profiler.pii import looks_like_pii_value

REDACTED = "[скрыто]"
MAX_SAMPLES = 3
MAX_VALUE_LEN = 40


class LLMColumnSummary(BaseModel):
    name: str
    kind: str
    dtype: str
    missing_pct: float
    unique_count: int | None = None
    sample_values: list[str] = Field(default_factory=list)
    stats: dict[str, float | str] = Field(default_factory=dict)
    top_values: dict[str, int] = Field(default_factory=dict)
    note: str = ""


class LLMDatasetProfile(BaseModel):
    description: str
    row_count: int
    column_count: int
    columns: list[LLMColumnSummary]
    quality_summary: list[str] = Field(default_factory=list)
    hidden_columns: list[str] = Field(default_factory=list)

    def to_prompt(self) -> str:
        """Компактное текстовое представление для промпта."""
        lines = [
            f"Датасет: {self.row_count} строк, {self.column_count} колонок.",
            f"Описание: {self.description}",
            "Колонки:",
        ]
        for col in self.columns:
            parts = [f"- {col.name} ({col.kind}, {col.dtype}), пропусков {col.missing_pct:g}%"]
            if col.unique_count is not None:
                parts.append(f"уникальных {col.unique_count}")
            if col.stats:
                parts.append("статистика: " + ", ".join(f"{k}={v}" for k, v in col.stats.items()))
            if col.top_values:
                parts.append("частые: " + ", ".join(f"{k} ({v})" for k, v in col.top_values.items()))
            if col.sample_values:
                parts.append("примеры: " + ", ".join(col.sample_values))
            if col.note:
                parts.append(col.note)
            lines.append("; ".join(parts))
        if self.quality_summary:
            lines.append("Качество данных: " + "; ".join(self.quality_summary))
        return "\n".join(lines)


def _safe(value: str) -> str:
    text = value[:MAX_VALUE_LEN]
    return REDACTED if looks_like_pii_value(text) else text


def _summarize(col: ColumnProfile, max_samples: int) -> LLMColumnSummary:
    summary = LLMColumnSummary(
        name=col.name, kind=col.kind.value, dtype=col.dtype, missing_pct=col.missing_pct,
    )
    if col.is_sensitive:
        summary.note = "персональные данные — значения скрыты"
        return summary

    summary.unique_count = col.unique_count
    if col.kind is ColumnKind.TEXT:
        # Свободный текст (имена, адреса, комментарии) — примеры не передаём.
        summary.note = "свободный текст — примеры скрыты"
        return summary

    summary.sample_values = [_safe(v) for v in col.sample_values[:max_samples]]
    if col.numeric_stats:
        stats = col.numeric_stats
        summary.stats = {
            k: round(v, 4) for k, v in
            (("min", stats.min), ("max", stats.max), ("mean", stats.mean), ("median", stats.median), ("std", stats.std))
            if v is not None
        }
    if col.date_min and col.date_max:
        summary.stats = {"min": col.date_min, "max": col.date_max}
    summary.top_values = {_safe(t.value): t.count for t in col.top_values[:max_samples]}
    return summary


def build_llm_profile(profile: DatasetProfile, *, max_samples: int = MAX_SAMPLES) -> LLMDatasetProfile:
    summaries = [_summarize(col, max_samples) for col in profile.columns]
    quality = [
        f"{issue.code}" + (f" [{issue.column}]" if issue.column else "") + (f": {issue.count}" if issue.count else "")
        for issue in profile.quality_issues
        if issue.code != "possible_personal_data"
    ]
    return LLMDatasetProfile(
        description=profile.description,
        row_count=profile.row_count,
        column_count=profile.column_count,
        columns=summaries,
        quality_summary=quality,
        hidden_columns=[c.name for c in profile.columns if c.is_sensitive],
    )
