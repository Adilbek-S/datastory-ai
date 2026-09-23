"""Dataset Profiler: строит структурированный профиль таблицы и приводит колонки к типам.

Все расчёты (строки, пропуски, дубликаты, статистика) выполняет Pandas — LLM здесь не участвует.
"""
from __future__ import annotations

import uuid

import pandas as pd

from datastory.errors import NoDataError
from datastory.models import (
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    NumericStats,
    TopValue,
)
from datastory.profiler.column_types import coerce, detect_kind
from datastory.profiler.pii import detect_sensitive
from datastory.profiler.quality import find_quality_issues

MAX_SAMPLES = 5
MAX_TOP_VALUES = 5
MAX_VALUE_LEN = 60
DESCRIPTION_LIST_LIMIT = 8


def new_dataset_id() -> str:
    return uuid.uuid4().hex[:12]


def _fmt(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M") if (value.hour or value.minute or value.second) else value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)[:MAX_VALUE_LEN]


def _numeric_stats(series: pd.Series) -> NumericStats | None:
    values = series.dropna().astype("float64")
    if values.empty:
        return None
    std = values.std()
    return NumericStats(
        min=float(values.min()),
        max=float(values.max()),
        mean=float(values.mean()),
        median=float(values.median()),
        std=None if pd.isna(std) else float(std),
    )


def _profile_column(
    position: int, name: str, raw: pd.Series, typed: pd.Series, invalid: pd.Series,
    kind: ColumnKind, detected: ColumnKind, sensitive_reason: str, rows: int,
) -> ColumnProfile:
    missing = int(typed.isna().sum() - invalid.sum())  # нераспознанные значения — не «пропуски»
    non_null = typed.dropna()
    profile = ColumnProfile(
        name=name,
        position=position,
        dtype=str(typed.dtype),
        original_dtype=str(raw.dtype),
        kind=kind,
        detected_kind=detected,
        missing_count=missing,
        missing_pct=round(missing / rows * 100, 2) if rows else 0.0,
        unique_count=int(non_null.nunique()),
        sample_values=[_fmt(v) for v in non_null.drop_duplicates().head(MAX_SAMPLES)],
        is_sensitive=bool(sensitive_reason),
        sensitive_reason=sensitive_reason,
    )
    if kind is ColumnKind.NUMERIC:
        profile.numeric_stats = _numeric_stats(typed)
    elif kind is ColumnKind.DATETIME and not non_null.empty:
        profile.date_min, profile.date_max = _fmt(non_null.min()), _fmt(non_null.max())
    elif kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
        counts = non_null.value_counts().head(MAX_TOP_VALUES)
        profile.top_values = [TopValue(value=_fmt(v), count=int(c)) for v, c in counts.items()]
    return profile


def default_description(profile: DatasetProfile) -> str:
    def listing(names: list[str]) -> str:
        shown = ", ".join(names[:DESCRIPTION_LIST_LIMIT])
        return shown + (f" и ещё {len(names) - DESCRIPTION_LIST_LIMIT}" if len(names) > DESCRIPTION_LIST_LIMIT else "")

    parts = [f"Таблица: {profile.row_count} строк, {profile.column_count} колонок."]
    if profile.detected_numeric_columns:
        parts.append(f"Числовые показатели: {listing(profile.detected_numeric_columns)}.")
    if profile.detected_date_columns:
        parts.append(f"Временные колонки: {listing(profile.detected_date_columns)}.")
    if profile.detected_category_columns:
        parts.append(f"Категории: {listing(profile.detected_category_columns)}.")
    return " ".join(parts)


def build_profile(
    df: pd.DataFrame,
    filename: str,
    sheet_name: str | None = None,
    *,
    type_overrides: dict[str, ColumnKind] | None = None,
    sensitive_overrides: dict[str, bool] | None = None,
    description: str | None = None,
    dataset_id: str | None = None,
) -> tuple[DatasetProfile, pd.DataFrame]:
    """Возвращает (профиль, DataFrame с типами, приведёнными согласно профилю).

    type_overrides — ручные исправления типов колонок; sensitive_overrides — ручная
    отметка «персональные данные» (True/False поверх автоопределения).
    """
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise NoDataError("В таблице нет данных для анализа.")
    type_overrides = type_overrides or {}
    sensitive_overrides = sensitive_overrides or {}
    rows = len(df)

    typed_columns: dict[str, pd.Series] = {}
    columns: list[ColumnProfile] = []
    invalid_masks: dict[str, pd.Series] = {}
    for position, (name, raw) in enumerate(df.items()):
        name = str(name)
        detected = detect_kind(raw)
        kind = type_overrides.get(name, detected)
        result = coerce(raw, kind)

        reason = detect_sensitive(name, raw)
        if name in sensitive_overrides:
            reason = "отмечено пользователем" if sensitive_overrides[name] else ""

        typed_columns[name] = result.series
        invalid_masks[name] = result.invalid
        columns.append(_profile_column(position, name, raw, result.series, result.invalid, kind, detected, reason, rows))

    typed = pd.DataFrame(typed_columns, index=df.index)
    duplicates = int(typed.duplicated().sum())
    raw_named = df.rename(columns=str)

    profile = DatasetProfile(
        dataset_id=dataset_id or new_dataset_id(),
        filename=filename,
        sheet_name=sheet_name,
        row_count=rows,
        column_count=len(columns),
        columns=columns,
        detected_date_columns=[c.name for c in columns if c.kind is ColumnKind.DATETIME],
        detected_numeric_columns=[c.name for c in columns if c.kind is ColumnKind.NUMERIC],
        detected_category_columns=[c.name for c in columns if c.kind is ColumnKind.CATEGORICAL],
        missing_cell_count=sum(c.missing_count for c in columns),
        duplicate_row_count=duplicates,
    )
    profile.quality_issues = find_quality_issues(typed, raw_named, columns, invalid_masks, duplicates)
    profile.description = description if description is not None else default_description(profile)
    return profile, typed
