"""Сводка датасета для инструмента profile_dataset (строится из сохранённого профиля, без LLM)."""
from __future__ import annotations

from datastory.analytics.metrics import available_metrics
from datastory.analytics.models import ColumnSummary, DatasetSummary, MissingInfo
from datastory.models import ColumnKind, ColumnProfile, DatasetProfile

DIMENSION_KINDS = (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN, ColumnKind.DATETIME)


def _statistics(column: ColumnProfile) -> dict | None:
    """Статистики колонки. У персональных данных скрыты: сводка уходит LLM-агенту."""
    if column.is_sensitive:
        return None
    if column.kind is ColumnKind.NUMERIC and column.numeric_stats:
        return {k: v for k, v in column.numeric_stats.model_dump().items() if v is not None}
    if column.kind is ColumnKind.DATETIME and column.date_min:
        return {"min": column.date_min, "max": column.date_max}
    if column.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN) and column.top_values:
        return {"top_values": {t.value: t.count for t in column.top_values}}
    return None


def build_dataset_summary(profile: DatasetProfile) -> DatasetSummary:
    columns = [
        ColumnSummary(
            name=c.name, kind=c.kind.value, dtype=c.dtype, missing_count=c.missing_count, missing_pct=c.missing_pct,
            unique_count=c.unique_count, is_sensitive=c.is_sensitive, statistics=_statistics(c),
        )
        for c in profile.columns
    ]
    numeric = [c.name for c in profile.columns if c.kind is ColumnKind.NUMERIC and not c.is_sensitive]
    return DatasetSummary(
        dataset_id=profile.dataset_id,
        filename=profile.filename,
        sheet_name=profile.sheet_name,
        row_count=profile.row_count,
        column_count=profile.column_count,
        column_names=[c.name for c in profile.columns],
        column_types={c.name: c.kind.value for c in profile.columns},
        columns=columns,
        numeric_measures=numeric,
        dimensions=[c.name for c in profile.columns if c.kind in DIMENSION_KINDS and not c.is_sensitive],
        sensitive_columns=[c.name for c in profile.columns if c.is_sensitive],
        missing_cell_count=profile.missing_cell_count,
        duplicate_row_count=profile.duplicate_row_count,
        missing_by_column={c.name: MissingInfo(count=c.missing_count, pct=c.missing_pct) for c in profile.columns if c.missing_count},
        available_metrics=available_metrics(numeric),
    )
