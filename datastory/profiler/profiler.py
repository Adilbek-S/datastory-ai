"""Dataset Profiler: определяет структуру таблицы и тип каждого показателя."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from datastory.models import ColumnKind, ColumnProfile, DatasetProfile

CATEGORICAL_MAX_UNIQUE_RATIO = 0.5
CATEGORICAL_MAX_UNIQUE = 50


def detect_kind(series: pd.Series) -> ColumnKind:
    if pdt.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if pdt.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME
    if pdt.is_numeric_dtype(series):
        return ColumnKind.NUMERIC

    non_null = series.dropna()
    if non_null.empty:
        return ColumnKind.TEXT
    unique = non_null.nunique()
    if unique <= CATEGORICAL_MAX_UNIQUE and unique / len(non_null) <= CATEGORICAL_MAX_UNIQUE_RATIO:
        return ColumnKind.CATEGORICAL
    return ColumnKind.TEXT


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    columns = [
        ColumnProfile(
            name=str(name),
            dtype=str(series.dtype),
            kind=detect_kind(series),
            missing=int(series.isna().sum()),
            unique=int(series.nunique(dropna=True)),
            sample=[str(v) for v in series.dropna().head(3).tolist()],
        )
        for name, series in df.items()
    ]
    return DatasetProfile(
        rows=len(df),
        columns=df.shape[1],
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        column_profiles=columns,
    )
