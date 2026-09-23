"""Определение и приведение типов колонок (без LLM).

Тип определяется по фактическим значениям: числа, записанные текстом («1 234,5»),
и даты в популярных форматах («2026-03», «31.12.2026») распознаются автоматически.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt

from datastory.models import ColumnKind

# Значения, которые считаются «пусто», а не ошибкой.
NULL_TOKENS = {"", "-", "—", "–", "n/a", "na", "nan", "null", "none", "н/д"}
NUMERIC_MIN_RATIO = 0.8
DATETIME_MIN_RATIO = 0.8
CATEGORICAL_MAX_UNIQUE = 50
CATEGORICAL_MAX_UNIQUE_RATIO = 0.5
FORMAT_SAMPLE_SIZE = 500

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "ISO8601",
]

TRUE_TOKENS = {"true", "да", "yes", "y", "истина", "1"}
FALSE_TOKENS = {"false", "нет", "no", "n", "ложь", "0"}
_BOOL_WORDS = (TRUE_TOKENS | FALSE_TOKENS) - {"1", "0"}


@dataclass
class Coerced:
    """Результат приведения колонки к типу."""

    series: pd.Series
    invalid: pd.Series  # True там, где значение есть, но не распознано как нужный тип


def is_text_dtype(series: pd.Series) -> bool:
    return pdt.is_string_dtype(series) or pdt.is_object_dtype(series)


def clean_strings(series: pd.Series) -> pd.Series:
    """Строки без пробелов по краям; пустые и «нулевые» токены превращаются в NA."""
    text = series.astype("string").str.strip()
    return text.mask(text.str.lower().isin(NULL_TOKENS))


# --------------------------------------------------------------------------- числа
def parse_numeric(series: pd.Series) -> Coerced:
    if pdt.is_bool_dtype(series):
        return Coerced(series.astype("float64"), pd.Series(False, index=series.index))
    if pdt.is_numeric_dtype(series):
        return Coerced(series, pd.Series(False, index=series.index))

    text = clean_strings(series)
    text = text.str.replace(r"[\s  ₸$€£₽%]", "", regex=True)
    both = (text.str.contains(",", regex=False) & text.str.contains(".", regex=False)).fillna(False)
    text = text.mask(both, text.str.replace(",", "", regex=False))  # «1,234.56» -> «1234.56»
    text = text.str.replace(",", ".", regex=False)  # «12,5» -> «12.5»

    parsed = pd.to_numeric(text, errors="coerce").astype("float64")
    invalid = parsed.isna() & text.notna()
    non_null = parsed.dropna()
    if not parsed.isna().any() and not non_null.empty and (non_null % 1 == 0).all():
        parsed = parsed.astype("int64")
    return Coerced(parsed, invalid)


# --------------------------------------------------------------------------- даты
def infer_datetime_format(strings: pd.Series) -> str | None:
    """Формат даты, подходящий ≥80% значений, либо None."""
    sample = strings.dropna().head(FORMAT_SAMPLE_SIZE)
    if sample.empty:
        return None
    best_format, best_ratio = None, 0.0
    for fmt in DATE_FORMATS:
        try:
            ratio = pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean()
        except (ValueError, TypeError):
            continue
        if ratio > best_ratio:
            best_format, best_ratio = fmt, ratio
    return best_format if best_ratio >= DATETIME_MIN_RATIO else None


def parse_datetime(series: pd.Series, *, allow_mixed: bool = False) -> Coerced:
    if pdt.is_datetime64_any_dtype(series):
        return Coerced(series, pd.Series(False, index=series.index))
    text = clean_strings(series)
    fmt = infer_datetime_format(text)
    if fmt is not None:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
    elif allow_mixed:  # пользователь явно выбрал «дата», а единого формата нет
        parsed = pd.to_datetime(text, format="mixed", dayfirst=True, errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return Coerced(parsed, parsed.isna() & text.notna())


# --------------------------------------------------------------------------- логические
def parse_boolean(series: pd.Series) -> Coerced:
    if pdt.is_bool_dtype(series):
        return Coerced(series.astype("boolean"), pd.Series(False, index=series.index))
    if pdt.is_numeric_dtype(series):
        text = series.astype("string")
        text = text.str.replace(r"\.0$", "", regex=True)
    else:
        text = clean_strings(series)
    lowered = text.str.lower()
    mapped = pd.Series(pd.NA, index=series.index, dtype="boolean")
    mapped = mapped.mask(lowered.isin(TRUE_TOKENS).fillna(False), True)
    mapped = mapped.mask(lowered.isin(FALSE_TOKENS).fillna(False), False)
    return Coerced(mapped, mapped.isna() & text.notna())


# --------------------------------------------------------------------------- определение типа
def detect_kind(series: pd.Series) -> ColumnKind:
    if pdt.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if pdt.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME
    if pdt.is_numeric_dtype(series):
        return ColumnKind.NUMERIC

    text = clean_strings(series)
    non_null = text.dropna()
    if non_null.empty:
        return ColumnKind.TEXT

    if set(non_null.str.lower().unique()) <= _BOOL_WORDS:
        return ColumnKind.BOOLEAN

    numeric = parse_numeric(series)
    if numeric.series.notna().sum() / len(non_null) >= NUMERIC_MIN_RATIO:
        return ColumnKind.NUMERIC

    if infer_datetime_format(text) is not None:
        return ColumnKind.DATETIME

    unique = non_null.nunique()
    if unique <= CATEGORICAL_MAX_UNIQUE and unique / len(non_null) <= CATEGORICAL_MAX_UNIQUE_RATIO:
        return ColumnKind.CATEGORICAL
    return ColumnKind.TEXT


def coerce(series: pd.Series, kind: ColumnKind) -> Coerced:
    """Приводит колонку к выбранному типу. Нераспознанные значения становятся пропусками."""
    if kind is ColumnKind.NUMERIC:
        return parse_numeric(series)
    if kind is ColumnKind.DATETIME:
        return parse_datetime(series, allow_mixed=True)
    if kind is ColumnKind.BOOLEAN:
        return parse_boolean(series)

    no_invalid = pd.Series(False, index=series.index)
    if is_text_dtype(series):
        return Coerced(series, no_invalid)
    text = series.astype("string")
    if pdt.is_float_dtype(series):  # 7.0 -> «7», чтобы коды не получали лишний «.0»
        text = text.str.replace(r"\.0$", "", regex=True)
    return Coerced(text, no_invalid)
