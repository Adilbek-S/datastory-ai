"""Расчёт метрик обычным Python/Pandas-кодом.

Здесь нет LLM, eval, exec, DataFrame.query и SQL. group_by и filters — это данные: имена колонок
сверяются со списком существующих колонок, операторы — с фиксированным набором, значения — по типу
колонки. Всё, что не прошло проверку, отклоняется понятной ошибкой AnalyticsError.

Формулы (суммы считаются по строкам, а не усредняются по группам):
    success_rate               = SUM(Successful) / SUM(Transactions) × 100
    average_transaction_amount = SUM(Amount_KZT) / SUM(Transactions)
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from pandas.api import types as pdt

from datastory.analytics.models import FilterCondition, FilterOp, MetricResult, MetricRow
from datastory.errors import AnalyticsError
from datastory.models import ColumnKind, DatasetProfile

MAX_GROUP_BY = 3
MAX_GROUPS = 200
MAX_FILTERS = 10
MAX_LIST_VALUES = 100
MAX_EXAMPLES = 8

TRANSACTIONS, SUCCESSFUL, FAILED, AMOUNT = "Transactions", "Successful", "Failed", "Amount_KZT"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    label: str
    unit: str
    formula: str
    columns: tuple[str, ...]  # канонические названия исходных колонок
    compute: Callable[[dict[str, float | int]], float | int | None]


def _ratio(numerator: str, denominator: str, scale: float = 1.0):
    def compute(sums: dict[str, float | int]) -> float | None:
        den = sums[denominator]
        return None if den == 0 else sums[numerator] / den * scale  # SUM / SUM (× 100)

    return compute


METRICS: dict[str, MetricDefinition] = {
    d.name: d
    for d in (
        MetricDefinition("transaction_count", "Количество транзакций", "шт.", "SUM(Transactions)", (TRANSACTIONS,), lambda s: s[TRANSACTIONS]),
        MetricDefinition("successful_count", "Успешные транзакции", "шт.", "SUM(Successful)", (SUCCESSFUL,), lambda s: s[SUCCESSFUL]),
        MetricDefinition("failed_count", "Неуспешные транзакции", "шт.", "SUM(Failed)", (FAILED,), lambda s: s[FAILED]),
        MetricDefinition("transaction_volume", "Объём транзакций", "KZT", "SUM(Amount_KZT)", (AMOUNT,), lambda s: s[AMOUNT]),
        MetricDefinition(
            "success_rate", "Success Rate", "%", "SUM(Successful) / SUM(Transactions) × 100",
            (SUCCESSFUL, TRANSACTIONS), _ratio(SUCCESSFUL, TRANSACTIONS, 100.0),
        ),
        MetricDefinition(
            "average_transaction_amount", "Средняя сумма транзакции", "KZT", "SUM(Amount_KZT) / SUM(Transactions)",
            (AMOUNT, TRANSACTIONS), _ratio(AMOUNT, TRANSACTIONS),
        ),
    )
}


# --------------------------------------------------------------------------- вспомогательное
def _quoted(names: list[str], limit: int = 20) -> str:
    shown = ", ".join(f"«{n}»" for n in names[:limit])
    return shown + (f" и ещё {len(names) - limit}" if len(names) > limit else "")


SUM_PREFIX = "sum:"
_CURRENCY_SUFFIX = re.compile(r"[_ ](KZT|USD|EUR|RUB|UZS)$", re.IGNORECASE)


def sum_metric(column: str) -> MetricDefinition:
    """Универсальный показатель «сумма числовой колонки»: для наборов данных без колонок платёжной системы (например, продаж)."""
    currency = _CURRENCY_SUFFIX.search(column)
    label = _CURRENCY_SUFFIX.sub("", column).replace("_", " ").strip() or column
    return MetricDefinition(
        f"{SUM_PREFIX}{column}", label, currency.group(1).upper() if currency else "", f"SUM({column})", (column,), lambda s: s[column]
    )


def get_metric(name: Any) -> MetricDefinition:
    if isinstance(name, str) and name.startswith(SUM_PREFIX) and len(name) > len(SUM_PREFIX):
        return sum_metric(name[len(SUM_PREFIX):])
    if not isinstance(name, str) or name not in METRICS:
        raise AnalyticsError(f"Неизвестная метрика {name!r}. Поддерживаются: {', '.join(METRICS)}.")
    return METRICS[name]


def resolve_source_columns(metric: MetricDefinition, df: pd.DataFrame) -> dict[str, str]:
    """Сопоставляет канонические колонки метрики с колонками датасета (регистр не важен)."""
    lookup = {str(c).casefold(): str(c) for c in df.columns}
    resolved, missing = {}, []
    for canonical in metric.columns:
        actual = lookup.get(canonical.casefold())
        if actual is None:
            missing.append(canonical)
        else:
            resolved[canonical] = actual
    if missing:
        raise AnalyticsError(
            f"Метрику {metric.name} ({metric.formula}) нельзя посчитать: в датасете нет колонок {_quoted(missing)}. "
            f"Доступные колонки: {_quoted([str(c) for c in df.columns])}."
        )
    for canonical, actual in resolved.items():
        if not pdt.is_numeric_dtype(df[actual]) or pdt.is_bool_dtype(df[actual]):
            raise AnalyticsError(f"Колонка «{actual}» должна быть числовой для метрики {metric.name}, а её тип — {df[actual].dtype}.")
    return resolved


def _require_column(name: Any, available: list[str], role: str) -> str:
    if not isinstance(name, str):
        raise AnalyticsError(f"{role}: название колонки должно быть строкой, получено {type(name).__name__}.")
    if name in available:
        return name
    hint = ""
    close = [c for c in available if c.casefold() == name.casefold()] or difflib.get_close_matches(name, available, n=1, cutoff=0.6)
    if close:
        hint = f" Возможно, вы имели в виду «{close[0]}»."
    raise AnalyticsError(f"{role}: колонка «{name}» не найдена в датасете.{hint} Доступные колонки: {_quoted(available)}.")


def _reject_sensitive(name: str, profile: DatasetProfile, role: str) -> None:
    if profile.column(name).is_sensitive:
        raise AnalyticsError(
            f"{role}: колонка «{name}» помечена как содержащая персональные данные — группировка и фильтрация по ней запрещены."
        )


def _python_number(value: Any) -> float | int:
    if hasattr(value, "item"):
        value = value.item()
    return value


def _exact_sum(series: pd.Series) -> float | int:
    """Сумма без накопления ошибки: целые — точно, дробные — через math.fsum."""
    if pdt.is_integer_dtype(series):
        return int(series.sum())
    return math.fsum(float(v) for v in series.tolist())


# --------------------------------------------------------------------------- group_by
def validate_group_by(group_by: Any, df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    if group_by is None:
        return []
    if isinstance(group_by, str):
        group_by = [group_by]
    if not isinstance(group_by, (list, tuple)):
        raise AnalyticsError("group_by: ожидается список названий колонок, например [\"Month\", \"Channel\"].")
    if len(group_by) > MAX_GROUP_BY:
        raise AnalyticsError(f"group_by: допускается не более {MAX_GROUP_BY} колонок, получено {len(group_by)}.")
    available = [str(c) for c in df.columns]
    names: list[str] = []
    for item in group_by:
        name = _require_column(item, available, "group_by")
        if name in names:
            raise AnalyticsError(f"group_by: колонка «{name}» указана дважды.")
        _reject_sensitive(name, profile, "group_by")
        names.append(name)
    return names


# --------------------------------------------------------------------------- filters
_BOOL_OPS = {FilterOp.EQ, FilterOp.NE, FilterOp.IN, FilterOp.NOT_IN}
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?)?$")


def normalize_filters(filters: Any) -> list[FilterCondition]:
    """Список условий или краткая запись {колонка: значение | [значения]} -> список FilterCondition."""
    if filters is None or filters == [] or filters == {}:
        return []
    if isinstance(filters, dict):
        items = [{"column": column, "op": "in" if isinstance(value, list) else "eq", "value": value} for column, value in filters.items()]
    elif isinstance(filters, (list, tuple)):
        items = list(filters)
    else:
        raise AnalyticsError("filters: ожидается список условий {column, op, value} или словарь {колонка: значение}.")
    if len(items) > MAX_FILTERS:
        raise AnalyticsError(f"filters: допускается не более {MAX_FILTERS} условий, получено {len(items)}.")

    conditions = []
    for item in items:
        if isinstance(item, FilterCondition):
            conditions.append(item)
            continue
        if not isinstance(item, dict) or set(item) - {"column", "op", "value"} or "column" not in item or "value" not in item:
            raise AnalyticsError("filters: каждое условие — объект с полями column, op (необязательно) и value.")
        if not isinstance(item["column"], str):
            raise AnalyticsError(f"filters: column должен быть строкой, получено {item['column']!r}.")
        try:
            conditions.append(FilterCondition(**item))
        except ValueError:
            allowed = ", ".join(op.value for op in FilterOp)
            raise AnalyticsError(f"filters: неизвестный оператор {item.get('op')!r}. Допустимы: {allowed}.") from None
    return conditions


def _as_number(value: Any, column: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
        raise AnalyticsError(f"filters: для числовой колонки «{column}» значение должно быть числом, получено {value!r}.")
    return value


def _as_text(value: Any, column: str) -> str:
    if not isinstance(value, str):
        raise AnalyticsError(f"filters: для колонки «{column}» значение должно быть строкой, получено {value!r}.")
    return value


def _as_bool(value: Any, column: str) -> bool:
    if not isinstance(value, bool):
        raise AnalyticsError(f"filters: для логической колонки «{column}» значение должно быть true или false, получено {value!r}.")
    return value


def _date_range(value: Any, column: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Дата -> полуинтервал [начало, конец): «2026-03» — месяц целиком, «2026-03-15» — сутки."""
    match = _DATE_RE.match(value.strip()) if isinstance(value, str) else None
    if not match:
        raise AnalyticsError(
            f"filters: для колонки «{column}» (дата) значение должно быть строкой YYYY-MM, YYYY-MM-DD или YYYY-MM-DD HH:MM:SS, "
            f"получено {value!r}."
        )
    year, month, day, hour, minute, second = match.groups()
    try:
        if day is None:
            start = pd.Timestamp(int(year), int(month), 1)
            return start, start + pd.DateOffset(months=1)
        if hour is None:
            start = pd.Timestamp(int(year), int(month), int(day))
            return start, start + pd.Timedelta(days=1)
        start = pd.Timestamp(int(year), int(month), int(day), int(hour), int(minute), int(second or 0))
        return start, start + pd.Timedelta(seconds=1)
    except ValueError:
        raise AnalyticsError(f"filters: для колонки «{column}» указана несуществующая дата {value!r}.") from None


def _as_list(value: Any, column: str, op: FilterOp) -> list:
    if not isinstance(value, (list, tuple)) or not value:
        raise AnalyticsError(f"filters: для оператора {op.value} по колонке «{column}» нужен непустой список значений.")
    if len(value) > MAX_LIST_VALUES:
        raise AnalyticsError(f"filters: в списке для колонки «{column}» не более {MAX_LIST_VALUES} значений.")
    return list(value)


def _as_pair(value: Any, column: str) -> list:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AnalyticsError(f"filters: для between по колонке «{column}» нужен список из двух значений [от, до].")
    return list(value)


def _condition_mask(series: pd.Series, kind: ColumnKind, cond: FilterCondition, warnings: list[str]) -> pd.Series:
    column, op, value = cond.column, cond.op, cond.value
    present = series.notna()

    if kind is ColumnKind.NUMERIC:
        if op is FilterOp.BETWEEN:
            low, high = (_as_number(v, column) for v in _as_pair(value, column))
            if low > high:
                raise AnalyticsError(f"filters: для between по колонке «{column}» начало диапазона больше конца.")
            return (series >= low) & (series <= high)
        if op in (FilterOp.IN, FilterOp.NOT_IN):
            values = [_as_number(v, column) for v in _as_list(value, column, op)]
            inside = series.isin(values)
            return inside if op is FilterOp.IN else present & ~inside
        number = _as_number(value, column)
        return {
            FilterOp.EQ: lambda: series == number,
            FilterOp.NE: lambda: present & (series != number),
            FilterOp.GT: lambda: series > number,
            FilterOp.GTE: lambda: series >= number,
            FilterOp.LT: lambda: series < number,
            FilterOp.LTE: lambda: series <= number,
        }[op]()

    if kind is ColumnKind.DATETIME:
        def within(v: Any) -> pd.Series:
            start, end = _date_range(v, column)
            return (series >= start) & (series < end)

        if op is FilterOp.BETWEEN:
            low, high = _as_pair(value, column)
            return (series >= _date_range(low, column)[0]) & (series < _date_range(high, column)[1])
        if op in (FilterOp.IN, FilterOp.NOT_IN):
            hit = pd.Series(False, index=series.index)
            for v in _as_list(value, column, op):
                hit |= within(v)
            return hit if op is FilterOp.IN else present & ~hit
        start, end = _date_range(value, column)
        return {
            FilterOp.EQ: lambda: within(value),
            FilterOp.NE: lambda: present & ~within(value),
            FilterOp.GT: lambda: series >= end,
            FilterOp.GTE: lambda: series >= start,
            FilterOp.LT: lambda: series < start,
            FilterOp.LTE: lambda: series < end,
        }[op]()

    # категории, текст и логические значения: только сравнение на равенство и вхождение
    if op not in _BOOL_OPS:
        raise AnalyticsError(
            f"filters: для колонки «{column}» (тип {kind.value}) допустимы операторы eq, ne, in, not_in; получен {op.value}."
        )
    if kind is ColumnKind.BOOLEAN:
        values = [_as_bool(v, column) for v in (_as_list(value, column, op) if op in (FilterOp.IN, FilterOp.NOT_IN) else [value])]
        inside = series.astype("boolean").isin(values).fillna(False)
    else:
        values = [_as_text(v, column) for v in (_as_list(value, column, op) if op in (FilterOp.IN, FilterOp.NOT_IN) else [value])]
        known = set(series.dropna().astype(str).unique())
        unknown = [v for v in values if v not in known]
        if unknown and op in (FilterOp.EQ, FilterOp.IN):
            sample = ", ".join(sorted(known)[:MAX_EXAMPLES])
            warnings.append(f"Значение {_quoted(unknown)} не найдено в колонке «{column}» (есть: {sample}).")
        inside = series.astype("string").isin(values).fillna(False).astype(bool)
    return inside if op in (FilterOp.EQ, FilterOp.IN) else present & ~inside


def validate_and_mask(df: pd.DataFrame, profile: DatasetProfile, filters: Any) -> tuple[list[FilterCondition], pd.Series, list[str]]:
    conditions = normalize_filters(filters)
    available = [str(c) for c in df.columns]
    warnings: list[str] = []
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        _require_column(cond.column, available, "filters")
        _reject_sensitive(cond.column, profile, "filters")
        mask &= _condition_mask(df[cond.column], profile.column(cond.column).kind, cond, warnings).fillna(False).astype(bool)
    return conditions, mask, warnings


# --------------------------------------------------------------------------- группы
def _month_start_columns(df: pd.DataFrame, names: list[str]) -> set[str]:
    result = set()
    for name in names:
        series = df[name]
        if pdt.is_datetime64_any_dtype(series):
            values = series.dropna()
            if not values.empty and (values.dt.day == 1).all() and (values == values.dt.normalize()).all():
                result.add(name)
    return result


def _format_key(value: Any, column: str, month_columns: set[str]):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        if column in month_columns:
            return value.strftime("%Y-%m")
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.strftime("%Y-%m-%d %H:%M:%S")
    return _python_number(value) if not isinstance(value, str) else value


def _sort_key(key: tuple) -> tuple:
    return tuple((1, 0) if v is None or (not isinstance(v, str) and pd.isna(v)) else (0, v) for v in key)


def _sums(part: pd.DataFrame, columns: dict[str, str]) -> dict[str, float | int]:
    return {canonical: _exact_sum(part[actual]) for canonical, actual in columns.items()}


# --------------------------------------------------------------------------- главная функция
def calculate_metric(
    df: pd.DataFrame,
    profile: DatasetProfile,
    dataset_id: str,
    metric: str,
    group_by: Any = None,
    filters: Any = None,
) -> MetricResult:
    definition = get_metric(metric)
    columns = resolve_source_columns(definition, df)
    group_cols = validate_group_by(group_by, df, profile)
    conditions, mask, warnings = validate_and_mask(df, profile, filters)

    subset = df.loc[mask]
    matched = len(subset)
    if matched == 0:
        warnings.append("Фильтры не оставили ни одной строки: значение не определено.")
    # Числитель и знаменатель берутся по одним и тем же строкам: строки с пропуском в любой нужной колонке исключаются.
    usable = subset.dropna(subset=list(columns.values()))
    if len(usable) < matched:
        warnings.append(
            f"Исключено строк с пропусками в колонках {_quoted(list(columns.values()))}: {matched - len(usable)}."
        )

    def make_row(part: pd.DataFrame, group: dict) -> MetricRow:
        sums = _sums(part, columns) if len(part) else {c: 0 for c in columns}
        value = definition.compute(sums)  # пустая выборка: счётчики = 0, отношения не определены (None)
        return MetricRow(group=group, value=value, components={f"sum_{k.lower()}": v for k, v in sums.items()}, rows_used=len(part))

    overall = make_row(usable, {})
    if not group_cols:
        rows = [overall]
    else:
        grouped = usable.groupby(group_cols, dropna=False, sort=False)
        if grouped.ngroups > MAX_GROUPS:
            raise AnalyticsError(
                f"Слишком много групп ({grouped.ngroups}, максимум {MAX_GROUPS}). Выберите колонку с меньшим числом значений или добавьте фильтры."
            )
        month_columns = _month_start_columns(df, group_cols)
        collected = []
        for key, part in grouped:
            key = key if isinstance(key, tuple) else (key,)
            collected.append((key, part))
        collected.sort(key=lambda item: _sort_key(tuple(None if (not isinstance(v, str) and pd.isna(v)) else v for v in item[0])))
        rows = [
            make_row(part, {col: _format_key(v, col, month_columns) for col, v in zip(group_cols, key)})
            for key, part in collected
        ]
        empty_groups = [r for r in rows if r.value is None]
        if empty_groups:
            warnings.append(f"Для {len(empty_groups)} групп значение не определено (знаменатель равен 0).")

    if matched and overall.value is None and not group_cols:
        warnings.append("Знаменатель формулы равен 0: значение не определено.")

    return MetricResult(
        dataset_id=dataset_id,
        metric=definition.name,
        label=definition.label,
        unit=definition.unit,
        formula=definition.formula,
        group_by=group_cols,
        filters=conditions,
        rows=rows,
        overall=overall,
        rows_matched=matched,
        warnings=warnings,
    )


def available_metrics(numeric_columns: list[str]) -> list[str]:
    """Метрики, для которых в датасете есть все нужные числовые колонки.

    Если нет ни одной метрики платёжной системы, доступны суммы числовых колонок («sum:<колонка>»).
    """
    numeric = {c.casefold() for c in numeric_columns}
    found = [name for name, d in METRICS.items() if all(canonical.casefold() in numeric for canonical in d.columns)]
    return found or [f"{SUM_PREFIX}{column}" for column in numeric_columns]
