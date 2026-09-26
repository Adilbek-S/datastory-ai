"""Модели аналитических инструментов: сводка датасета, фильтры, результат расчёта метрики."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

Scalar = str | int | float | bool | None


# --------------------------------------------------------------------------- фильтры и метрики
class FilterOp(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"


class FilterCondition(BaseModel):
    """Условие отбора строк. Значение только данные: оно никогда не исполняется как код."""

    column: str = Field(description="Название существующей колонки датасета")
    op: FilterOp = Field(default=FilterOp.EQ, description="eq, ne, gt, gte, lt, lte, in, not_in, between")
    value: Any = Field(description="Значение; для in/not_in — список, для between — [от, до]")


class MetricRow(BaseModel):
    group: dict[str, Scalar] = Field(default_factory=dict)
    value: float | int | None = None  # None — значение не определено (например, SUM(Transactions) = 0)
    components: dict[str, float | int] = Field(default_factory=dict)  # суммы, из которых посчитано значение
    rows_used: int = 0


class MetricResult(BaseModel):
    dataset_id: str
    metric: str
    label: str
    unit: str
    formula: str
    group_by: list[str] = Field(default_factory=list)
    filters: list[FilterCondition] = Field(default_factory=list)
    rows: list[MetricRow] = Field(default_factory=list)  # по группам (без group_by — одна строка с пустой group)
    overall: MetricRow = Field(default_factory=MetricRow)  # по всем отобранным строкам (сумма к сумме, не среднее групп)
    rows_matched: int = 0
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- сводка датасета
class MissingInfo(BaseModel):
    count: int
    pct: float


class ColumnSummary(BaseModel):
    name: str
    kind: str
    dtype: str
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    is_sensitive: bool = False
    statistics: dict[str, Any] | None = None  # у персональных колонок скрыта


class DatasetSummary(BaseModel):
    """Результат инструмента profile_dataset."""

    dataset_id: str
    filename: str
    sheet_name: str | None = None
    row_count: int
    column_count: int
    column_names: list[str]
    column_types: dict[str, str]
    columns: list[ColumnSummary]
    numeric_measures: list[str] = Field(description="Числовые колонки — доступные показатели")
    dimensions: list[str] = Field(description="Категории, даты и логические колонки — доступные измерения")
    sensitive_columns: list[str] = Field(default_factory=list, description="Персональные данные: недоступны для group_by и filters")
    missing_cell_count: int = 0
    duplicate_row_count: int = 0
    missing_by_column: dict[str, MissingInfo] = Field(default_factory=dict)
    available_metrics: list[str] = Field(default_factory=list, description="Метрики, для которых в датасете есть нужные колонки")
