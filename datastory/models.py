"""Структурированные модели данных, общие для всех модулей."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DataQualityIssue(BaseModel):
    """Найденная проблема качества данных."""

    code: str
    severity: Severity = Severity.WARNING
    column: str | None = None
    message: str
    count: int | None = None


class NumericStats(BaseModel):
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None


class TopValue(BaseModel):
    value: str
    count: int


class ColumnProfile(BaseModel):
    name: str
    position: int
    dtype: str
    original_dtype: str
    kind: ColumnKind
    detected_kind: ColumnKind
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    sample_values: list[str] = Field(default_factory=list)
    numeric_stats: NumericStats | None = None
    date_min: str | None = None
    date_max: str | None = None
    top_values: list[TopValue] = Field(default_factory=list)
    is_sensitive: bool = False
    sensitive_reason: str = ""

    @property
    def is_overridden(self) -> bool:
        return self.kind != self.detected_kind


class DatasetProfile(BaseModel):
    dataset_id: str
    filename: str
    sheet_name: str | None = None
    row_count: int
    column_count: int
    columns: list[ColumnProfile] = Field(default_factory=list)
    quality_issues: list[DataQualityIssue] = Field(default_factory=list)
    detected_date_columns: list[str] = Field(default_factory=list)
    detected_numeric_columns: list[str] = Field(default_factory=list)
    detected_category_columns: list[str] = Field(default_factory=list)
    description: str = ""
    missing_cell_count: int = 0
    duplicate_row_count: int = 0

    def column(self, name: str) -> ColumnProfile:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def names_of(self, kind: ColumnKind) -> list[str]:
        return [c.name for c in self.columns if c.kind == kind]


class DatasetReference(BaseModel):
    """Ссылка на подтверждённый датасет в рабочем хранилище: по dataset_id его находят другие компоненты."""

    dataset_id: str
    filename: str
    sheet_name: str | None = None
    row_count: int
    column_count: int
    storage_path: str
    created_at: datetime
    confirmed: bool = True


class KPI(BaseModel):
    label: str
    value: str
    hint: str = ""


class ChartSpec(BaseModel):
    title: str
    kind: str  # histogram | bar | line | scatter
    x: str
    y: str | None = None
    color: str | None = None
    description: str = ""


class Insight(BaseModel):
    title: str
    text: str
    severity: str = "info"  # info | warning | positive


class AnalysisResult(BaseModel):
    profile: DatasetProfile
    kpis: list[KPI] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)


class KnowledgeChunk(BaseModel):
    source: str
    page: int
    text: str


class EvalCase(BaseModel):
    name: str
    passed: bool
    details: str = ""
