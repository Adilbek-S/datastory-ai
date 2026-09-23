"""Структурированные модели данных, общие для всех модулей."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: ColumnKind
    missing: int = 0
    unique: int = 0
    sample: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    missing_cells: int = 0
    duplicate_rows: int = 0
    column_profiles: list[ColumnProfile] = Field(default_factory=list)

    def columns_of(self, kind: ColumnKind) -> list[str]:
        return [c.name for c in self.column_profiles if c.kind == kind]


class KPI(BaseModel):
    label: str
    value: str
    hint: str = ""


class ChartSpec(BaseModel):
    title: str
    kind: str  # histogram | bar | line | scatter
    x: str
    y: str | None = None
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
