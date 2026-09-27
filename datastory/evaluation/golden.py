"""Golden cases: модель, загрузка и хеш. Данные и эталоны полностью синтетические (см. golden_builder.py)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from datastory.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "evaluation"
GOLDEN_PATH = EVAL_DIR / "golden" / "cases.json"
RESULTS_PATH = EVAL_DIR / "results" / "latest.json"
GROUPS = ("metric_interpretation", "time_series", "category_comparison", "filters", "rag_terminology", "invalid_ambiguous")
Behavior = Literal["answer", "reject", "clarify"]


class ExpectedFilter(BaseModel):
    column: str
    op: Literal["eq", "in"]
    value: Any  # строка для eq, список для in


class ExpectedRow(BaseModel):
    group: dict[str, str] = Field(default_factory=dict)
    value: float


class ExpectedNumeric(BaseModel):
    """Эталон, рассчитанный обычным Python-кодом по исходным строкам синтетического набора (без pandas и MCP)."""

    metric: str
    group_by: list[str]
    filters: list[ExpectedFilter] = Field(default_factory=list)
    rows: list[ExpectedRow]
    overall: float


class GoldenCase(BaseModel):
    id: str
    group: Literal[GROUPS]  # type: ignore[valid-type]
    user_query: str
    expected_behavior: Behavior
    expected_metric: str | None = None
    expected_columns: list[str] = Field(default_factory=list)
    expected_group_by: list[str] = Field(default_factory=list)
    expected_filters: list[ExpectedFilter] = Field(default_factory=list)
    expected_chart_type: str | None = None
    expected_rag_document: str | None = None
    expected_rag_section: str | None = None
    expected_numeric: ExpectedNumeric | None = None
    notes: str = ""


class GoldenSet(BaseModel):
    description: str
    dataset: str
    documents: list[str]
    cases: list[GoldenCase]


def load_golden(path: Path = GOLDEN_PATH) -> GoldenSet:
    return GoldenSet.model_validate_json(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_golden(golden: GoldenSet) -> str:
    """Каноничный JSON: одинаковый вывод при одинаковых данных (используется для проверки актуальности файла)."""
    return json.dumps(golden.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=1) + "\n"
