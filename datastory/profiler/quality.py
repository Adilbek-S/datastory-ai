"""Data Quality: поиск проблем в данных (правила, без LLM)."""
from __future__ import annotations

import pandas as pd

from datastory.models import ColumnKind, ColumnProfile, DataQualityIssue, Severity

MISSING_WARNING_PCT = 5.0
MISSING_ERROR_PCT = 50.0
INVALID_ERROR_PCT = 20.0
MAX_EXAMPLES = 3


def _examples(raw: pd.Series, invalid: pd.Series) -> str:
    values = [str(v)[:30] for v in raw[invalid].dropna().unique()[:MAX_EXAMPLES]]
    return f" Примеры: {', '.join(values)}." if values else ""


def find_quality_issues(
    typed: pd.DataFrame,
    raw: pd.DataFrame,
    columns: list[ColumnProfile],
    invalid_masks: dict[str, pd.Series],
    duplicate_rows: int,
) -> list[DataQualityIssue]:
    rows = len(typed)
    issues: list[DataQualityIssue] = []

    if duplicate_rows:
        issues.append(
            DataQualityIssue(
                code="duplicate_rows",
                severity=Severity.WARNING,
                message=f"Найдено полностью повторяющихся строк: {duplicate_rows} ({duplicate_rows / rows:.1%}).",
                count=duplicate_rows,
            )
        )

    for col in columns:
        name = col.name
        if name.startswith("Unnamed"):
            issues.append(
                DataQualityIssue(
                    code="unnamed_column",
                    column=name,
                    message=f"У колонки №{col.position + 1} нет названия. Добавьте заголовок в исходном файле.",
                )
            )

        if col.missing_count == rows:
            issues.append(
                DataQualityIssue(
                    code="empty_column", column=name, count=rows,
                    message=f"Колонка «{name}» полностью пуста.",
                )
            )
        elif col.missing_count:
            severity = (
                Severity.ERROR if col.missing_pct >= MISSING_ERROR_PCT
                else Severity.WARNING if col.missing_pct >= MISSING_WARNING_PCT
                else Severity.INFO
            )
            issues.append(
                DataQualityIssue(
                    code="missing_values", severity=severity, column=name, count=col.missing_count,
                    message=f"В колонке «{name}» пропущено значений: {col.missing_count} ({col.missing_pct:.1f}%).",
                )
            )

        if col.unique_count == 1 and col.missing_count < rows and rows > 1:
            issues.append(
                DataQualityIssue(
                    code="constant_column", severity=Severity.INFO, column=name,
                    message=f"Колонка «{name}» содержит одно и то же значение во всех строках.",
                )
            )

        invalid = invalid_masks.get(name)
        if invalid is not None and invalid.any() and col.kind in (ColumnKind.NUMERIC, ColumnKind.DATETIME, ColumnKind.BOOLEAN):
            count = int(invalid.sum())
            pct = count / rows * 100
            noun = {ColumnKind.NUMERIC: "числа", ColumnKind.DATETIME: "даты", ColumnKind.BOOLEAN: "логические значения"}[col.kind]
            examples = "" if col.is_sensitive else _examples(raw[name], invalid)
            issues.append(
                DataQualityIssue(
                    code=f"invalid_{col.kind.value}",
                    severity=Severity.ERROR if pct >= INVALID_ERROR_PCT else Severity.WARNING,
                    column=name,
                    count=count,
                    message=(
                        f"В колонке «{name}» значений, которые не распознаны как {noun}: {count} ({pct:.1f}%). "
                        f"Они будут считаться пропусками.{examples}"
                    ),
                )
            )

        if col.is_sensitive:
            issues.append(
                DataQualityIssue(
                    code="possible_personal_data", severity=Severity.INFO, column=name,
                    message=(
                        f"Колонка «{name}» может содержать персональные данные ({col.sensitive_reason}). "
                        "Её значения не передаются в LLM."
                    ),
                )
            )
    return issues
