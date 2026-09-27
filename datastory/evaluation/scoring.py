"""Метрики evaluation: чистые функции без обращений к LLM, MCP и базе знаний (проверяются отдельными тестами).

- Retrieval Hit@3: нужный документ базы знаний среди трёх найденных фрагментов;
- Analysis Plan Accuracy: совпали показатель, колонки и группировка с golden case;
- Numeric Accuracy: числа MCP совпали с эталонным Python-расчётом с допуском на погрешность float;
- Invalid request rejection rate: недопустимый запрос отклонён, показатель не выдуман.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from datastory.analytics.metrics import get_metric
from datastory.evaluation.golden import ExpectedFilter, ExpectedNumeric, GoldenCase

REL_TOL = 1e-9  # относительная погрешность float
ABS_TOL = 1e-6


# --------------------------------------------------------------------------- Retrieval Hit@k
def retrieval_hit(hits: list[dict], expected_document: str, k: int = 3) -> bool:
    """hits — найденные фрагменты по убыванию близости: [{document, section, score}, …]."""
    return any(h["document"] == expected_document for h in hits[:k])


def section_hit(hits: list[dict], expected_document: str, expected_section: str, k: int = 3) -> bool:
    """Строже документа: среди Top-k есть фрагмент нужного раздела нужного документа."""
    return any(h["document"] == expected_document and expected_section in (h.get("section") or "") for h in hits[:k])


# --------------------------------------------------------------------------- Analysis Plan Accuracy
def plan_columns(metric: str, group_by: list[str], filters: list[dict[str, Any]]) -> list[str]:
    """Колонки набора, которые использует шаг: источники показателя, группировка и колонки фильтров."""
    return list(dict.fromkeys([*get_metric(metric).columns, *group_by, *(f["column"] for f in filters)]))


def _filter_key(column: str, op: str, value: Any) -> tuple:
    values = value if isinstance(value, list) else [value]
    return (column, "in" if op in ("eq", "in") else op, tuple(sorted(str(v) for v in values)))  # eq и in с одним значением равны


def filters_match(expected: list[ExpectedFilter], predicted: list[dict[str, Any]]) -> bool:
    return sorted(_filter_key(f.column, f.op, f.value) for f in expected) == sorted(_filter_key(f["column"], f["op"], f["value"]) for f in predicted)


@dataclass
class PlanVerdict:
    metric_ok: bool = False
    columns_ok: bool = False
    grouping_ok: bool = False
    filters_ok: bool = False
    chart_ok: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def accurate(self) -> bool:
        """Совпали показатель, колонки и группировка (определение метрики Analysis Plan Accuracy)."""
        return self.metric_ok and self.columns_ok and self.grouping_ok


def score_plan(case: GoldenCase, predicted: dict[str, Any] | None) -> PlanVerdict:
    verdict = PlanVerdict()
    if predicted is None or predicted.get("metric") is None:
        verdict.reasons.append("план не содержит шага с показателем")
        return verdict
    verdict.metric_ok = predicted["metric"] == case.expected_metric
    verdict.columns_ok = set(predicted["columns"]) == set(case.expected_columns)
    verdict.grouping_ok = list(predicted["group_by"]) == list(case.expected_group_by)
    verdict.filters_ok = filters_match(case.expected_filters, predicted["filters"])
    verdict.chart_ok = predicted.get("chart_type") == case.expected_chart_type
    if not verdict.metric_ok:
        verdict.reasons.append(f"показатель {predicted['metric']!r}, ожидался {case.expected_metric!r}")
    if not verdict.columns_ok:
        verdict.reasons.append(f"колонки {sorted(predicted['columns'])}, ожидались {sorted(case.expected_columns)}")
    if not verdict.grouping_ok:
        verdict.reasons.append(f"группировка {predicted['group_by']}, ожидалась {case.expected_group_by}")
    if not verdict.filters_ok:
        verdict.reasons.append(f"фильтры {predicted['filters']}, ожидались {[f.model_dump() for f in case.expected_filters]}")
    return verdict


# --------------------------------------------------------------------------- Numeric Accuracy
@dataclass
class NumericVerdict:
    ok: bool
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    detail: str = ""


def _key(group: dict[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in group.items()))


def numbers_close(actual: float | None, expected: float) -> bool:
    return actual is not None and math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def score_numeric(expected: ExpectedNumeric, rows: list[dict[str, Any]], overall: float | None) -> NumericVerdict:
    """rows — строки результата MCP: [{group: {...}, value: число}, …]. Группы и значения должны совпасть."""
    actual = {_key(r["group"]): r["value"] for r in rows}
    wanted = {_key(r.group): r.value for r in expected.rows}
    if set(actual) != set(wanted):
        missing, extra = sorted(set(wanted) - set(actual)), sorted(set(actual) - set(wanted))
        return NumericVerdict(False, detail=f"группы не совпали: нет {missing[:3]}, лишние {extra[:3]}")
    worst_abs = worst_rel = 0.0
    for key, value in wanted.items():
        got = actual[key]
        if got is None or not numbers_close(got, value):
            return NumericVerdict(False, detail=f"{dict(key)}: получено {got}, ожидалось {value}")
        worst_abs = max(worst_abs, abs(got - value))
        worst_rel = max(worst_rel, abs(got - value) / abs(value) if value else 0.0)
    if not numbers_close(overall, expected.overall):
        return NumericVerdict(False, detail=f"итог: получено {overall}, ожидалось {expected.overall}")
    return NumericVerdict(True, worst_abs, worst_rel)


# --------------------------------------------------------------------------- агрегаты
def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def percentile(values: list[float], q: float) -> float | None:
    """Перцентиль с линейной интерполяцией (q от 0 до 100)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 1)


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": round(sum(values) / len(values), 1), "median": percentile(values, 50), "p95": percentile(values, 95), "max": round(max(values), 1)}
