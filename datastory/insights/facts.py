"""Числовые доказательства для выводов.

Единственный источник чисел для вывода — результат MCP calculate_metrics. Здесь из его строк собирается набор
фактов: сами значения и простые сравнения (разность соседних периодов, доля категории в итоге). Сравнения считает
обычный Python, а не LLM; у каждого факта записано, как он получен. LLM получает готовые факты и обязан
использовать только их — числа вне набора проверяет verifier.
"""
from __future__ import annotations

from datastory.analytics.formatting import format_metric_value, format_number
from datastory.analytics.models import MetricResult
from datastory.models import NumericEvidence

MAX_CATEGORY_FACTS = 12


def _diff_text(delta: float, unit: str) -> str:
    return f"{format_number(abs(delta), 2)} п.п." if unit == "%" else format_metric_value(abs(delta), unit)


def _direction(delta: float) -> str:
    return "Без изменения" if delta == 0 else "Рост" if delta > 0 else "Снижение"


def _source(result: MetricResult, detail: str = "") -> str:
    by = f", group_by={result.group_by}" if result.group_by else ""
    return f"calculate_metrics({result.metric}{by}){detail}"


def _time_series_facts(result: MetricResult) -> list[NumericEvidence]:
    axis = result.group_by[0]
    points = [(str(r.group.get(axis)), r.value) for r in result.rows if r.value is not None]
    unit, label = result.unit, result.label
    facts: list[NumericEvidence] = []

    def value(fid: str, text: str, number: float, detail: str, group: str | None = None) -> None:
        facts.append(
            NumericEvidence(id=fid, label=text, value=number, formatted=format_metric_value(number, unit), unit=unit,
                            kind="value", computation=_source(result, detail), group=group)
        )

    def diff(fid: str, a: tuple[str, float], b: tuple[str, float], prefix: str = "") -> None:
        delta = b[1] - a[1]
        facts.append(
            NumericEvidence(
                id=fid, label=f"{prefix}{_direction(delta)} «{label}»: {a[0]} → {b[0]}", value=delta,
                formatted=_diff_text(delta, unit), unit="п.п." if unit == "%" else unit, kind="difference", group=f"{a[0]} → {b[0]}",
                computation=f"значение({b[0]}) − значение({a[0]}) по результату calculate_metrics",
            )
        )

    if result.overall.value is not None:
        value("overall", f"{label}: итог по всем строкам", result.overall.value, " → overall")
    if not points:
        return facts
    first, last = points[0], points[-1]
    lowest, highest = min(points, key=lambda p: p[1]), max(points, key=lambda p: p[1])
    value("first", f"{label} в первом периоде ({first[0]})", first[1], f" → строка {axis}={first[0]}", first[0])
    if len(points) > 1:
        value("last", f"{label} в последнем периоде ({last[0]})", last[1], f" → строка {axis}={last[0]}", last[0])
        value("min", f"Минимум {label}: период {lowest[0]}", lowest[1], f" → строка {axis}={lowest[0]}", lowest[0])
        value("max", f"Максимум {label}: период {highest[0]}", highest[1], f" → строка {axis}={highest[0]}", highest[0])
        diff("change", first, last, "Общее изменение за весь ряд: ")
        if unit != "%" and first[1]:
            pct = (last[1] - first[1]) / first[1] * 100
            facts.append(
                NumericEvidence(
                    id="change_pct", label=f"Относительное изменение «{label}»: {first[0]} → {last[0]}", value=pct,
                    formatted=f"{format_number(abs(pct), 1)}%", unit="%", kind="difference", group=f"{first[0]} → {last[0]}",
                    computation=f"(значение({last[0]}) − значение({first[0]})) / значение({first[0]}) × 100",
                )
            )
        steps = list(zip(points, points[1:]))
        drop = min(steps, key=lambda s: s[1][1] - s[0][1])
        rise = max(steps, key=lambda s: s[1][1] - s[0][1])
        if drop[1][1] - drop[0][1] < 0 and len(points) > 2:
            diff("drop", *drop, prefix="Наибольшее падение между соседними периодами: ")
        if rise[1][1] - rise[0][1] > 0 and len(points) > 2:
            diff("rise", *rise, prefix="Наибольший рост между соседними периодами: ")
    facts.append(
        NumericEvidence(id="periods", label="Число периодов в ряду", value=len(points), formatted=str(len(points)), kind="count",
                        computation="число строк результата calculate_metrics")
    )
    return facts


def _category_facts(result: MetricResult) -> list[NumericEvidence]:
    axis = result.group_by[0]
    unit, label = result.unit, result.label
    rows = sorted((r for r in result.rows if r.value is not None), key=lambda r: r.value, reverse=True)
    total = result.overall.value
    facts: list[NumericEvidence] = []
    if total is not None:
        facts.append(
            NumericEvidence(id="overall", label=f"{label}: итог по всем категориям", value=total,
                            formatted=format_metric_value(total, unit), unit=unit, kind="value",
                            computation=_source(result, " → overall"))
        )
    for i, row in enumerate(rows[:MAX_CATEGORY_FACTS], 1):
        name = str(row.group.get(axis))
        facts.append(
            NumericEvidence(id=f"cat{i}_value", label=f"{label}: {name}", value=row.value,
                            formatted=format_metric_value(row.value, unit), unit=unit, kind="value",
                            computation=_source(result, f" → строка {axis}={name}"), group=name)
        )
        if total:
            share = row.value / total * 100
            facts.append(
                NumericEvidence(id=f"cat{i}_share", label=f"Доля «{name}» в итоге", value=share,
                                formatted=f"{format_number(share, 1)}%", unit="%", kind="share", group=name,
                                computation=f"значение({name}) / итог × 100 по результату calculate_metrics")
            )
    facts.append(
        NumericEvidence(id="categories", label="Число категорий", value=len(rows), formatted=str(len(rows)), kind="count",
                        computation="число строк результата calculate_metrics")
    )
    return facts


def build_facts(result: MetricResult, role: str) -> list[NumericEvidence]:
    """role: 'time' — динамика по периодам, 'category' — распределение по категориям."""
    if len(result.group_by) != 1:
        return []
    return _time_series_facts(result) if role == "time" else _category_facts(result)


def group_labels(result: MetricResult) -> list[str]:
    """Значения группировки (периоды, категории): числа внутри них — не «числа вывода»."""
    return [str(v) for row in result.rows for v in row.group.values()]
