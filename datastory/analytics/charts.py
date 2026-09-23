"""create_chart_spec: спецификация графика Plotly из результата calculate_metrics.

Возвращается ChartSpec с данными и настройками; изображение не создаётся. Отрисовывает Streamlit.
"""
from __future__ import annotations

from datastory.analytics.models import MetricResult
from datastory.errors import AnalyticsError
from datastory.models import ChartSpec

CHART_TYPES = ("line", "bar", "pie")
MAX_POINTS = 500
MAX_TITLE = 200
# Круговая диаграмма показывает доли целого: у отношений (%, средние) части не складываются в целое.
NON_ADDITIVE_METRICS = ("success_rate", "average_transaction_amount")


def _natural_order(values: list) -> list:
    unique = list(dict.fromkeys(values))
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in unique if v is not None)
    return sorted(unique, key=lambda v: (v is None, v if v is not None and (numeric or isinstance(v, str)) else str(v)))


def build_chart_spec(result: MetricResult, chart_type: str, title: str, x_axis: str, y_axis: str = "value") -> ChartSpec:
    if chart_type not in CHART_TYPES:
        raise AnalyticsError(f"Неизвестный тип графика {chart_type!r}. Поддерживаются: {', '.join(CHART_TYPES)}.")
    title = (title or "").strip()
    if not title:
        raise AnalyticsError("title: укажите непустое название графика.")
    if len(title) > MAX_TITLE:
        raise AnalyticsError(f"title: не более {MAX_TITLE} символов.")
    if not result.group_by:
        raise AnalyticsError("Для графика нужен результат calculate_metrics с group_by: без группировки одно число, а не ряд данных.")
    if len(result.group_by) > 2:
        raise AnalyticsError("График строится по одной или двум колонкам группировки (ось X и, при необходимости, серии).")
    if x_axis not in result.group_by:
        raise AnalyticsError(f"x_axis: колонка «{x_axis}» отсутствует в group_by результата. Доступно: {', '.join(result.group_by)}.")
    if y_axis not in ("value", result.metric):
        raise AnalyticsError(f"y_axis: допустимо только «value» или «{result.metric}» (значение метрики), получено {y_axis!r}.")

    series_columns = [c for c in result.group_by if c != x_axis]
    series = series_columns[0] if series_columns else None
    if chart_type == "pie":
        if series:
            raise AnalyticsError("Круговая диаграмма строится по одной колонке группировки; для двух используйте bar или line.")
        if result.metric in NON_ADDITIVE_METRICS:
            raise AnalyticsError(
                f"Круговая диаграмма показывает доли целого, а {result.metric} ({result.unit}) — отношение: его части не складываются "
                "в целое. Используйте bar или line."
            )

    points = [(r.group.get(x_axis), r.group.get(series) if series else None, r.value) for r in result.rows]
    if len(points) > MAX_POINTS:
        raise AnalyticsError(f"Слишком много точек ({len(points)}, максимум {MAX_POINTS}). Добавьте фильтры или выберите другую группировку.")
    if chart_type == "pie":
        points = [p for p in points if p[2] is not None]
        if not any(p[2] > 0 for p in points):
            raise AnalyticsError("Для круговой диаграммы нет положительных значений.")
        if any(p[2] < 0 for p in points):
            raise AnalyticsError("Круговая диаграмма не показывает отрицательные значения.")

    x_order = _natural_order([p[0] for p in points])
    series_order = _natural_order([p[1] for p in points]) if series else []
    rank = {v: i for i, v in enumerate(x_order)}
    series_rank = {v: i for i, v in enumerate(series_order)}
    points.sort(key=lambda p: (rank[p[0]], series_rank.get(p[1], 0)))

    metric_key = result.metric
    data = [
        {x_axis: x, **({series: s} if series else {}), metric_key: value}
        for x, s, value in points
    ]
    settings: dict = {"category_orders": {x_axis: [v for v in x_order if v is not None]}}
    if series:
        settings["category_orders"][series] = [v for v in series_order if v is not None]
    if chart_type == "line":
        settings["markers"] = True
    if chart_type == "bar" and series:
        settings["barmode"] = "group"

    filters = "; ".join(f"{c.column} {c.op.value} {c.value}" for c in result.filters)
    description = f"{result.label} = {result.formula}." + (f" Фильтры: {filters}." if filters else "")
    return ChartSpec(
        title=title,
        kind=chart_type,
        x=x_axis,
        y=metric_key,
        color=series,
        description=description,
        data=data,
        x_title=x_axis,
        y_title=f"{result.label}, {result.unit}",
        unit=result.unit,
        settings=settings,
        source_metric=result.metric,
    )
