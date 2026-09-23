"""Visualization Engine: подбор и построение графиков Plotly."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from datastory.models import ChartSpec, DatasetProfile

PALETTE = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444"]


def suggest_charts(profile: DatasetProfile) -> list[ChartSpec]:
    """Эвристический подбор графиков по профилю. Подбор через LLM появится позже."""
    numeric = profile.detected_numeric_columns
    categorical = profile.detected_category_columns
    datetime_cols = profile.detected_date_columns
    # Разбивка по категории читаема, только если категорий немного.
    split = next((c for c in categorical if profile.column(c).unique_count <= 8), None)

    specs: list[ChartSpec] = []
    if numeric:
        specs.append(ChartSpec(title=f"Распределение: {numeric[0]}", kind="histogram", x=numeric[0]))
    if categorical:
        specs.append(ChartSpec(title=f"Частоты: {categorical[0]}", kind="bar", x=categorical[0]))
    if datetime_cols and numeric:
        specs.append(
            ChartSpec(title=f"Динамика: {numeric[0]}", kind="line", x=datetime_cols[0], y=numeric[0], color=split)
        )
    if len(numeric) >= 2:
        specs.append(
            ChartSpec(title=f"{numeric[0]} и {numeric[1]}", kind="scatter", x=numeric[0], y=numeric[1], color=split)
        )
    return specs


_ALLOWED_SETTINGS = {"markers", "barmode", "category_orders"}  # белый список настроек из ChartSpec.settings


def _figure_from_data(spec: ChartSpec) -> Figure:
    """График из ChartSpec с данными (результат create_chart_spec): без DataFrame и без обращения к датасету."""
    frame = pd.DataFrame(spec.data)
    settings = {k: v for k, v in spec.settings.items() if k in _ALLOWED_SETTINGS}
    if spec.kind == "line":
        fig = px.line(frame, x=spec.x, y=spec.y, color=spec.color, color_discrete_sequence=PALETTE, **settings)
    elif spec.kind == "bar":
        fig = px.bar(frame, x=spec.x, y=spec.y, color=spec.color, color_discrete_sequence=PALETTE, **settings)
    elif spec.kind == "pie":
        fig = px.pie(frame, names=spec.x, values=spec.y, color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="inside", textinfo="percent+label")
    else:
        raise ValueError(f"Неизвестный тип графика: {spec.kind}")
    if spec.kind != "pie":
        fig.update_layout(xaxis_title=spec.x_title or spec.x, yaxis_title=spec.y_title or spec.y)
    return fig


def build_figure(df: pd.DataFrame, spec: ChartSpec) -> Figure:
    if spec.data:
        fig = _figure_from_data(spec)
    elif spec.kind == "histogram":
        fig = px.histogram(df, x=spec.x, color_discrete_sequence=PALETTE)
    elif spec.kind == "bar":
        counts = df[spec.x].value_counts().head(15).rename_axis(spec.x).reset_index(name="count")
        fig = px.bar(counts, x=spec.x, y="count", color_discrete_sequence=PALETTE)
    elif spec.kind == "line":
        fig = px.line(df.sort_values(spec.x), x=spec.x, y=spec.y, color=spec.color, markers=True, color_discrete_sequence=PALETTE)
    elif spec.kind == "scatter":
        fig = px.scatter(df, x=spec.x, y=spec.y, color=spec.color, color_discrete_sequence=PALETTE)
    else:
        raise ValueError(f"Неизвестный тип графика: {spec.kind}")

    fig.update_layout(
        title=spec.title,
        template="plotly_white",
        margin=dict(l=16, r=16, t=48, b=16),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    return fig
