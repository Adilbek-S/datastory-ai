"""LangGraph Workflow: profile -> metrics -> charts -> insights.

Все аналитические шаги выполняются MCP-сервером: узлы графа вызывают инструменты profile_dataset,
calculate_metrics и create_chart_spec через настоящий MCP-клиент (stdio). Функции расчёта здесь
не импортируются и не вызываются напрямую: ни арифметики, ни группировки в самом графе нет.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from datastory.analytics.engine import basic_kpis, metric_kpis
from datastory.analytics.models import AnalysisResult, DatasetSummary, MetricResult
from datastory.insights.generator import generate_insights, generate_metric_insights
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.models import KPI, ChartSpec, Insight

OVERALL_METRICS = ("transaction_count", "success_rate", "transaction_volume", "average_transaction_amount")
MAX_SERIES = 12  # больше линий/столбцов на графике — нечитаемо


class WorkflowState(TypedDict, total=False):
    dataset_id: str
    summary: DatasetSummary
    overall: dict[str, MetricResult]  # метрика -> итог по всему датасету
    grouped: dict[str, MetricResult]  # ключ запроса -> результат с группировкой
    kpis: list[KPI]
    charts: list[ChartSpec]
    insights: list[Insight]


def _dimensions(summary: DatasetSummary) -> tuple[str | None, str | None]:
    """(временное измерение, категориальное измерение с небольшим числом значений)."""
    unique = {c.name: c.unique_count for c in summary.columns}
    time_dim = next((n for n in summary.dimensions if summary.column_types[n] == "datetime"), None)
    category = next(
        (n for n in summary.dimensions if summary.column_types[n] == "categorical" and unique[n] <= MAX_SERIES), None
    )
    return time_dim, category


def build_graph(client: AnalyticsMcpClient):
    def profile_node(state: WorkflowState) -> WorkflowState:
        return {"summary": client.profile_dataset(state["dataset_id"])}

    def metrics_node(state: WorkflowState) -> WorkflowState:
        summary, dataset_id = state["summary"], state["dataset_id"]
        available = set(summary.available_metrics)
        overall = {m: client.calculate_metrics(dataset_id, m) for m in OVERALL_METRICS if m in available}
        time_dim, category = _dimensions(summary)

        grouped: dict[str, MetricResult] = {}

        def add(key: str, metric: str, group_by: list[str | None]) -> None:
            if metric in available and all(group_by):  # нужны все колонки группировки
                grouped[key] = client.calculate_metrics(dataset_id, metric, group_by)

        add("success_rate_by_period", "success_rate", [time_dim])
        add("success_rate_by_period_and_category", "success_rate", [time_dim, category])
        if time_dim is None:  # без временной оси Success Rate показываем по категориям
            add("success_rate_by_category", "success_rate", [category])
        add("volume_by_category", "transaction_volume", [category])
        add("average_amount_by_category", "average_transaction_amount", [category])
        add("count_by_period", "transaction_count", [time_dim])
        return {"overall": overall, "grouped": grouped}

    def charts_node(state: WorkflowState) -> WorkflowState:
        grouped = state["grouped"]
        time_dim, category = _dimensions(state["summary"])
        charts: list[ChartSpec] = []

        def chart(key: str, kind: str, title: str, x: str | None) -> bool:
            if key in grouped and x:
                charts.append(client.create_chart_spec(grouped[key], kind, title, x))
                return True
            return False

        # Success Rate: лучший из доступных вариантов (периоды × категории, только периоды, только категории)
        for key, kind, title, x in (
            ("success_rate_by_period_and_category", "line", "Success Rate по периодам и категориям", time_dim),
            ("success_rate_by_period", "line", "Success Rate по периодам", time_dim),
            ("success_rate_by_category", "bar", f"Success Rate: {category}", category),
        ):
            if chart(key, kind, title, x):
                break
        chart("volume_by_category", "pie", f"Доля в объёме транзакций: {category}", category)
        chart("average_amount_by_category", "bar", f"Средняя сумма транзакции: {category}", category)
        chart("count_by_period", "bar", "Количество транзакций по периодам", time_dim)
        return {"charts": charts}

    def insights_node(state: WorkflowState) -> WorkflowState:
        summary, overall = state["summary"], state["overall"]
        insights = generate_insights(summary) + generate_metric_insights(state["grouped"].get("success_rate_by_period"))
        kpis = metric_kpis(overall) or basic_kpis(summary)
        return {"insights": insights, "kpis": kpis}

    graph = StateGraph(WorkflowState)
    graph.add_node("profile", profile_node)
    graph.add_node("metrics", metrics_node)
    graph.add_node("charts", charts_node)
    graph.add_node("insights", insights_node)
    graph.add_edge(START, "profile")
    graph.add_edge("profile", "metrics")
    graph.add_edge("metrics", "charts")
    graph.add_edge("charts", "insights")
    graph.add_edge("insights", END)
    return graph.compile()


def run_analysis_for_dataset(dataset_id: str, client: AnalyticsMcpClient) -> AnalysisResult:
    """Анализирует подтверждённый датасет: все расчёты выполняет MCP-сервер.

    Бросает McpConnectionError, если сервер недоступен, и McpToolError, если инструмент отклонил запрос.
    """
    state = build_graph(client).invoke({"dataset_id": dataset_id})
    return AnalysisResult(
        dataset_id=dataset_id,
        summary=state["summary"],
        kpis=state["kpis"],
        metrics=[*state["overall"].values(), *state["grouped"].values()],
        charts=state["charts"],
        insights=state["insights"],
    )
