"""Типизированный клиент аналитических MCP-инструментов.

Каждый метод — настоящий вызов инструмента по протоколу MCP через stdio; локальные функции
расчёта здесь не вызываются. Ответы сервера разбираются в Pydantic-модели.
"""
from __future__ import annotations

import sys
from typing import Any

from mcp.client.stdio import get_default_environment

from datastory.analytics.models import DatasetSummary, FilterCondition, MetricResult
from datastory.config import PROJECT_ROOT, Settings, get_settings
from datastory.mcp_client.connection import McpConnection, McpServerConfig
from datastory.models import ChartSpec

TOOLS = ("profile_dataset", "calculate_metrics", "create_chart_spec")


def analytics_server_config(settings: Settings | None = None) -> McpServerConfig:
    """Команда запуска аналитического сервера. Хранилище датасетов передаётся через окружение."""
    settings = settings or get_settings()
    env = {
        **get_default_environment(),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "WORKSPACE_DIR": str(settings.workspace_dir),
    }
    return McpServerConfig(
        command=sys.executable,
        args=("-m", settings.mcp_server_module),
        env=env,
        cwd=str(PROJECT_ROOT),
        name="datastory-analytics",
    )


class AnalyticsMcpClient:
    def __init__(self, connection: McpConnection):
        self.connection = connection
        self.calls: list[str] = []  # названия вызванных инструментов (диагностика и тесты)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "AnalyticsMcpClient":
        settings = settings or get_settings()
        return cls(
            McpConnection(
                analytics_server_config(settings),
                startup_timeout=settings.mcp_startup_timeout,
                call_timeout=settings.mcp_call_timeout,
            )
        )

    # ------------------------------------------------------------------ жизненный цикл
    @property
    def is_closed(self) -> bool:
        return self.connection.is_closed

    def list_tools(self) -> list[str]:
        return self.connection.list_tools()

    def close(self) -> None:
        self.connection.close()

    # ------------------------------------------------------------------ инструменты
    def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        self.calls.append(tool)
        return self.connection.call_tool(tool, arguments)

    def profile_dataset(self, dataset_id: str) -> DatasetSummary:
        return DatasetSummary.model_validate(self._call("profile_dataset", {"dataset_id": dataset_id}))

    def calculate_metrics(
        self,
        dataset_id: str,
        metric: str,
        group_by: list[str] | None = None,
        filters: list[FilterCondition | dict[str, Any]] | dict[str, Any] | None = None,
    ) -> MetricResult:
        arguments: dict[str, Any] = {"dataset_id": dataset_id, "metric": metric}
        if group_by:
            arguments["group_by"] = group_by
        if filters:
            arguments["filters"] = (
                filters if isinstance(filters, dict)
                else [f.model_dump(mode="json") if isinstance(f, FilterCondition) else f for f in filters]
            )
        return MetricResult.model_validate(self._call("calculate_metrics", arguments))

    def create_chart_spec(
        self, metric_result: MetricResult, chart_type: str, title: str, x_axis: str, y_axis: str = "value"
    ) -> ChartSpec:
        return ChartSpec.model_validate(
            self._call(
                "create_chart_spec",
                {
                    "metric_result": metric_result.model_dump(mode="json"),
                    "chart_type": chart_type,
                    "title": title,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                },
            )
        )
