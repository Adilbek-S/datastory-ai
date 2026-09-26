"""Типизированный клиент аналитических MCP-инструментов.

Каждый метод — настоящий вызов инструмента по протоколу MCP через stdio; локальные функции
расчёта здесь не вызываются. Ответы сервера разбираются в Pydantic-модели.
"""
from __future__ import annotations

import sys
import time
from typing import Any

from mcp.client.stdio import get_default_environment

from datastory.analytics.models import DatasetSummary, FilterCondition, MetricResult
from datastory.config import PROJECT_ROOT, Settings, get_settings
from datastory.mcp_client.connection import McpConnection, McpServerConfig
from datastory.models import ChartSpec
from datastory.observability import clip, milliseconds_since, span

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


def _summarize_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Короткое описание запроса для трассы: без больших вложенных результатов."""
    if tool == "create_chart_spec":
        result = arguments.get("metric_result") or {}
        return {
            "chart_type": arguments.get("chart_type"), "title": clip(arguments.get("title")), "x_axis": arguments.get("x_axis"),
            "metric": result.get("metric"), "points": len(result.get("rows", [])),
        }
    return {k: clip(v) if isinstance(v, str) else v for k, v in arguments.items()}


def _summarize_result(tool: str, result: Any) -> dict[str, Any]:
    """Сводка ответа инструмента: размеры и признаки, без значений таблицы."""
    if not isinstance(result, dict):
        return {}
    if tool == "profile_dataset":
        return {"rows": result.get("row_count"), "columns": result.get("column_count"), "available_metrics": result.get("available_metrics")}
    if tool == "calculate_metrics":
        return {"metric": result.get("metric"), "groups": len(result.get("rows", [])), "rows_matched": result.get("rows_matched"), "warnings": len(result.get("warnings", []))}
    if tool == "create_chart_spec":
        return {"kind": result.get("kind"), "points": len(result.get("data", []))}
    return {}


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
        """Вызов инструмента MCP. В LangSmith — span «mcp.<tool>»: название инструмента, длительность, успех или ошибка."""
        self.calls.append(tool)
        with span(
            f"mcp.{tool}", run_type="tool", tags=("mcp",), inputs={"tool": tool, "arguments": _summarize_arguments(tool, arguments)},
            metadata={"mcp.tool": tool, "mcp.server": "datastory-analytics"},
        ) as sp:
            started = time.perf_counter()
            try:
                result = self.connection.call_tool(tool, arguments)
            except Exception as exc:
                sp.metadata({"mcp.status": "error", "mcp.duration_ms": milliseconds_since(started), "mcp.error_type": type(exc).__name__})
                raise  # span закроется с ошибкой, вызывающий код увидит то же исключение
            duration = milliseconds_since(started)
            sp.metadata({"mcp.status": "success", "mcp.duration_ms": duration})
            sp.outputs({"tool": tool, "status": "success", "duration_ms": duration, **_summarize_result(tool, result)})
            return result

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
