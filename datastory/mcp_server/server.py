"""MCP-сервер аналитических инструментов DataStory AI (FastMCP, транспорт stdio).

Три инструмента: profile_dataset, calculate_metrics, create_chart_spec.
Все расчёты — обычный Python/Pandas-код: LLM не участвует в арифметике, а group_by и filters
проверяются по существующим колонкам и никогда не исполняются как код. Изображения сервер
не возвращает — только данные и ChartSpec; график рисует Streamlit через Plotly.

Запуск (stdio):  python -m datastory.mcp_server.server
Важно: по stdout идёт протокол MCP, поэтому в этом модуле нельзя использовать print().
"""
import functools
import logging
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from datastory.analytics.charts import CHART_TYPES, build_chart_spec
from datastory.analytics.metrics import METRICS, calculate_metric
from datastory.analytics.models import DatasetSummary, MetricResult
from datastory.analytics.summary import build_dataset_summary
from datastory.errors import AnalyticsError, DatasetNotFoundError
from datastory.models import ChartSpec
from datastory.storage.store import DatasetStore

logger = logging.getLogger("datastory.mcp.analytics")

mcp = FastMCP(
    "datastory-analytics",
    instructions=(
        "Аналитические инструменты по подтверждённым датасетам. Сначала profile_dataset (какие колонки и метрики доступны), "
        "затем calculate_metrics, затем create_chart_spec. Числа считает сервер, а не модель."
    ),
    log_level="WARNING",
)


def _guarded(tool):
    """Ошибки запроса превращаются в понятное сообщение инструмента; внутренние сбои не раскрывают детали."""

    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        try:
            return tool(*args, **kwargs)
        except AnalyticsError as exc:
            raise ToolError(exc.user_message) from None
        except DatasetNotFoundError as exc:
            raise ToolError(str(exc)) from None
        except ToolError:
            raise
        except Exception:  # noqa: BLE001 — неожиданный сбой: подробности только в лог (stderr)
            logger.exception("Сбой инструмента %s", tool.__name__)
            raise ToolError("Внутренняя ошибка сервера при выполнении инструмента. Подробности записаны в журнал сервера.") from None

    return wrapper


def _load(dataset_id: str):
    store = DatasetStore()
    return store.load_dataframe(dataset_id), store.load_profile(dataset_id)


@mcp.tool()
@_guarded
def profile_dataset(
    dataset_id: Annotated[str, Field(description="Идентификатор подтверждённого датасета")],
) -> DatasetSummary:
    """Профиль датасета: число строк, названия и типы колонок, числовые показатели, измерения,
    статистики, пропуски и список метрик, доступных для calculate_metrics.
    У колонок с персональными данными статистики скрыты."""
    return build_dataset_summary(DatasetStore().load_profile(dataset_id))


@mcp.tool()
@_guarded
def calculate_metrics(
    dataset_id: Annotated[str, Field(description="Идентификатор подтверждённого датасета")],
    metric: Annotated[
        str,
        Field(
            description="Метрика: transaction_count = SUM(Transactions); successful_count = SUM(Successful); "
            "failed_count = SUM(Failed); transaction_volume = SUM(Amount_KZT); "
            "success_rate = SUM(Successful) / SUM(Transactions) × 100; "
            "average_transaction_amount = SUM(Amount_KZT) / SUM(Transactions); "
            "sum:<колонка> = SUM(<числовая колонка>) для наборов данных без колонок платёжной системы",
            json_schema_extra={"enum": list(METRICS)},
        ),
    ],
    group_by: Annotated[
        list[str] | None,
        Field(description="До 3 существующих колонок для группировки, например [\"Month\", \"Channel\"]. Колонки с персональными данными недоступны."),
    ] = None,
    filters: Annotated[
        list[dict[str, Any]] | dict[str, Any] | None,
        Field(
            description="Условия отбора строк: список {column, op, value} (op: eq, ne, gt, gte, lt, lte, in, not_in, between) "
            "или краткая запись {\"Channel\": \"Mobile\"} / {\"Channel\": [\"Mobile\", \"Web\"]}. Даты: YYYY-MM или YYYY-MM-DD.",
            json_schema_extra={"examples": [[{"column": "Channel", "op": "in", "value": ["Mobile", "Web"]}, {"column": "Month", "op": "gte", "value": "2026-03"}]]},
        ),
    ] = None,
) -> MetricResult:
    """Считает метрику по датасету, при необходимости с группировкой и фильтрами.
    Отношения считаются как сумма к сумме (не среднее процентов по строкам). Возвращает значения по группам,
    итог по всем отобранным строкам, использованные суммы и предупреждения."""
    df, profile = _load(dataset_id)
    return calculate_metric(df, profile, dataset_id, metric, group_by, filters)


@mcp.tool()
@_guarded
def create_chart_spec(
    metric_result: Annotated[MetricResult, Field(description="Результат calculate_metrics (с group_by)")],
    chart_type: Annotated[str, Field(description="Тип графика: line, bar или pie", json_schema_extra={"enum": list(CHART_TYPES)})],
    title: Annotated[str, Field(description="Название графика")],
    x_axis: Annotated[str, Field(description="Колонка из group_by результата для оси X (для pie — подписи секторов)")],
    y_axis: Annotated[str, Field(description="Всегда значение метрики: «value» или название метрики")] = "value",
) -> ChartSpec:
    """Возвращает спецификацию графика Plotly (данные и настройки) для отрисовки в Streamlit.
    Если в group_by две колонки, вторая становится сериями. Круговая диаграмма недоступна для отношений
    (success_rate, average_transaction_amount). Изображение не создаётся."""
    return build_chart_spec(metric_result, chart_type, title, x_axis, y_axis)


if __name__ == "__main__":
    mcp.run()
