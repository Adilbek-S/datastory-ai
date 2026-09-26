"""Каталог анализов MVP: четыре вида графиков и то, что нужно каждому из них.

Названия, единицы и формулы показателей берутся из определений MCP-сервера (datastory.analytics.metrics.METRICS):
здесь только метаданные, расчётные функции не вызываются — числа считает инструмент calculate_metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from datastory.analytics.metrics import METRICS, SUM_PREFIX, get_metric
from datastory.analytics.models import DatasetSummary
from datastory.workflow.models import AnalysisCandidate

MIN_CATEGORIES = 2
MAX_CATEGORIES = 12  # больше значений на оси — нечитаемый график
PIE_MAX_CATEGORIES = 6  # круговая диаграмма — только для небольшого числа категорий


@dataclass(frozen=True)
class AnalysisKind:
    id: str
    label: str
    metric: str
    role: Literal["time", "category"]  # тип измерения для оси X
    default_chart: str
    allowed_charts: tuple[str, ...]
    doc_query: str  # запрос к RAG за определением показателя
    doc_terms: tuple[str, ...]  # фрагмент считается определением, только если называет показатель
    role_phrase: str  # фраза для сопоставления оси X с колонкой через RAG
    purpose: str = ""  # зачем нужен анализ (карточка плана)
    generic: bool = False  # универсальный анализ суммы числовой колонки; метрика выбирается по датасету


ANALYSES: tuple[AnalysisKind, ...] = (
    AnalysisKind(
        "count_dynamics", "Динамика количества операций", "transaction_count", "time", "line", ("line",),
        "количество транзакций определение колонки Transactions", ("количество транзакций", "число инициированных транзакций"),
        "месяц период дата",
        "Показывает, растёт или падает нагрузка на платёжную систему от периода к периоду.",
    ),
    AnalysisKind(
        "volume_dynamics", "Динамика объёма", "transaction_volume", "time", "line", ("line",),
        "Transaction Volume объём транзакций определение", ("transaction volume", "объем транзакций"),
        "месяц период дата",
        "Показывает, как менялся денежный объём платежей.",
    ),
    AnalysisKind(
        "success_dynamics", "Динамика успешности", "success_rate", "time", "line", ("line",),
        "Success Rate успешность транзакций формула определение", ("success rate", "успешность"),
        "месяц период дата",
        "Помогает заметить снижения успешности транзакций и оценить их масштаб.",
    ),
    AnalysisKind(
        "channel_distribution", "Распределение по каналам", "transaction_count", "category", "pie", ("pie", "bar"),
        "количество транзакций определение колонки Transactions", ("количество транзакций", "число инициированных транзакций"),
        "канал платежа",
        "Показывает, через какие каналы идёт основная часть операций.",
    ),
)
# Универсальные анализы включаются только для наборов без колонок платёжной системы (например, продажи):
# показатель — сумма основной числовой колонки, метрика «sum:<колонка>» считается тем же MCP-инструментом.
GENERIC_ANALYSES: tuple[AnalysisKind, ...] = (
    AnalysisKind(
        "measure_dynamics", "Динамика показателя", "", "time", "line", ("line",), "", (), "месяц период дата",
        "Показывает, как менялся основной показатель от периода к периоду.", generic=True,
    ),
    AnalysisKind(
        "measure_comparison", "Сравнение по категориям", "", "category", "bar", ("bar", "pie"), "", (), "категория регион",
        "Показывает, какие категории дают наибольший вклад в основной показатель.", generic=True,
    ),
)
EVERY_ANALYSIS = ANALYSES + GENERIC_ANALYSES
BY_ID = {a.id: a for a in EVERY_ANALYSIS}
ALL_IDS = [a.id for a in ANALYSES]
MEASURE_HINTS = ("revenue", "sales", "amount", "income", "turnover", "total", "выруч", "продаж", "сумм", "доход", "оборот")


def measure_metrics(summary: DatasetSummary) -> list[str]:
    return [m for m in summary.available_metrics if m.startswith(SUM_PREFIX)]


def primary_measure(measures: list[str]) -> str:
    """Основной показатель: колонка с названием вроде Revenue/Sales/Выручка, иначе первая числовая."""
    for metric in measures:
        if any(hint in metric.casefold() for hint in MEASURE_HINTS):
            return metric
    return measures[0]


def is_generic_mode(summary: DatasetSummary) -> bool:
    """Нет показателей платёжной системы, но есть числовые колонки: анализируем суммы колонок."""
    return not any(m in METRICS for m in summary.available_metrics) and bool(measure_metrics(summary))


def doc_query(kind: AnalysisKind, metric: str) -> str:
    return kind.doc_query or f"{get_metric(metric).label} определение показателя"


def doc_terms(kind: AnalysisKind, metric: str) -> tuple[str, ...]:
    if kind.doc_terms:
        return kind.doc_terms
    definition = get_metric(metric)
    return (definition.label.casefold(), definition.columns[0].casefold())


def metric_columns(metric: str) -> list[str]:
    return list(get_metric(metric).columns)


def time_columns(summary: DatasetSummary) -> list[str]:
    return [n for n in summary.dimensions if summary.column_types[n] == "datetime"]


def category_columns(summary: DatasetSummary) -> list[str]:
    unique = {c.name: c.unique_count for c in summary.columns}
    return [
        n for n in summary.dimensions
        if summary.column_types[n] == "categorical" and MIN_CATEGORIES <= unique[n] <= MAX_CATEGORIES
    ]


def evaluate_candidates(summary: DatasetSummary) -> list[AnalysisCandidate]:
    """Какие анализы возможны для датасета. Нет нужной колонки — анализ недоступен, причина названа.

    Платёжные показатели — если есть их колонки (или нет числовых колонок вовсе: тогда причины названы); иначе универсальные.
    """
    have = {n.casefold() for n in summary.column_names}
    generic = is_generic_mode(summary)
    measure = primary_measure(measure_metrics(summary)) if generic else ""
    candidates = []
    for kind in (GENERIC_ANALYSES if generic else ANALYSES):
        reasons: list[str] = []
        metric = measure if kind.generic else kind.metric
        label = get_metric(metric).label
        title = f"{kind.label}: {label}" if kind.generic else kind.label
        if metric not in summary.available_metrics:
            missing = [c for c in metric_columns(metric) if c.casefold() not in have]
            reasons.append(
                f"для показателя «{label}» нет колонки {', '.join(f'«{c}»' for c in missing)}" if missing
                else f"колонки показателя «{label}» не числовые или содержат персональные данные"
            )
        dims = time_columns(summary) if kind.role == "time" else category_columns(summary)
        if not dims:
            reasons.append(
                "нет колонки с датой или периодом" if kind.role == "time"
                else f"нет категориальной колонки с числом значений от {MIN_CATEGORIES} до {MAX_CATEGORIES}"
            )
        candidates.append(
            AnalysisCandidate(
                id=kind.id, label=title, metric=metric, available=not reasons, reasons=reasons,
                dimension_candidates=dims,
            )
        )
    return candidates


def unique_count(summary: DatasetSummary, column: str) -> int:
    return next((c.unique_count for c in summary.columns if c.name == column), 0)


def fit_chart_type(kind: AnalysisKind, requested: str | None, categories: int | None) -> tuple[str, str]:
    """Тип графика по правилам визуализации. Возвращает (тип, пояснение — пусто, если запрос принят как есть)."""
    chart, note = requested or kind.default_chart, ""
    if chart not in kind.allowed_charts:
        note = (
            "круговая диаграмма не применяется к динамике во времени"
            if kind.role == "time" and chart == "pie"
            else f"для анализа «{kind.label}» допустимы типы: {', '.join(kind.allowed_charts)}"
        )
        note += f"; использован «{kind.default_chart}»"
        chart = kind.default_chart
    if chart == "pie" and categories is not None and categories > PIE_MAX_CATEGORIES:
        chart, note = "bar", f"категорий {categories}, а круговая диаграмма допустима не более чем для {PIE_MAX_CATEGORIES}; использован «bar»"
    return chart, note
