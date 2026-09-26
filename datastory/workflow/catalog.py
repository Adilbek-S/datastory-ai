"""Каталог анализов MVP: четыре вида графиков и то, что нужно каждому из них.

Названия, единицы и формулы показателей берутся из определений MCP-сервера (datastory.analytics.metrics.METRICS):
здесь только метаданные, расчётные функции не вызываются — числа считает инструмент calculate_metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from datastory.analytics.metrics import METRICS
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


ANALYSES: tuple[AnalysisKind, ...] = (
    AnalysisKind(
        "count_dynamics", "Динамика количества операций", "transaction_count", "time", "line", ("line",),
        "количество транзакций определение колонки Transactions", ("количество транзакций", "число инициированных транзакций"),
        "месяц период дата",
    ),
    AnalysisKind(
        "volume_dynamics", "Динамика объёма", "transaction_volume", "time", "line", ("line",),
        "Transaction Volume объём транзакций определение", ("transaction volume", "объем транзакций"),
        "месяц период дата",
    ),
    AnalysisKind(
        "success_dynamics", "Динамика успешности", "success_rate", "time", "line", ("line",),
        "Success Rate успешность транзакций формула определение", ("success rate", "успешность"),
        "месяц период дата",
    ),
    AnalysisKind(
        "channel_distribution", "Распределение по каналам", "transaction_count", "category", "pie", ("pie", "bar"),
        "количество транзакций определение колонки Transactions", ("количество транзакций", "число инициированных транзакций"),
        "канал платежа",
    ),
)
BY_ID = {a.id: a for a in ANALYSES}
ALL_IDS = [a.id for a in ANALYSES]


def metric_columns(metric: str) -> list[str]:
    return list(METRICS[metric].columns)


def time_columns(summary: DatasetSummary) -> list[str]:
    return [n for n in summary.dimensions if summary.column_types[n] == "datetime"]


def category_columns(summary: DatasetSummary) -> list[str]:
    unique = {c.name: c.unique_count for c in summary.columns}
    return [
        n for n in summary.dimensions
        if summary.column_types[n] == "categorical" and MIN_CATEGORIES <= unique[n] <= MAX_CATEGORIES
    ]


def evaluate_candidates(summary: DatasetSummary) -> list[AnalysisCandidate]:
    """Какие из четырёх анализов возможны для датасета. Нет нужной колонки — анализ недоступен, причина названа."""
    have = {n.casefold() for n in summary.column_names}
    candidates = []
    for kind in ANALYSES:
        reasons: list[str] = []
        label = METRICS[kind.metric].label
        if kind.metric not in summary.available_metrics:
            missing = [c for c in metric_columns(kind.metric) if c.casefold() not in have]
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
                id=kind.id, label=kind.label, metric=kind.metric, available=not reasons, reasons=reasons,
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
