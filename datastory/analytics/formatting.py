"""Форматирование значений метрик для интерфейса (русская локаль: пробел между разрядами, запятая)."""
from __future__ import annotations


def format_number(value: float | int, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def format_metric_value(value: float | int | None, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "%":
        return f"{format_number(value, 2)}%"
    if unit == "KZT":
        magnitude = abs(value)
        if magnitude >= 1e9:
            return f"{format_number(value / 1e9, 2)} млрд KZT"
        if magnitude >= 1e6:
            return f"{format_number(value / 1e6, 2)} млн KZT"
        return f"{format_number(value)} KZT"
    return format_number(value)
