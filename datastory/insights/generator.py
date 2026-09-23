"""Insight Generator. Заглушка: LLM-выводы (GPT-4o-mini) будут добавлены на следующем этапе."""
from __future__ import annotations

from datastory.models import DatasetProfile, Insight


def generate_insights(profile: DatasetProfile) -> list[Insight]:
    insights: list[Insight] = []
    if profile.duplicate_row_count:
        insights.append(
            Insight(
                title="Найдены дубликаты",
                text=f"В данных {profile.duplicate_row_count} повторяющихся строк.",
                severity="warning",
            )
        )
    if profile.missing_cell_count:
        insights.append(
            Insight(
                title="Есть пропуски",
                text=f"Всего пропущенных ячеек: {profile.missing_cell_count}.",
                severity="warning",
            )
        )
    if not insights:
        insights.append(
            Insight(title="Данные чистые", text="Пропусков и дубликатов не найдено.", severity="positive")
        )
    return insights
