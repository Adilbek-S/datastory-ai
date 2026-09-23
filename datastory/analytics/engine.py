"""Analytics Engine: расчёт показателей. Пока — только базовые метрики набора данных."""
from __future__ import annotations

from datastory.models import KPI, DatasetProfile


def basic_kpis(profile: DatasetProfile) -> list[KPI]:
    total_cells = profile.row_count * profile.column_count
    missing_pct = (profile.missing_cell_count / total_cells * 100) if total_cells else 0.0
    return [
        KPI(label="Строк", value=f"{profile.row_count:,}".replace(",", " ")),
        KPI(label="Столбцов", value=str(profile.column_count)),
        KPI(
            label="Числовых показателей",
            value=str(len(profile.detected_numeric_columns)),
            hint="Кандидаты для расчётов",
        ),
        KPI(label="Пропусков", value=f"{missing_pct:.1f}%", hint=f"{profile.missing_cell_count} ячеек"),
    ]
