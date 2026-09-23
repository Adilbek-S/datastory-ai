"""Analytics Engine: расчёт показателей. Пока — только базовые метрики набора данных."""
from __future__ import annotations

from datastory.models import KPI, ColumnKind, DatasetProfile


def basic_kpis(profile: DatasetProfile) -> list[KPI]:
    total_cells = profile.rows * profile.columns
    missing_pct = (profile.missing_cells / total_cells * 100) if total_cells else 0.0
    return [
        KPI(label="Строк", value=f"{profile.rows:,}".replace(",", " ")),
        KPI(label="Столбцов", value=str(profile.columns)),
        KPI(
            label="Числовых показателей",
            value=str(len(profile.columns_of(ColumnKind.NUMERIC))),
            hint="Кандидаты для расчётов",
        ),
        KPI(label="Пропусков", value=f"{missing_pct:.1f}%", hint=f"{profile.missing_cells} ячеек"),
    ]
