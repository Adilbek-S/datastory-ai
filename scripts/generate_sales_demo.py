"""Генератор демонстрационного файла продаж data/demo/sales_2026.xlsx (полностью синтетические данные, фиксированный seed).

Колонки: Month, Region, Orders, Revenue_KZT. Четыре региона, январь — июнь 2026. В апреле выручка Astana заметно падает
(в данных нет причины: это сценарий для проверки выводов «изменение есть, причина не установлена»).

    python scripts/generate_sales_demo.py
"""
from __future__ import annotations

import random
from pathlib import Path

from openpyxl import Workbook

OUT = Path(__file__).resolve().parent.parent / "data" / "demo" / "sales_2026.xlsx"
MONTHS = [f"2026-0{m}" for m in range(1, 7)]
REGIONS = {"Almaty": 1.0, "Astana": 0.8, "Shymkent": 0.5, "Karaganda": 0.35}
BASE_ORDERS, AVG_CHECK = 5200, 18500
GROWTH = 0.035  # рост заказов от месяца к месяцу
DROP = {("2026-04", "Astana"): 0.62}  # коэффициент выручки в аномальной точке


def build_rows(seed: int = 2026) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index, month in enumerate(MONTHS):
        for region, weight in REGIONS.items():
            orders = round(BASE_ORDERS * weight * (1 + GROWTH) ** index * rng.uniform(0.97, 1.03))
            revenue = round(orders * AVG_CHECK * rng.uniform(0.96, 1.04) * DROP.get((month, region), 1.0))
            rows.append({"Month": month, "Region": region, "Orders": orders, "Revenue_KZT": revenue})
    return rows


def write_xlsx(rows: list[dict], path: Path = OUT) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "sales"
    sheet.append(list(rows[0]))
    for row in rows:
        sheet.append(list(row.values()))
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path


if __name__ == "__main__":
    print(write_xlsx(build_rows()))
