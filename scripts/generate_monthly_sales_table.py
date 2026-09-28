"""Демо-изображение для сценария C аудита: таблица месячных итогов продаж (Month, Orders, Revenue_KZT) как PNG.

Значения — суммы по месяцам из data/demo/sales_2026.xlsx, поэтому ответ Vision-модели можно сверить с эталоном.
Данные вымышлены. Запуск:  python scripts/generate_monthly_sales_table.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "data" / "demo" / "sales_2026.xlsx"
TARGET = PROJECT_ROOT / "data" / "demo" / "monthly_sales_table.png"
COLUMNS = ["Month", "Orders", "Revenue_KZT"]


def monthly_rows() -> list[list[str]]:
    frame = pd.read_excel(SOURCE, sheet_name="sales")
    totals = frame.groupby("Month", sort=True)[["Orders", "Revenue_KZT"]].sum()
    return [[str(month), str(int(row.Orders)), str(int(round(row.Revenue_KZT)))] for month, row in totals.iterrows()]


def write_png(rows: list[list[str]], path: Path) -> None:
    reg_buf, bold_buf = pymupdf.Font("helv").buffer, pymupdf.Font("hebo").buffer
    font, bold = ImageFont.truetype(io.BytesIO(reg_buf), 32), ImageFont.truetype(io.BytesIO(bold_buf), 32)
    table = [COLUMNS, *rows]
    pad, row_h, margin = 24, 64, 32
    widths = [max(int((bold if n == 0 else font).getlength(line[i])) for n, line in enumerate(table)) + 2 * pad for i in range(len(COLUMNS))]
    width, height = sum(widths) + 2 * margin, row_h * len(table) + 2 * margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for r, line in enumerate(table):
        y0, y1 = margin + r * row_h, margin + (r + 1) * row_h
        if r == 0:
            draw.rectangle([margin, y0, width - margin, y1], fill="#E5E7EB")
        x = margin
        for i, text in enumerate(line):
            f = bold if r == 0 else font
            tx = x + pad if i == 0 else x + widths[i] - pad - f.getlength(text)
            draw.text((tx, (y0 + y1) / 2), text, font=f, fill="#111827", anchor="lm")
            x += widths[i]
    for r in range(len(table) + 1):
        y = margin + r * row_h
        draw.line([margin, y, width - margin, y], fill="#374151", width=2)
    x = margin
    for w in [*widths, 0][: len(widths) + 1]:
        draw.line([x, margin, x, height - margin], fill="#374151", width=2)
        x += w
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


if __name__ == "__main__":
    rows = monthly_rows()
    write_png(rows, TARGET)
    print(f"{TARGET}: {len(rows)} строк")
    for row in rows:
        print("  ", *row)
