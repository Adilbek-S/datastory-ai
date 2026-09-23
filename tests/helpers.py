"""Общие вспомогательные функции тестов."""
import io

from openpyxl import Workbook


def make_xlsx(sheets: dict[str, list[list]]) -> bytes:
    """Книга Excel в памяти: {имя листа: строки}."""
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
