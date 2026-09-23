"""Общие вспомогательные функции тестов."""
import io

import pymupdf
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


def make_pdf(blocks: list[tuple[str, str]], *, page_footer: bool = False) -> bytes:
    """PDF с кириллицей (встроенный шрифт).

    blocks: ("h", заголовок) | ("p", абзац) | ("break", "") — принудительная новая страница.
    """
    regular = pymupdf.Font(fontbuffer=pymupdf.Font("helv").buffer)
    bold = pymupdf.Font(fontbuffer=pymupdf.Font("hebo").buffer)
    doc = pymupdf.open()
    state = {"page": None, "y": 60.0, "number": 0}

    def footer() -> None:
        writer = pymupdf.TextWriter(state["page"].rect)
        writer.append((280, 810), f"стр. {state['number']}", font=regular, fontsize=9)
        writer.write_text(state["page"])

    def new_page() -> None:
        if page_footer and state["page"] is not None:
            footer()
        state["page"] = doc.new_page(width=595, height=842)
        state["y"] = 60.0
        state["number"] += 1

    def put(text: str, font, size: float, gap_after: float) -> None:
        line, rows = "", []
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if font.text_length(candidate, fontsize=size) <= 480:
                line = candidate
            else:
                rows.append(line)
                line = word
        rows.append(line)
        for row in rows:
            if state["y"] + size * 1.4 > 780:
                new_page()
            writer = pymupdf.TextWriter(state["page"].rect)
            writer.append((56, state["y"] + size), row, font=font, fontsize=size)
            writer.write_text(state["page"])
            state["y"] += size * 1.4
        state["y"] += gap_after

    new_page()
    for kind, text in blocks:
        if kind == "h":
            state["y"] += 8
            put(text, bold, 16, 4)
        elif kind == "p":
            put(text, regular, 10.5, 8)
        elif kind == "break":
            new_page()
    if page_footer:
        footer()
    return doc.tobytes()
