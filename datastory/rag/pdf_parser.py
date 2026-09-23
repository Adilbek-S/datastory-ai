"""Разбор PDF (PyMuPDF): текст, номера страниц, заголовки разделов и chunking по смыслу.

Заголовок определяется по размеру шрифта (заметно крупнее основного текста) или по жирному
начертанию с нумерацией («4. Формула…»). Раздел режется на фрагменты по абзацам; длинные
абзацы делятся по предложениям, соседние фрагменты одного раздела перекрываются.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

from datastory.rag.models import DocumentChunk, DocumentSection, Paragraph, ParsedDocument

HEADING_SIZE_RATIO = 1.2
MAX_HEADING_LEN = 120
NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*[.)]|[IVXLC]+\.)\s+\S")
PAGE_NUMBER_LINE = re.compile(r"^\s*(стр\.?|page|—|-)?\s*\d{1,4}\s*(—|-)?\s*$", re.IGNORECASE)
BULLET = re.compile(r"^\s*([•\-–*·]|\d+[.)])\s+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
PARAGRAPH_GAP_RATIO = 0.45  # разрыв между строками > 0.45 размера шрифта — новый абзац

DEFAULT_MAX_CHARS = 900
DEFAULT_OVERLAP = 120


class PdfError(ValueError):
    """PDF нельзя прочитать; user_message показывается пользователю."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class _Line:
    page: int
    text: str
    size: float
    bold: bool
    top: float
    bottom: float


def document_id_for(data: bytes) -> str:
    """Идентификатор по содержимому: один и тот же файл всегда даёт один document_id."""
    return hashlib.sha256(data).hexdigest()[:16]


# --------------------------------------------------------------------------- чтение строк
def _join_spans(spans: list[dict]) -> str:
    """Склеивает спаны строки; пробел ставится, если между ними виден зазор (маркер списка и текст)."""
    text = spans[0]["text"]
    for prev, span in zip(spans, spans[1:]):
        gap = span["bbox"][0] - prev["bbox"][2]
        joiner = " " if gap > 0.1 * span["size"] and not text.endswith(" ") and not span["text"].startswith(" ") else ""
        text += joiner + span["text"]
    return " ".join(text.split())


def _read_lines(doc: pymupdf.Document) -> list[_Line]:
    lines: list[_Line] = []
    for page_number, page in enumerate(doc, start=1):
        for block in page.get_text("dict", sort=True)["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = _join_spans(spans)
                if PAGE_NUMBER_LINE.match(text):
                    continue  # номер страницы, а не содержимое
                lines.append(
                    _Line(
                        page=page_number,
                        text=text,
                        size=max(s["size"] for s in spans),
                        bold=any(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans),
                        top=line["bbox"][1],
                        bottom=line["bbox"][3],
                    )
                )
    return lines


def _body_size(lines: list[_Line]) -> float:
    sizes: Counter[float] = Counter()
    for line in lines:
        sizes[round(line.size, 1)] += len(line.text)
    return sizes.most_common(1)[0][0]


def _is_heading(line: _Line, body: float) -> bool:
    if len(line.text) > MAX_HEADING_LEN:
        return False
    if line.size >= body * HEADING_SIZE_RATIO:
        return True
    return line.bold and bool(NUMBERED_HEADING.match(line.text))


# --------------------------------------------------------------------------- разделы
def parse_pdf(data: bytes, filename: str) -> ParsedDocument:
    """Извлекает текст и структуру. Бросает PdfError с понятным сообщением."""
    if not data:
        raise PdfError("Файл пустой.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — PyMuPDF бросает разные типы на плохих файлах
        raise PdfError("Не удалось открыть файл как PDF: он повреждён или не является PDF.") from exc

    with doc:
        if doc.needs_pass:
            raise PdfError("PDF защищён паролем. Снимите защиту и загрузите файл снова.")
        if doc.page_count == 0:
            raise PdfError("В PDF нет страниц.")
        lines = _read_lines(doc)
        page_count = doc.page_count

    if not lines:
        raise PdfError("В PDF нет текстового слоя (возможно, это скан). Распознавание сканов пока не поддерживается.")

    body = _body_size(lines)
    sections: list[DocumentSection] = []
    current = DocumentSection(title=None)
    paragraph: list[_Line] = []

    def flush_paragraph() -> None:
        if paragraph:
            current.paragraphs.append(Paragraph(page=paragraph[0].page, text=" ".join(l.text for l in paragraph)))
            paragraph.clear()

    def start_section(title: str) -> None:
        nonlocal current
        flush_paragraph()
        if current.title or current.paragraphs:
            sections.append(current)
        current = DocumentSection(title=title)

    previous: _Line | None = None
    for line in lines:
        if _is_heading(line, body):
            if previous is not None and _is_heading(previous, body) and previous.page == line.page \
                    and abs(previous.size - line.size) < 0.1 and current.title and not current.paragraphs:
                current.title = f"{current.title} {line.text}"  # заголовок, перенесённый на вторую строку
            else:
                start_section(line.text)
        else:
            new_paragraph = (
                previous is None
                or previous.page != line.page
                or line.top - previous.bottom > PARAGRAPH_GAP_RATIO * line.size
                or bool(BULLET.match(line.text))
                or _is_heading(previous, body)
            )
            if new_paragraph:
                flush_paragraph()
            paragraph.append(line)
        previous = line
    flush_paragraph()
    if current.title or current.paragraphs:
        sections.append(current)

    return ParsedDocument(
        document_id=document_id_for(data), filename=filename, page_count=page_count, sections=sections
    )


# --------------------------------------------------------------------------- chunking
def _sentences(text: str, max_chars: int) -> list[str]:
    """Предложения абзаца; предложение длиннее max_chars режется по словам."""
    result: list[str] = []
    for sentence in SENTENCE_SPLIT.split(text):
        if len(sentence) <= max_chars:
            result.append(sentence)
            continue
        buffer = ""
        for word in sentence.split():
            if buffer and len(buffer) + 1 + len(word) > max_chars:
                result.append(buffer)
                buffer = word
            else:
                buffer = f"{buffer} {word}".strip()
        if buffer:
            result.append(buffer)
    return result


# Единица chunking: (страница, текст, начинает ли новый абзац). Фрагмент собирается из целых предложений.
_Unit = tuple[int, str, bool]


def _join_units(units: list[_Unit]) -> str:
    text = ""
    for index, (_, piece, new_paragraph) in enumerate(units):
        text += piece if index == 0 else ("\n" if new_paragraph else " ") + piece
    return text


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """Последние предложения предыдущего фрагмента суммарной длиной не более overlap символов."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + len(unit[1]) > overlap:
            break
        tail.insert(0, unit)
        total += len(unit[1]) + 1
    return tail


def chunk_document(
    parsed: ParsedDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    dataset_id: str | None = None,
) -> list[DocumentChunk]:
    """Фрагменты по смысловым разделам; раздел длиннее max_chars делится с небольшим перекрытием."""
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Нужно 0 <= overlap < max_chars.")

    chunks: list[DocumentChunk] = []

    def emit(section: DocumentSection, units: list[_Unit]) -> None:
        index = len(chunks)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{parsed.document_id}-{index:04d}",
                document_id=parsed.document_id,
                filename=parsed.filename,
                chunk_index=index,
                page=units[0][0],
                page_end=units[-1][0],
                section=section.title,
                text=_join_units(units),
                dataset_id=dataset_id,
            )
        )

    for section in parsed.sections:
        units: list[_Unit] = [
            (p.page, sentence, position == 0)
            for p in section.paragraphs
            for position, sentence in enumerate(_sentences(p.text, max_chars))
        ]
        if not units:
            continue
        current: list[_Unit] = []
        size = 0
        fresh = 0  # сколько единиц в current — новые (не перекрытие)
        for unit in units:
            length = len(unit[1]) + 1
            if current and fresh and size + length > max_chars:
                emit(section, current)
                current = _overlap_tail(current, overlap)
                size = sum(len(u[1]) + 1 for u in current)
                fresh = 0
            current.append(unit)
            size += length
            fresh += 1
        if fresh:
            emit(section, current)
    return chunks
