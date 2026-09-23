"""Загрузка табличных файлов (CSV / Excel) и извлечение текста из PDF."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pymupdf

from datastory.models import KnowledgeChunk

TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class UnsupportedFileError(ValueError):
    """Формат файла не поддерживается или ещё не реализован."""


def file_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in TABLE_EXTENSIONS:
        return "table"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext == ".pdf":
        return "pdf"
    return "unknown"


def load_table(data: bytes, filename: str) -> pd.DataFrame:
    """Читает CSV или Excel в DataFrame. Изображения (OCR) появятся на следующем этапе."""
    kind = file_kind(filename)
    if kind == "image":
        raise UnsupportedFileError(
            "Распознавание таблиц на изображениях будет добавлено на следующем этапе."
        )
    if kind != "table":
        raise UnsupportedFileError(f"Неподдерживаемый формат файла: {filename}")

    if Path(filename).suffix.lower() == ".csv":
        return _read_csv(data)
    return pd.read_excel(io.BytesIO(data))


def _read_csv(data: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise UnsupportedFileError(f"Не удалось прочитать CSV: {last_error}")


def extract_pdf_chunks(data: bytes, source: str) -> list[KnowledgeChunk]:
    """Извлекает текст PDF постранично (PyMuPDF). Пустые страницы пропускаются."""
    chunks: list[KnowledgeChunk] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                chunks.append(KnowledgeChunk(source=source, page=number, text=text))
    return chunks
