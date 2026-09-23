"""Загрузка структурированных данных (CSV / XLSX) и извлечение текста из PDF.

Все проблемы с файлом превращаются в DataLoadError с понятным сообщением на русском.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pymupdf
from openpyxl import load_workbook

from datastory.errors import (
    CorruptFileError,
    DataLoadError,
    EmptyFileError,
    NoDataError,
    UnsupportedFileError,
)
from datastory.models import KnowledgeChunk

TABLE_EXTENSIONS = {".csv", ".xlsx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CSV_ENCODINGS = ("utf-8-sig", "cp1251")
CSV_DELIMITERS = (",", ";", "\t", "|")

__all__ = [
    "DataLoadError",
    "UnsupportedFileError",
    "detect_format",
    "extract_pdf_chunks",
    "file_kind",
    "list_sheets",
    "load_table",
    "read_table",
]


def file_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in TABLE_EXTENSIONS:
        return "table"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext == ".pdf":
        return "pdf"
    return "unknown"


def detect_format(filename: str, data: bytes) -> str:
    """Возвращает 'csv' или 'xlsx'. Проверяет расширение, пустоту и «магические» байты."""
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        raise UnsupportedFileError(
            "Распознавание таблиц на изображениях будет добавлено на следующем этапе. "
            "Сейчас загрузите файл XLSX или CSV."
        )
    if ext not in TABLE_EXTENSIONS:
        shown = ext or "без расширения"
        raise UnsupportedFileError(
            f"Формат файла ({shown}) не поддерживается. Загрузите файл XLSX или CSV."
        )
    if not data or not data.strip():
        raise EmptyFileError("Файл пустой: в нём нет ни данных, ни заголовков.")

    if ext == ".xlsx":
        if not data.startswith(b"PK"):
            raise CorruptFileError(
                "Файл повреждён или не является книгой Excel (XLSX). "
                "Пересохраните его в Excel и загрузите снова."
            )
        return "xlsx"
    is_utf16 = data.startswith((b"\xff\xfe", b"\xfe\xff"))
    if data.startswith(b"PK") or (b"\x00" in data[:4096] and not is_utf16):
        raise CorruptFileError(
            "Файл с расширением CSV содержит бинарные данные. Убедитесь, что это текстовый CSV."
        )
    return "csv"


def list_sheets(data: bytes) -> list[str]:
    """Названия листов XLSX-книги."""
    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True)
        try:
            return list(workbook.sheetnames)
        finally:
            workbook.close()
    except Exception as exc:  # noqa: BLE001 — openpyxl бросает разные типы на повреждённых файлах
        raise CorruptFileError(f"Не удалось открыть книгу Excel: файл повреждён ({type(exc).__name__}).") from exc


def read_table(data: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Читает CSV или лист XLSX в DataFrame и убирает пустые строки/колонки."""
    fmt = detect_format(filename, data)
    frame = _read_xlsx(data, sheet_name) if fmt == "xlsx" else _read_csv(data)
    return _clean_frame(frame)


def load_table(data: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Совместимый алиас read_table."""
    return read_table(data, filename, sheet_name)


# --------------------------------------------------------------------------- XLSX
def _read_xlsx(data: bytes, sheet_name: str | None) -> pd.DataFrame:
    sheets = list_sheets(data)
    if not sheets:
        raise NoDataError("В книге Excel нет листов.")
    target = sheet_name if sheet_name is not None else sheets[0]
    if target not in sheets:
        raise NoDataError(f"Лист «{target}» не найден. Доступные листы: {', '.join(sheets)}.")
    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=target, engine="openpyxl")
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise CorruptFileError(f"Не удалось прочитать лист «{target}»: файл повреждён.") from exc


# --------------------------------------------------------------------------- CSV
def _decode(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise CorruptFileError("Не удалось определить кодировку CSV-файла.") from exc
    for encoding in CSV_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CorruptFileError("Не удалось определить кодировку CSV-файла. Сохраните его в UTF-8.")


def _detect_delimiter(text: str) -> str:
    header = next((line for line in text.splitlines() if line.strip()), "")
    counts = {d: header.count(d) for d in CSV_DELIMITERS}
    best = max(CSV_DELIMITERS, key=lambda d: counts[d])
    return best if counts[best] > 0 else ","


def _read_csv(data: bytes) -> pd.DataFrame:
    text = _decode(data)
    try:
        return pd.read_csv(io.StringIO(text), sep=_detect_delimiter(text), skip_blank_lines=True)
    except pd.errors.EmptyDataError as exc:
        raise NoDataError("В CSV-файле нет данных.") from exc
    except pd.errors.ParserError as exc:
        raise CorruptFileError(f"Не удалось разобрать CSV: {exc}") from exc


# --------------------------------------------------------------------------- общее
def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    names = [str(c).strip() for c in frame.columns]
    frame.columns = [n if n else f"Unnamed: {i}" for i, n in enumerate(names)]

    frame = frame.dropna(how="all")
    empty_unnamed = [c for c in frame.columns if str(c).startswith("Unnamed") and frame[c].isna().all()]
    frame = frame.drop(columns=empty_unnamed)

    if frame.shape[1] == 0 or frame.shape[0] == 0:
        raise NoDataError(
            "В файле нет данных: найдены только заголовки или пустые строки."
            if frame.shape[1]
            else "В файле нет данных: таблица пуста."
        )
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- PDF
def extract_pdf_chunks(data: bytes, source: str) -> list[KnowledgeChunk]:
    """Извлекает текст PDF постранично (PyMuPDF). Пустые страницы пропускаются."""
    chunks: list[KnowledgeChunk] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                chunks.append(KnowledgeChunk(source=source, page=number, text=text))
    return chunks
