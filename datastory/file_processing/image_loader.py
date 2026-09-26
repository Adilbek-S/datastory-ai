"""Таблицы на изображениях (PNG/JPG): Vision-модель → структурный ответ → pandas DataFrame.

Дальше DataFrame идёт тем же путём, что таблица из XLSX/CSV: Dataset Profiler, хранилище, LangGraph. Отдельной
аналитической логики для изображений нет. Модель возвращает значения ячеек как текст «как напечатано»; типы (числа,
даты) определяет обычный профайлер, а исправить ошибки распознавания пользователь может до подтверждения.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from datastory.errors import CorruptFileError, DataLoadError, EmptyFileError, LLMError, LLMUnavailableError, NoDataError, UnsupportedFileError
from datastory.file_processing.loader import IMAGE_EXTENSIONS, clean_frame
from datastory.llm.client import VisionLLM
from datastory.observability import span

logger = logging.getLogger("datastory.vision")

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_SIDE = 32
MAX_SIDE = 2048  # больше модели не нужно: изображение уменьшается перед отправкой
MAX_ROWS = 500
MAX_COLUMNS = 40
MIME = {"PNG": "image/png", "JPEG": "image/jpeg"}

SYSTEM = (
    "Ты распознаёшь таблицу на изображении и возвращаешь её в структурированном виде. "
    "columns — заголовки колонок слева направо в точности как на изображении. "
    "rows — строки таблицы сверху вниз; значения ячеек — строки, записанные ровно так, как напечатано: не округляй, "
    "не исправляй, не переводи, не вычисляй, не добавляй итогов и строк, которых нет на изображении. "
    "В каждой строке ровно столько значений, сколько колонок. Нечитаемую ячейку оставь пустой строкой и опиши её в unreadable_cells. "
    "Если на изображении нет таблицы (фото, график, текст), верни contains_table=false и пустые columns и rows."
)
USER = "Распознай таблицу на изображении."


class TableNotFoundError(NoDataError):
    code = "table_not_found"


class VisionUnavailableError(DataLoadError):
    code = "vision_unavailable"


class TableExtraction(BaseModel):
    """Структурный ответ Vision-модели."""

    contains_table: bool = Field(description="Есть ли на изображении таблица")
    columns: list[str] = Field(description="Заголовки колонок слева направо")
    rows: list[list[str]] = Field(description="Строки таблицы; значения ячеек как текст, точно как на изображении")
    unreadable_cells: list[str] = Field(description="Описание нечитаемых ячеек (например, «строка 3, колонка Amount»); пустой список, если всё читается")
    comment: str = Field(description="Одно предложение о таблице или о причине, почему таблицы нет")


@dataclass
class RecognizedTable:
    frame: pd.DataFrame  # значения — текст; типы определяет профайлер после подтверждения
    warnings: list[str] = field(default_factory=list)
    comment: str = ""
    model: str = ""


# --------------------------------------------------------------------------- изображение
def prepare_image(data: bytes, filename: str = "image.png") -> tuple[bytes, str]:
    """Проверяет изображение и приводит к PNG/JPEG допустимого размера. Возвращает (байты, MIME-тип)."""
    if not data:
        raise EmptyFileError("Файл изображения пустой.")
    if len(data) > MAX_IMAGE_BYTES:
        raise UnsupportedFileError(f"Изображение слишком большое ({len(data) // (1024 * 1024)} МБ): допустимо не более {MAX_IMAGE_BYTES // (1024 * 1024)} МБ.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise CorruptFileError("Не удалось открыть изображение: файл повреждён или не является PNG/JPG.") from None
    if image.format not in MIME:
        raise UnsupportedFileError(f"Формат изображения ({image.format or 'неизвестный'}) не поддерживается. Загрузите PNG или JPG.")
    if min(image.size) < MIN_SIDE:
        raise UnsupportedFileError(f"Изображение слишком маленькое ({image.width}×{image.height}): таблицу на нём не разобрать.")

    fmt = image.format
    image = ImageOps.exif_transpose(image)  # фото с поворотом по EXIF
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue(), MIME[fmt]


# --------------------------------------------------------------------------- ответ модели → DataFrame
def frame_from_extraction(extraction: TableExtraction) -> tuple[pd.DataFrame, list[str]]:
    """Структурный ответ → DataFrame. Возвращает (таблица, предупреждения). Бросает TableNotFoundError / NoDataError."""
    if not extraction.contains_table:
        reason = f" {extraction.comment.strip()}" if extraction.comment.strip() else ""
        raise TableNotFoundError(f"На изображении не найдена таблица.{reason} Загрузите изображение, на котором таблица видна целиком.")
    if not extraction.columns:
        raise TableNotFoundError("Не удалось прочитать заголовки таблицы на изображении. Загрузите более чёткое изображение.")
    if len(extraction.columns) > MAX_COLUMNS or len(extraction.rows) > MAX_ROWS:
        raise NoDataError(f"Таблица слишком большая для распознавания по изображению (не более {MAX_COLUMNS} колонок и {MAX_ROWS} строк).")

    warnings: list[str] = []
    names: list[str] = []
    for index, raw in enumerate(extraction.columns, 1):
        name = (raw or "").strip() or f"Колонка {index}"
        if not (raw or "").strip():
            warnings.append(f"Заголовок колонки {index} не прочитан: назначено имя «{name}».")
        base, suffix = name, 2
        while name in names:
            name, suffix = f"{base}_{suffix}", suffix + 1
        if name != base:
            warnings.append(f"Заголовок «{base}» повторяется: колонка переименована в «{name}».")
        names.append(name)

    width, rows = len(names), []
    for number, row in enumerate(extraction.rows, 1):
        cells = [(c or "").strip() for c in row]
        if len(cells) != width:
            warnings.append(f"В строке {number} значений {len(cells)}, а колонок {width}: строка выровнена, проверьте её.")
            cells = (cells + [""] * width)[:width]
        rows.append([c or None for c in cells])
    if not any(any(c is not None for c in row) for row in rows):
        raise NoDataError("На изображении найдены только заголовки таблицы, строк с данными нет.")
    if extraction.unreadable_cells:
        warnings.append("Модель не смогла прочитать ячейки: " + "; ".join(extraction.unreadable_cells) + ". Заполните их вручную.")
    return clean_frame(pd.DataFrame(rows, columns=names)), warnings


def recognize_table(data: bytes, filename: str, vision: VisionLLM | None) -> RecognizedTable:
    """Изображение → Vision-модель → DataFrame. Все сбои превращаются в DataLoadError с понятным сообщением.

    В LangSmith попадает только span со сводкой (имя файла, размер, число строк): само изображение не отправляется.
    """
    with span("vision.recognize_table", tags=("vision",), inputs={"filename": filename, "image_bytes": len(data)}) as sp:
        table = _recognize_table(data, filename, vision)
        sp.outputs({"rows": len(table.frame), "columns": len(table.frame.columns), "warnings": len(table.warnings), "model": table.model})
        return table


def _recognize_table(data: bytes, filename: str, vision: VisionLLM | None) -> RecognizedTable:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if f".{ext}" not in IMAGE_EXTENSIONS:
        raise UnsupportedFileError("Ожидалось изображение PNG или JPG.")
    image, mime = prepare_image(data, filename)
    if vision is None:
        raise VisionUnavailableError(
            "Для распознавания таблицы на изображении нужна Vision-модель OpenAI: задайте OPENAI_API_KEY в файле .env "
            "или загрузите таблицу в формате XLSX или CSV."
        )
    try:
        extraction = vision.generate_from_image(TableExtraction, system=SYSTEM, user=USER, image=image, mime=mime)
    except LLMUnavailableError as exc:
        raise VisionUnavailableError(exc.user_message) from None
    except LLMError as exc:
        raise DataLoadError(f"Не удалось распознать таблицу: {exc.user_message}") from None
    frame, warnings = frame_from_extraction(extraction)
    return RecognizedTable(frame=frame, warnings=warnings, comment=extraction.comment, model=getattr(vision, "name", ""))
