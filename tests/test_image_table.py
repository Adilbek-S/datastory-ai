"""Таблицы на изображениях: Vision → DataFrame → тот же профайлер и LangGraph, что и у XLSX/CSV."""
import io

import pandas as pd
import pytest
from PIL import Image

from datastory.errors import CorruptFileError, DataLoadError, EmptyFileError, NoDataError, UnsupportedFileError
from datastory.file_processing import image_loader
from datastory.file_processing.image_loader import (
    SYSTEM,
    TableExtraction,
    TableNotFoundError,
    VisionUnavailableError,
    frame_from_extraction,
    prepare_image,
    recognize_table,
)
from datastory.models import ColumnKind
from datastory.profiler.profiler import build_profile
from datastory.storage.store import DatasetStore
from scripts import generate_demo_data as gen
from tests.helpers import live_openai
from tests.vision_fakes import COLUMNS, FakeVision, failing_vision, screenshot_extraction
from tests.workflow_helpers import approve, make_runner

SCREENSHOT = (gen.DEFAULT_OUT / "transactions_screenshot.png").read_bytes()
SHOT_ROWS = [r for r in gen.build_rows() if r["Month"] in gen.SCREENSHOT_MONTHS]


def image_bytes(fmt="PNG", size=(300, 200), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, (240, 240, 240)).save(buffer, format=fmt)
    return buffer.getvalue()


def extraction(columns, rows, **kwargs) -> TableExtraction:
    return TableExtraction(
        contains_table=kwargs.pop("contains_table", True), columns=columns, rows=rows,
        unreadable_cells=kwargs.pop("unreadable", []), comment="",
    )


# ================================================================== изображение → модель
def test_the_screenshot_is_sent_to_the_vision_model_and_becomes_a_dataframe():
    vision = FakeVision()
    table = recognize_table(SCREENSHOT, "transactions_screenshot.png", vision)
    call = vision.calls[0]
    assert call["schema"] == "TableExtraction" and call["mime"] == "image/png" and call["image"][:8] == b"\x89PNG\r\n\x1a\n"
    assert "не округляй" in call["system"] and call["system"] == SYSTEM
    assert list(table.frame.columns) == COLUMNS and len(table.frame) == 6
    assert table.frame.iloc[0]["Amount_KZT"] == str(SHOT_ROWS[0]["Amount_KZT"]) and table.model == "fake-vision"


def test_recognized_table_uses_the_common_profiler_and_langgraph(client, workspace, datasets):
    """Отдельной логики для изображений нет: DataFrame → DatasetProfile → тот же граф, что для XLSX."""
    table = recognize_table(SCREENSHOT, "transactions_screenshot.png", FakeVision())
    profile, typed = build_profile(table.frame, "transactions_screenshot.png")
    assert profile.column("Month").kind is ColumnKind.DATETIME and profile.column("Transactions").kind is ColumnKind.NUMERIC
    assert profile.column("Channel").kind is ColumnKind.CATEGORICAL and profile.row_count == 6
    DatasetStore(workspace).save(typed, profile)

    runner = make_runner(client)
    snapshot = runner.start(profile.dataset_id)
    assert [c.analysis for c in snapshot.request.plan.charts] == ["count_dynamics", "volume_dynamics", "success_dynamics", "channel_distribution"]
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    march = [r for r in SHOT_ROWS if r["Month"] == "2026-03"]
    expected = sum(r["Successful"] for r in march) / sum(r["Transactions"] for r in march) * 100
    rate = next(m for m in result.metrics if m.metric == "success_rate")
    assert next(r.value for r in rate.rows if r.group["Month"] == "2026-03") == pytest.approx(expected)


def test_editing_a_recognized_cell_changes_the_data_used_for_analysis():
    table = recognize_table(SCREENSHOT, "shot.png", FakeVision())
    frame = table.frame.copy()
    frame.loc[0, "Transactions"] = "186 174"  # пользователь исправил ошибку распознавания
    profile, typed = build_profile(frame, "shot.png")
    assert typed["Transactions"].iloc[0] == 186174


# ================================================================== ошибки
def test_no_key_gives_a_clear_message():
    with pytest.raises(VisionUnavailableError, match="OPENAI_API_KEY"):
        recognize_table(SCREENSHOT, "shot.png", None)


def test_image_without_a_table_gives_a_clear_error():
    answer = TableExtraction(contains_table=False, columns=[], rows=[], unreadable_cells=[], comment="На изображении фотография кошки.")
    with pytest.raises(TableNotFoundError, match="не найдена таблица.*кошки"):
        recognize_table(SCREENSHOT, "cat.png", FakeVision(answer))


def test_model_failure_is_reported_with_the_reason():
    with pytest.raises(DataLoadError, match="Не удалось распознать таблицу: Превышен лимит"):
        recognize_table(SCREENSHOT, "shot.png", failing_vision())


@pytest.mark.parametrize(
    "data, error, message",
    [
        (b"", EmptyFileError, "пустой"),
        (b"\x89PNG\r\n\x1a\nnot really", CorruptFileError, "повреждён"),
        (b"just text", CorruptFileError, "повреждён"),
        (image_bytes("GIF"), UnsupportedFileError, "не поддерживается"),
        (image_bytes("PNG", (10, 10)), UnsupportedFileError, "слишком маленькое"),
    ],
)
def test_bad_images_are_rejected_before_the_model_is_called(data, error, message):
    vision = FakeVision()
    with pytest.raises(error, match=message):
        recognize_table(data, "x.png", vision)
    assert vision.calls == []  # запрос к модели не тратится


def test_oversized_file_is_rejected(monkeypatch):
    monkeypatch.setattr(image_loader, "MAX_IMAGE_BYTES", 100)
    with pytest.raises(UnsupportedFileError, match="слишком большое"):
        prepare_image(SCREENSHOT)


def test_non_image_extension_is_rejected():
    with pytest.raises(UnsupportedFileError, match="PNG или JPG"):
        recognize_table(SCREENSHOT, "table.xlsx", FakeVision())


def test_large_images_are_downscaled_and_jpeg_stays_jpeg():
    big, mime = prepare_image(image_bytes("PNG", (4000, 1000)))
    assert mime == "image/png" and max(Image.open(io.BytesIO(big)).size) == image_loader.MAX_SIDE
    jpg, mime = prepare_image(image_bytes("JPEG", (600, 300)))
    assert mime == "image/jpeg" and Image.open(io.BytesIO(jpg)).format == "JPEG"
    rgba, _ = prepare_image(image_bytes("PNG", (200, 100), "RGBA"))
    assert Image.open(io.BytesIO(rgba)).size == (200, 100)


# ================================================================== ответ модели → таблица
def test_ragged_rows_are_aligned_with_a_warning():
    frame, warnings = frame_from_extraction(extraction(["a", "b", "c"], [["1", "2"], ["3", "4", "5", "6"], ["7", "8", "9"]]))
    assert frame.shape == (3, 3) and pd.isna(frame.iloc[0]["c"]) and frame.iloc[1]["c"] == "5"
    assert len(warnings) == 2 and "строке 1" in warnings[0] and "строке 2" in warnings[1]


def test_blank_and_duplicate_headers_get_names():
    frame, warnings = frame_from_extraction(extraction(["Сумма", "", "Сумма"], [["1", "2", "3"]]))
    assert list(frame.columns) == ["Сумма", "Колонка 2", "Сумма_2"] and len(warnings) == 2


def test_unreadable_cells_are_reported_and_left_empty():
    frame, warnings = frame_from_extraction(extraction(["a", "b"], [["1", ""], ["3", "4"]], unreadable=["строка 1, колонка b"]))
    assert pd.isna(frame.iloc[0]["b"]) and any("строка 1, колонка b" in w for w in warnings)


@pytest.mark.parametrize(
    "answer, message",
    [
        (extraction([], [["1"]]), "заголовки"),
        (extraction(["a", "b"], []), "только заголовки"),
        (extraction(["a", "b"], [["", ""], ["", ""]]), "только заголовки"),
        (extraction([f"c{i}" for i in range(50)], [["1"] * 50]), "слишком большая"),
    ],
)
def test_unusable_extractions_are_rejected(answer, message):
    with pytest.raises(NoDataError, match=message):
        frame_from_extraction(answer)


# ================================================================== настоящая Vision-модель (тратит запросы OpenAI)
@pytest.mark.live
def test_live_vision_reads_the_demo_screenshot():
    table = recognize_table(SCREENSHOT, "transactions_screenshot.png", live_openai())
    expected = screenshot_extraction()
    assert list(table.frame.columns) == expected.columns
    assert table.frame.fillna("").values.tolist() == expected.rows


@pytest.mark.live
def test_live_vision_reports_an_image_without_a_table():
    with pytest.raises(TableNotFoundError):
        recognize_table(image_bytes("PNG", (400, 300)), "blank.png", live_openai())
