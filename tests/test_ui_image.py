"""Загрузка изображения с таблицей на странице «Анализ данных»: Vision → правка → подтверждение → общий конвейер."""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from datastory.file_processing.image_loader import TableExtraction
from datastory.storage.store import DatasetStore
from datastory.ui.mcp import get_analytics_client
from scripts import generate_demo_data as gen
from tests.vision_fakes import FakeVision, failing_vision

SCREENSHOT = (gen.DEFAULT_OUT / "transactions_screenshot.png").read_bytes()
SHOT_ROWS = [r for r in gen.build_rows() if r["Month"] in gen.SCREENSHOT_MONTHS]
CONFIRM_TABLE, CONFIRM_STRUCTURE = "Подтвердить распознанную таблицу", "Подтвердить структуру и сохранить датасет"


def _page():
    from datastory.ui.views import analysis

    analysis.render()


def texts(elements) -> str:
    return "\n".join(e.value for e in elements)


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def open_with(monkeypatch, vision, name="transactions_screenshot.png", data=SCREENSHOT) -> AppTest:
    monkeypatch.setattr("datastory.ui.views.image_table._vision", lambda: vision)
    at = AppTest.from_function(_page, default_timeout=90).run()
    at.file_uploader[0].upload(name, data)
    return at.run()


def confirmed_table(monkeypatch, vision=None) -> tuple[AppTest, FakeVision]:
    vision = vision or FakeVision()
    return button(open_with(monkeypatch, vision), CONFIRM_TABLE).click().run(), vision


# ================================================================== распознавание и подтверждение
def test_screenshot_is_recognized_and_shown_for_review(monkeypatch):
    vision = FakeVision()
    at = open_with(monkeypatch, vision)
    assert not at.exception and not at.error
    assert len(vision.calls) == 1 and vision.calls[0]["mime"] == "image/png"
    assert "Таблица распознана моделью fake-vision: 6 строк, 6 колонок" in texts(at.caption)
    assert "Проверьте таблицу и подтвердите её" in texts(at.info)
    assert not [s for s in at.subheader if s.value.startswith("2.")]  # профиль появится только после подтверждения


def test_reruns_do_not_send_the_image_to_the_model_again(monkeypatch):
    vision = FakeVision()
    at = open_with(monkeypatch, vision)
    for _ in range(3):
        at = at.run()
        assert not at.exception
    assert len(vision.calls) == 1


def test_confirmed_table_goes_through_the_common_profiler(monkeypatch):
    at, _ = confirmed_table(monkeypatch)
    assert not at.exception and not at.error
    assert [s.value for s in at.subheader if s.value[0].isdigit()][:3] == ["1. Предпросмотр", "2. Профиль датасета", "3. Качество данных"]
    body = texts(at.markdown) + texts(at.caption)
    assert "Формат: ИЗОБРАЖЕНИЕ (VISION)" in body
    assert "Числовые:** Transactions, Successful, Failed, Amount_KZT" in body
    assert "Временные:** Month" in body and "Категориальные:** Channel" in body


def test_recognized_screenshot_flows_into_the_langgraph_dashboard(monkeypatch):
    at, _ = confirmed_table(monkeypatch)
    at = button(at, CONFIRM_STRUCTURE).click().run()
    ref = DatasetStore().list()[0]
    assert ref.filename == "transactions_screenshot.png" and ref.row_count == 6

    at = button(at, "Запустить анализ").click().run()
    for box in at.checkbox:
        if box.label.startswith("Подтверждаю правило расчёта"):
            box.check()
    at = button(at, "Подтвердить и построить дашборд").click().run()
    assert not at.exception and not at.error and len(at.get("plotly_chart")) == 4
    march = [r for r in SHOT_ROWS if r["Month"] == "2026-03"]
    rate = sum(r["Successful"] for r in march) / sum(r["Transactions"] for r in march) * 100
    assert f"{rate:.2f}".replace(".", ",") + "%" in texts(at.markdown) + texts(at.info) + texts(at.warning)
    assert get_analytics_client().calls.count("calculate_metrics") == 4  # те же MCP-инструменты, что и для XLSX


def test_user_corrections_are_used_instead_of_the_recognized_values(monkeypatch):
    original = st.data_editor

    def edited(data, *args, **kwargs):
        if str(kwargs.get("key", "")).startswith("vision-body"):
            data = data.copy()
            data.loc[0, "c1"] = "186 174"  # ошибка распознавания исправлена вручную
        return original(data, *args, **kwargs)

    monkeypatch.setattr(st, "data_editor", edited)
    at, _ = confirmed_table(monkeypatch)
    at = button(at, CONFIRM_STRUCTURE).click().run()
    ref = DatasetStore().list()[0]
    assert DatasetStore().load_dataframe(ref.dataset_id)["Transactions"].iloc[0] == 186174


def test_edited_headers_are_applied(monkeypatch):
    original = st.data_editor

    def edited(data, *args, **kwargs):
        if str(kwargs.get("key", "")).startswith("vision-headers"):
            data = data.copy()
            data.loc[data["Заголовок"] == "Amount_KZT", "Заголовок"] = "Amount"
        return original(data, *args, **kwargs)

    monkeypatch.setattr(st, "data_editor", edited)
    at, _ = confirmed_table(monkeypatch)
    assert "Числовые:** Transactions, Successful, Failed, Amount" in texts(at.markdown)


# ================================================================== ошибки
def test_image_without_a_table_shows_a_clear_error_and_no_analysis(monkeypatch):
    answer = TableExtraction(contains_table=False, columns=[], rows=[], unreadable_cells=[], comment="На изображении график.")
    at = open_with(monkeypatch, FakeVision(answer))
    assert not at.exception
    assert "На изображении не найдена таблица" in texts(at.error) and "график" in texts(at.error)
    assert not [s for s in at.subheader if s.value.startswith("2.")] and not at.get("plotly_chart")


def test_model_failure_can_be_retried(monkeypatch):
    vision = failing_vision()
    at = open_with(monkeypatch, vision)
    assert "Не удалось распознать таблицу: Превышен лимит" in texts(at.error)
    vision.answer = FakeVision().answer  # лимит снят
    at = button(at, "Повторить распознавание").click().run()
    assert not at.error and "Таблица распознана" in texts(at.caption) and len(vision.calls) == 2


def test_missing_key_is_explained(monkeypatch):
    at = open_with(monkeypatch, None)
    assert not at.exception and "OPENAI_API_KEY" in texts(at.error)


def test_corrupt_image_does_not_crash_the_page(monkeypatch):
    vision = FakeVision()
    at = open_with(monkeypatch, vision, "shot.png", b"\x89PNG\r\n\x1a\nbroken")
    assert not at.exception and "повреждён" in texts(at.error) and vision.calls == []
