"""Сценарии интерфейса загрузки данных (Streamlit AppTest, без браузера)."""
import pytest
from streamlit.testing.v1 import AppTest

from datastory.config import get_settings
from datastory.storage.store import DatasetStore
from scripts import generate_demo_data as gen
from tests.helpers import make_xlsx

DEMO_XLSX = (gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes()
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _page():
    from datastory.ui.views import analysis

    analysis.render()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    get_settings.cache_clear()
    yield DatasetStore(tmp_path / "workspace")
    get_settings.cache_clear()


def open_page() -> AppTest:
    return AppTest.from_function(_page, default_timeout=60).run()


def upload(at: AppTest, name: str, content: bytes) -> AppTest:
    at.file_uploader[0].upload(name, content)
    return at.run()


def texts(elements) -> str:
    return "\n".join(e.value for e in elements)


def confirm_button(at: AppTest):
    return next(b for b in at.button if "Подтвердить" in b.label)


def test_empty_state(workspace):
    at = open_page()
    assert not at.exception
    assert "Загрузите файл" in texts(at.info)
    assert not at.subheader


def test_full_flow_with_demo_file(workspace):
    at = upload(open_page(), "transactions_2026.xlsx", DEMO_XLSX)
    assert not at.exception
    assert [s.value for s in at.subheader] == [
        "1. Предпросмотр", "2. Профиль датасета", "3. Качество данных", "4. Подтверждение структуры",
    ]
    assert "Проблем не найдено" in texts(at.success)
    assert "Числовые:** Transactions, Successful, Failed, Amount_KZT" in texts(at.markdown)
    assert "Временные:** Month" in texts(at.markdown)
    assert "Категориальные:** Channel" in texts(at.markdown)
    assert workspace.list() == []  # до подтверждения ничего не сохраняется

    at = confirm_button(at).click().run()
    assert not at.exception

    refs = workspace.list()
    assert len(refs) == 1 and refs[0].row_count == 18 and refs[0].filename == "transactions_2026.xlsx"
    assert refs[0].dataset_id in texts(at.success)
    assert "5. Результаты анализа" in [s.value for s in at.subheader]
    stored = workspace.load_profile(refs[0].dataset_id)
    assert stored.detected_date_columns == ["Month"] and stored.description.startswith("Таблица: 18 строк")


def test_custom_description_is_saved(workspace):
    at = upload(open_page(), "transactions_2026.xlsx", DEMO_XLSX)
    at.text_area[0].set_value("Статистика DemoPay за полугодие").run()
    confirm_button(at).click().run()
    ref = workspace.list()[0]
    assert workspace.load_profile(ref.dataset_id).description == "Статистика DemoPay за полугодие"


def test_excel_sheet_selection(workspace):
    book = make_xlsx({"First": [["a", "b"], [1, 2], [3, 4]], "Second": [["x", "y", "z"], [1, 2, 3]]})
    at = upload(open_page(), "book.xlsx", book)
    assert at.selectbox[0].label == "Лист Excel" and at.selectbox[0].options == ["First", "Second"]
    assert "Числовые:** a, b" in texts(at.markdown)

    at.selectbox[0].select("Second").run()
    assert not at.exception
    assert "Числовые:** x, y, z" in texts(at.markdown)
    confirm_button(at).click().run()
    assert workspace.list()[0].sheet_name == "Second"


def test_structure_change_after_confirmation_requires_reconfirmation(workspace):
    at = upload(open_page(), "transactions_2026.xlsx", DEMO_XLSX)
    confirm_button(at).click().run()
    at = upload(at, "other.csv", b"a,b\n1,2\n3,4\n")
    assert "Подтвердите структуру ещё раз" in texts(at.info)
    assert not at.exception


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("empty.csv", b"", "пустой"),
        ("empty.xlsx", b"", "пустой"),
        ("header_only.csv", b"a,b,c\n", "нет данных"),
        ("broken.xlsx", b"PK\x03\x04not-really-excel", "повреждён"),
        ("fake.csv", b"\x00\x01\x02\x00" * 100, "бинарные"),
        ("ragged.csv", b"a,b\n1,2\n1,2,3,4\n", "разобрать"),
    ],
)
def test_file_errors_are_shown_in_plain_language(workspace, name, content, expected):
    at = upload(open_page(), name, content)
    assert not at.exception  # приложение не падает
    assert expected in texts(at.error)
    assert not at.subheader  # дальнейшие шаги не показываются


@pytest.mark.parametrize("name", ["notes.txt", "old.xls", "report.pdf", "data.json"])
def test_unsupported_formats_get_plain_message(workspace, name):
    at = upload(open_page(), name, b"some content")
    assert not at.exception
    assert "не поддерживается" in texts(at.error) and "XLSX или CSV" in texts(at.error)
    assert not at.subheader


def test_image_upload_explains_it_is_not_available_yet(workspace):
    screenshot = (gen.DEFAULT_OUT / "transactions_screenshot.png").read_bytes()
    at = upload(open_page(), "transactions_screenshot.png", screenshot)
    assert not at.exception
    assert "изображениях" in texts(at.warning)
    assert not at.error


def test_corrupt_image_does_not_crash_page(workspace):
    at = upload(open_page(), "shot.png", PNG_HEADER)
    assert not at.exception
    assert "изображениях" in texts(at.warning) and "повреждён" in texts(at.error)


def test_quality_problems_are_displayed(workspace):
    csv = "id;amount;note\n" + "\n".join(f"{i};{i * 10};ok" for i in range(1, 10)) + "\n10;abc;ok\n1;10;ok\n"
    at = upload(open_page(), "dirty.csv", csv.encode("utf-8"))
    assert not at.exception
    warnings = texts(at.warning)
    assert "не распознаны как числа" in warnings and "abc" in warnings
    assert "повторяющихся строк" in warnings
    assert "Проблем не найдено" not in texts(at.success)


def _page_with_edited_types():
    """Страница, у которой редактор типов «возвращает» правки пользователя (canvas-таблицу AppTest не умеет)."""
    import streamlit as st

    original = st.data_editor

    def edited(data, *args, **kwargs):
        table = data.copy()
        table.loc[table["Колонка"] == "Channel", "Тип"] = "Текст"
        table.loc[table["Колонка"] == "Transactions", "Тип"] = "Категория"
        table.loc[table["Колонка"] == "Month", "Персональные данные"] = True
        return table

    st.data_editor = edited
    try:
        from datastory.ui.views import analysis

        analysis.render()
    finally:
        st.data_editor = original


def test_manual_type_overrides_are_applied_and_saved(workspace):
    at = AppTest.from_function(_page_with_edited_types, default_timeout=60).run()
    at = upload(at, "transactions_2026.xlsx", DEMO_XLSX)
    assert not at.exception
    markdown = texts(at.markdown)
    assert "Числовые:** Successful, Failed, Amount_KZT" in markdown  # Transactions больше не число
    assert "Категориальные:** Transactions" in markdown  # Channel стал текстом, Transactions — категорией
    assert "Числовые:** Transactions" not in markdown

    at = confirm_button(at).click().run()
    assert not at.exception
    ref = workspace.list()[0]
    profile, frame = workspace.load_profile(ref.dataset_id), workspace.load_dataframe(ref.dataset_id)

    assert profile.column("Channel").kind.value == "text" and profile.column("Channel").is_overridden
    assert profile.column("Transactions").kind.value == "categorical"
    assert profile.column("Transactions").detected_kind.value == "numeric"
    assert profile.column("Month").is_sensitive and profile.column("Month").sensitive_reason == "отмечено пользователем"
    assert profile.detected_numeric_columns == ["Successful", "Failed", "Amount_KZT"]
    assert frame["Transactions"].iloc[0] == "177841"  # данные приведены к выбранному типу и сохранены
