"""Интерфейс «База знаний» (Streamlit AppTest, офлайн-эмбеддинги из conftest) и автоиндексация датасета."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from datastory.file_processing.loader import read_table
from datastory.profiler.profiler import build_profile
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.storage.store import DatasetStore
from scripts import generate_demo_data as gen

DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
DEMO_XLSX = (gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes()


def _knowledge_page():
    from datastory.ui.views import knowledge

    knowledge.render()


def _analysis_page():
    from datastory.ui.views import analysis

    analysis.render()


def open_page(section: str | None = None) -> AppTest:
    at = AppTest.from_function(_knowledge_page, default_timeout=60).run()
    if section:
        at = at.radio(key="kb_section").set_value(section).run()
    return at


def search_mode(at: AppTest, mode: str) -> AppTest:
    return next(r for r in at.radio if r.label == "Что искать").set_value(mode).run()


def texts(elements) -> str:
    return "\n".join(e.value for e in elements)


def button(at: AppTest, part: str):
    return next(b for b in at.button if part in b.label)



def saved_demo_profile():
    """Подтверждённый датасет в рабочем хранилище (как после шага «Подтверждение»)."""
    raw = read_table(DEMO_XLSX, "transactions_2026.xlsx", "transactions")
    profile, typed = build_profile(raw, "transactions_2026.xlsx", "transactions")
    DatasetStore().save(typed, profile)
    return profile


# ------------------------------------------------------------------ страница
def test_empty_page_and_offline_warning():
    at = open_page()
    assert not at.exception
    assert "офлайн-режим" in texts(at.warning)  # ключ OpenAI в тестах не задан
    assert "Документов пока нет" in texts(at.info)
    assert at.radio(key="kb_section").options == ["Документы", "Проверка поиска", "Датасеты"]


def test_demo_document_can_be_indexed_once():
    at = open_page()
    at = button(at, "демо-документ").click().run()
    assert not at.exception
    assert "проиндексирован: 10 фрагментов" in texts(at.success)
    assert KnowledgeBase().stats() == {**KnowledgeBase().stats(), "documents": 1, "document_chunks": 10}

    (table,) = [d for d in at.dataframe if "Файл" in d.value.columns]
    row = table.value.iloc[0]
    assert row["Файл"] == "business_metrics.pdf" and row["Фрагментов"] == 10 and row["Страниц"] == 2

    at = button(at, "демо-документ").click().run()  # повторно: защита от дублей
    assert "уже в базе знаний" in texts(at.info)
    assert KnowledgeBase().stats()["document_chunks"] == 10


def test_uploaded_pdf_is_previewed_and_indexed():
    at = open_page()
    at.file_uploader[0].upload("my_metrics.pdf", DEMO_PDF)
    at = at.run()
    assert "разделов: 10" in texts(at.caption) and "фрагментов для индекса: 10" in texts(at.caption)

    at = button(at, "Проиндексировать документ").click().run()
    assert not at.exception and KnowledgeBase().list_documents()[0].filename == "my_metrics.pdf"


def test_broken_pdf_shows_plain_error():
    at = open_page()
    at.file_uploader[0].upload("broken.pdf", b"%PDF-1.7 broken")
    at = at.run()
    assert not at.exception and "PDF" in texts(at.error)
    assert not any("Проиндексировать документ" in b.label for b in at.button)


def test_document_can_be_deleted():
    KnowledgeBase().index_document(DEMO_PDF, "business_metrics.pdf")
    at = open_page()
    delete_box = next(s for s in at.selectbox if s.label == "Удалить документ из индекса")
    at = delete_box.select(delete_box.options[0]).run()
    at = button(at, "Удалить").click().run()
    assert "удалён из индекса" in texts(at.success)
    assert KnowledgeBase().stats()["documents"] == 0


# ------------------------------------------------------------------ поиск
def search(at: AppTest, query: str) -> AppTest:
    at.text_input(key="kb_query").set_value(query)
    return at.run()


def test_context_search_shows_fragments_and_sources():
    KnowledgeBase().index_document(DEMO_PDF, "business_metrics.pdf")
    at = search(open_page("Проверка поиска"), "формула Success Rate")
    assert not at.exception
    body = texts(at.markdown) + texts(at.caption)
    assert "4. Формула Success Rate" in body
    assert "Success Rate = Successful / Transactions × 100" in body
    assert "Факт из документа" in body
    assert "business_metrics.pdf, стр. 1, раздел «4. Формула Success Rate»" in body
    assert body.count("близость") == 3  # Top-3


def test_context_search_reports_nothing_relevant():
    KnowledgeBase().index_document(DEMO_PDF, "business_metrics.pdf")
    at = search(open_page("Проверка поиска"), "рецепт борща со сметаной")
    assert "нельзя опирать на документы" in texts(at.warning)


def test_search_without_documents():
    at = search(open_page("Проверка поиска"), "формула Success Rate")
    assert "нет документов" in texts(at.info)


def test_column_search_maps_phrase_and_marks_it_as_assumption():
    saved_demo_profile()
    KnowledgeBase().index_dataset_profile(DatasetStore().load_profile(DatasetStore().list()[0].dataset_id))
    at = search_mode(open_page("Проверка поиска"), "Колонки и датасеты")
    at = search(at, "Количество операций")
    assert not at.exception
    assert "**Transactions**" in texts(at.info) and "Предположение" in texts(at.info)


def test_ambiguous_column_asks_user_and_remembers_choice():
    frame = pd.DataFrame({"Amount_KZT": [1, 2], "Amount_USD": [3, 4]})
    profile, typed = build_profile(frame, "payments.xlsx")
    DatasetStore().save(typed, profile)
    KnowledgeBase().index_dataset_profile(profile)

    at = search_mode(open_page("Проверка поиска"), "Колонки и датасеты")
    at = search(at, "сумма")
    assert "Подтвердите" in texts(at.warning)
    choice = next(r for r in at.radio if r.label == "Какую колонку вы имели в виду?")
    choice.set_value(next(o for o in choice.options if o.startswith("Amount_USD"))).run()
    at = button(at, "Подтвердить выбор").click().run()
    assert "Amount_USD" in texts(at.success) and "подтверждено пользователем" in texts(at.success)


def test_ambiguity_shows_document_hint():
    frame = pd.DataFrame({"Amount_KZT": [1, 2], "Amount_USD": [3, 4]})
    profile, typed = build_profile(frame, "payments.xlsx")
    DatasetStore().save(typed, profile)
    kb = KnowledgeBase()
    kb.index_dataset_profile(profile)
    kb.index_document(DEMO_PDF, "business_metrics.pdf")

    at = search_mode(open_page("Проверка поиска"), "Колонки и датасеты")
    at = search(at, "объём транзакций")
    assert "Подсказка из документа" in texts(at.info)
    assert "Transaction Volume" in texts(at.info)


# ------------------------------------------------------------------ датасеты
def test_datasets_tab_indexes_saved_datasets():
    profile = saved_demo_profile()
    at = open_page("Датасеты")
    (table,) = [d for d in at.dataframe if "В индексе" in d.value.columns]
    assert table.value.iloc[0]["В индексе"] == "нет"

    at = button(at, "Проиндексировать все сохранённые датасеты").click().run()
    assert not at.exception and "Датасетов в индексе: 1" in texts(at.success)
    assert profile.dataset_id in KnowledgeBase().indexed_dataset_ids()


# ------------------------------------------------------------------ автоиндексация при подтверждении
def upload_and_confirm() -> AppTest:
    at = AppTest.from_function(_analysis_page, default_timeout=60).run()
    at.file_uploader[0].upload("transactions_2026.xlsx", DEMO_XLSX)
    at = at.run()
    return button(at, "Подтвердить").click().run()


def test_confirming_dataset_indexes_its_description_automatically():
    at = upload_and_confirm()
    assert not at.exception
    stats = KnowledgeBase().stats()
    assert stats["datasets"] == 1 and stats["dataset_records"] == 7
    assert "База знаний: Проиндексировано записей: 7" in texts(at.caption)
    assert not at.warning or "не проиндексировано" not in texts(at.warning)


def test_dataset_is_saved_even_if_indexing_is_unavailable(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")  # ключа нет: индексация невозможна
    from datastory.config import get_settings

    get_settings.cache_clear()
    at = upload_and_confirm()
    assert not at.exception
    assert len(DatasetStore().list()) == 1  # сохранение не зависит от индекса
    assert "не проиндексировано" in texts(at.warning) and "OPENAI_API_KEY" in texts(at.warning)


def test_knowledge_page_reports_provider_configuration_error(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    from datastory.config import get_settings

    get_settings.cache_clear()
    at = open_page()
    assert not at.exception and "OPENAI_API_KEY" in texts(at.error)
