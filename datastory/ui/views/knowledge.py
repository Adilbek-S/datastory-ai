"""Страница «База знаний»."""
import streamlit as st

from datastory.config import get_settings
from datastory.file_processing.loader import extract_pdf_chunks
from datastory.models import KPI
from datastory.rag.engine import RagEngine, split_text
from datastory.ui.theme import kpi_row


def render() -> None:
    st.title("База знаний")
    st.caption("PDF-документы с описанием бизнес-показателей помогают точнее интерпретировать данные.")

    settings = get_settings()
    try:
        indexed = RagEngine().count()
        store_status = "Готово"
    except Exception as exc:  # noqa: BLE001 — хранилище не должно ронять страницу
        indexed, store_status = 0, f"Ошибка: {exc}"

    kpi_row(
        [
            KPI(label="Фрагментов в индексе", value=str(indexed), hint="ChromaDB"),
            KPI(label="Хранилище", value=store_status, hint=str(settings.chroma_dir.name)),
        ]
    )

    st.subheader("Загрузка документа")
    pdf = st.file_uploader("PDF-файл", type=["pdf"])
    if pdf is None:
        return

    pages = extract_pdf_chunks(pdf.getvalue(), pdf.name)
    chunks = [piece for page in pages for piece in split_text(page)]
    st.success(f"Извлечено страниц: {len(pages)}, фрагментов для индексации: {len(chunks)}.")
    if pages:
        with st.expander("Текст первой страницы"):
            st.text(pages[0].text[:1500])

    st.button("Добавить в базу знаний", type="primary", disabled=True,
              help="Индексация (OpenAI Embeddings + ChromaDB) появится на следующем этапе.")
