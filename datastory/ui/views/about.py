"""Страница «О проекте»."""
import streamlit as st

from datastory.ui.theme import info_card

COMPONENTS = [
    ("User Interface", "Streamlit, светлая тема, русскоязычный интерфейс."),
    ("File Processing", "Чтение CSV/Excel, извлечение текста из PDF (PyMuPDF)."),
    ("Dataset Profiler", "Определение структуры данных и типов показателей."),
    ("RAG Engine", "База знаний на ChromaDB и OpenAI Embeddings."),
    ("LangGraph Workflow", "Оркестрация шагов анализа в виде графа."),
    ("MCP Server", "Инструменты анализа для агентов (FastMCP)."),
    ("Analytics Engine", "Расчёт показателей и метрик."),
    ("Visualization Engine", "Подбор и построение графиков Plotly."),
    ("Insight Generator", "Аналитические выводы на GPT-4o-mini."),
    ("Evaluation Pipeline", "Проверка качества результатов, трассировка в LangSmith."),
]


def render() -> None:
    st.title("О проекте")
    st.write(
        "**DataStory AI** — финальный проект курса LLM Engineering: приложение для интеллектуального "
        "анализа данных и автоматического формирования интерактивных аналитических дашбордов."
    )
    st.info("Текущая версия — минимальный каркас (MVP). Полноценная AI-логика будет добавлена поэтапно.")

    st.subheader("Компоненты")
    for row_start in range(0, len(COMPONENTS), 2):
        for col, (title, text) in zip(st.columns(2), COMPONENTS[row_start : row_start + 2]):
            col.markdown(info_card(title, text), unsafe_allow_html=True)
            col.write("")

    st.subheader("Технологии")
    st.write("Python 3.11 · Streamlit · Pandas · Plotly · LangGraph · OpenAI GPT-4o-mini · "
             "text-embedding-3-small · ChromaDB · FastMCP · PyMuPDF · Pydantic · LangSmith · Pytest")
