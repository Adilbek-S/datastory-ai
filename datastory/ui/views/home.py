"""Страница «Главная»."""
import streamlit as st

from datastory.config import get_settings
from datastory.models import KPI
from datastory.ui.theme import hero, info_card, kpi_row

STEPS = [
    ("Шаг 1", "Загрузите данные", "Excel, CSV или изображение с таблицей. По желанию — PDF с описанием показателей."),
    ("Шаг 2", "Профилирование", "DataStory AI определит структуру данных и доступные показатели."),
    ("Шаг 3", "Дашборд", "Приложение предложит визуализации и выполнит расчёты."),
    ("Шаг 4", "Выводы", "Аналитические выводы на основе данных и вашей базы знаний."),
]


def render() -> None:
    hero(
        "DataStory AI",
        "Загрузите таблицу — получите интерактивный дашборд и аналитические выводы без ручной настройки.",
        badge="MVP · Демо-версия",
    )

    settings = get_settings()
    kpi_row(
        [
            KPI(label="Форматы данных", value="3", hint="CSV · Excel · изображение"),
            KPI(label="Модель", value=settings.openai_model, hint="OpenAI"),
            KPI(label="Эмбеддинги", value="3-small", hint=settings.embedding_model),
            KPI(
                label="Ключ OpenAI",
                value="Задан" if settings.has_openai_key else "Не задан",
                hint="Настраивается в файле .env",
            ),
        ]
    )

    st.markdown("### Как это работает")
    for col, (step, title, text) in zip(st.columns(len(STEPS)), STEPS):
        col.markdown(info_card(title, text, step), unsafe_allow_html=True)

    st.write("")
    from datastory.ui.navigation import ANALYSIS  # локальный импорт: navigation импортирует этот модуль

    if st.button("Перейти к анализу данных", type="primary"):
        st.switch_page(ANALYSIS)
