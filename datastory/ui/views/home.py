"""Страница «Главная»."""
import streamlit as st

from datastory.config import get_settings
from datastory.models import KPI
from datastory.observability import tracing_status
from datastory.ui.theme import hero, info_card, kpi_row

STEPS = [
    ("Шаг 1", "Загрузите данные", "CSV, Excel или PNG/JPG с таблицей. Опишите, что хотите узнать, или выберите автоматический анализ."),
    ("Шаг 2", "Подтвердите план", "AI предложит до четырёх анализов с пояснениями. Отключите ненужные и нажмите «Построить аналитику»."),
    ("Шаг 3", "Получите дашборд", "KPI, графики, выводы с указанием источников, чат по дашборду и отчёт в Markdown."),
]


def render() -> None:
    hero(
        "DataStory AI",
        "Загрузите таблицу — получите интерактивный дашборд и аналитические выводы без ручной настройки.",
        badge="MVP · Демо-версия",
    )

    settings = get_settings()
    tracing = tracing_status(settings)
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
            KPI(label="LangSmith", value=tracing.label, hint=f"проект {tracing.project}" if tracing.enabled else "см. LANGSMITH_SETUP.md"),
        ]
    )
    if tracing.requested and not tracing.enabled:
        st.warning(tracing.reason, icon=":material/warning:")  # трассировка запрошена, но не работает: не делаем вид, что всё в порядке

    st.markdown("### Как это работает")
    for col, (step, title, text) in zip(st.columns(len(STEPS)), STEPS):
        col.markdown(info_card(title, text, step), unsafe_allow_html=True)

    st.write("")
    from datastory.ui.navigation import ANALYSIS  # локальный импорт: navigation импортирует этот модуль

    if st.button("Перейти к анализу данных", type="primary"):
        st.switch_page(ANALYSIS)
