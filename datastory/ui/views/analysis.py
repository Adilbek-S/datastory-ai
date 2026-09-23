"""Страница «Анализ данных»."""
import streamlit as st

from datastory.evaluation.pipeline import evaluate_result
from datastory.file_processing.loader import UnsupportedFileError, file_kind, load_table
from datastory.ui.theme import kpi_row
from datastory.visualization.engine import build_figure
from datastory.workflow.graph import run_analysis

SEVERITY_RENDERERS = {"warning": st.warning, "positive": st.success, "info": st.info}


def render() -> None:
    st.title("Анализ данных")
    st.caption("Загрузите таблицу — DataStory AI построит профиль, показатели и графики.")

    left, right = st.columns([3, 2])
    table_file = left.file_uploader(
        "Таблица с данными", type=["csv", "xlsx", "xls", "png", "jpg", "jpeg"],
        help="Excel, CSV или изображение с таблицей",
    )
    right.file_uploader(
        "PDF с описанием бизнес-показателей (необязательно)", type=["pdf"],
        help="Используется как контекст для анализа (появится на следующем этапе)",
    )

    if table_file is None:
        st.info("Загрузите файл, чтобы начать.", icon=":material/upload_file:")
        return

    if file_kind(table_file.name) == "image":
        st.warning("Распознавание таблиц на изображениях будет добавлено на следующем этапе.")
        st.image(table_file, caption=table_file.name)
        return

    try:
        df = load_table(table_file.getvalue(), table_file.name)
    except (UnsupportedFileError, ValueError) as exc:
        st.error(f"Не удалось прочитать файл: {exc}")
        return

    st.subheader("Предпросмотр")
    st.dataframe(df.head(100), use_container_width=True)

    if not st.button("Запустить анализ", type="primary"):
        return

    with st.spinner("Анализируем данные…"):
        result = run_analysis(df)

    st.subheader("Ключевые показатели")
    kpi_row(result.kpis)

    st.subheader("Визуализации")
    if result.charts:
        for col, spec in zip(st.columns(2) * len(result.charts), result.charts):
            col.plotly_chart(build_figure(df, spec), use_container_width=True)
    else:
        st.caption("Для этих данных пока нет подходящих графиков.")

    st.subheader("Выводы")
    for insight in result.insights:
        SEVERITY_RENDERERS.get(insight.severity, st.info)(f"**{insight.title}.** {insight.text}")

    with st.expander("Профиль столбцов"):
        st.dataframe(
            [c.model_dump(mode="json") for c in result.profile.column_profiles],
            use_container_width=True,
        )
    with st.expander("Проверка результата (Evaluation)"):
        for case in evaluate_result(result):
            st.write(("✅ " if case.passed else "❌ ") + case.name + (f" — {case.details}" if case.details else ""))
