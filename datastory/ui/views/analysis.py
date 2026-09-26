"""Основной сценарий в три экрана: 1) загрузка данных, 2) план анализа, 3) дашборд.

Этот модуль — экран 1: загрузка (CSV, XLSX, PNG/JPG), предпросмотр, сводка и цель анализа. Экраны 2 и 3 показывает
datastory.ui.views.analysis_run по состоянию сессии анализа (LangGraph thread_id в st.session_state).
"""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from datastory.errors import DataLoadError
from datastory.file_processing.loader import detect_format, file_kind, list_sheets, read_table
from datastory.models import ColumnKind, DatasetProfile, KPI, Severity
from datastory.profiler.profiler import build_profile, new_dataset_id
from datastory.rag.embeddings import EmbeddingError
from datastory.rag.pdf_parser import PdfError
from datastory.report.builder import dataset_period, quality_summary
from datastory.storage.store import DatasetStore
from datastory.ui.kb import get_kb, index_profile_safely
from datastory.ui.theme import kpi_row
from datastory.ui.views.analysis_run import SESSION_KEY, render_session, render_stepper
from datastory.ui.views.image_table import render_image_table
from datastory.ui.workflow import get_runner
from datastory.workflow.models import AUTO_GOAL

KIND_LABELS = {
    ColumnKind.NUMERIC: "Число",
    ColumnKind.CATEGORICAL: "Категория",
    ColumnKind.DATETIME: "Дата/время",
    ColumnKind.BOOLEAN: "Да/Нет",
    ColumnKind.TEXT: "Текст",
}
LABEL_TO_KIND = {label: kind for kind, label in KIND_LABELS.items()}
SEVERITY_ICONS = {Severity.ERROR: ":material/error:", Severity.WARNING: ":material/warning:", Severity.INFO: ":material/info:"}
PREVIEW_ROWS = 100
GOAL_LABEL = "Что вы хотите узнать из этих данных?"
PLAN_BUTTON, AUTO_BUTTON = "Составить план анализа", "Предложить анализ автоматически"


@st.cache_data(show_spinner=False, max_entries=8)
def _sheets(data: bytes) -> list[str]:
    return list_sheets(data)


@st.cache_data(show_spinner=False, max_entries=8)
def _read(data: bytes, filename: str, sheet: str | None) -> pd.DataFrame:
    return read_table(data, filename, sheet)


@st.cache_data(show_spinner=False, max_entries=16)
def _profile(
    raw: pd.DataFrame, filename: str, sheet: str | None,
    type_overrides: tuple, sensitive_overrides: tuple,
) -> tuple[DatasetProfile, pd.DataFrame]:
    # Единая точка для всех форматов: XLSX, CSV и распознанное изображение приходят сюда как DataFrame.
    return build_profile(
        raw, filename, sheet,
        type_overrides={n: ColumnKind(k) for n, k in type_overrides},
        sensitive_overrides=dict(sensitive_overrides),
        dataset_id="000000000000",  # ID присваивается при сохранении; так кэш не зависит от случайного значения
    )


def render() -> None:
    st.title("Анализ данных")
    if render_session():  # экраны 2 и 3: план анализа и дашборд
        return
    _upload_screen()


# --------------------------------------------------------------------------- экран 1: загрузка
def _upload_screen() -> None:
    render_stepper(1)
    st.caption("Загрузите таблицу — DataStory AI предложит план анализа, а после вашего подтверждения построит дашборд.")

    left, right = st.columns([3, 2])
    # Формат проверяет detect_format(): так пользователь видит понятное сообщение на русском,
    # а не английскую ошибку фильтра Streamlit.
    table_file = left.file_uploader(
        "Таблица с данными (CSV, XLSX, PNG или JPG)",
        help="CSV и XLSX читаются напрямую. На изображении (PNG/JPG) таблицу распознаёт Vision-модель, вы проверите результат до анализа.",
    )
    pdf_file = right.file_uploader(
        "PDF с описанием бизнес-показателей (необязательно)", type=["pdf"],
        help="Определения показателей и события из документа станут бизнес-контекстом анализа.",
    )

    if table_file is None:
        st.info("Загрузите файл, чтобы начать.", icon=":material/upload_file:")
        return

    data = table_file.getvalue()
    if file_kind(table_file.name) == "image":
        raw = render_image_table(data, table_file.name)  # None, пока пользователь не подтвердил распознанную таблицу
        if raw is None:
            return
        fmt, sheet = "изображение (Vision)", None
        content = hashlib.sha1(raw.to_json(orient="split", force_ascii=False).encode("utf-8")).hexdigest()[:12]
        signature = f"{table_file.name}|{len(data)}|image|{content}"  # правка ячейки меняет подпись
    else:
        try:
            fmt = detect_format(table_file.name, data)
            sheet = None
            if fmt == "xlsx":
                sheets = _sheets(data)
                sheet = sheets[0] if len(sheets) == 1 else st.selectbox("Лист Excel", sheets)
            raw = _read(data, table_file.name, sheet)
        except DataLoadError as exc:
            st.error(exc.user_message, icon=":material/error:")
            return
        signature = f"{table_file.name}|{len(data)}|{sheet}"

    st.subheader("Предпросмотр")
    st.caption(f"Формат: {fmt.upper()}" + (f" · лист «{sheet}»" if sheet else "") + f" · показаны первые {min(PREVIEW_ROWS, len(raw))} строк")
    st.dataframe(raw.head(PREVIEW_ROWS), width="stretch")

    profile, typed = _overview(raw, table_file.name, sheet, signature)
    _goal_section(profile, typed, pdf_file, table_file.name)


def _overview(raw: pd.DataFrame, filename: str, sheet: str | None, signature: str):
    """Строки, колонки, период данных, качество; типы колонок можно поправить в раскрывающемся блоке."""
    base, _ = _profile(raw, filename, sheet, (), ())
    kpi_slot, quality_slot = st.container(), st.container()

    with st.expander("Типы колонок и персональные данные (при необходимости)"):
        st.markdown("Если тип определён неверно, выберите правильный — сводка пересчитается.")
        table = pd.DataFrame(
            {
                "Колонка": [c.name for c in base.columns],
                "Определено": [KIND_LABELS[c.detected_kind] for c in base.columns],
                "Тип": [KIND_LABELS[c.detected_kind] for c in base.columns],
                "Персональные данные": [c.is_sensitive for c in base.columns],
                "Примеры": [", ".join(c.sample_values[:3]) for c in base.columns],
            }
        )
        edited = st.data_editor(
            table, key=f"types::{signature}", hide_index=True, width="stretch",
            disabled=["Колонка", "Определено", "Примеры"],
            column_config={
                "Тип": st.column_config.SelectboxColumn("Тип", options=list(KIND_LABELS.values()), required=True),
                "Персональные данные": st.column_config.CheckboxColumn(
                    "Персональные данные", help="Значения таких колонок не передаются в LLM"
                ),
            },
        )
    type_overrides = tuple(
        (name, LABEL_TO_KIND[label].value)
        for name, label, det in zip(edited["Колонка"], edited["Тип"], edited["Определено"])
        if label != det
    )
    sensitive_overrides = tuple(
        (name, bool(flag))
        for name, flag, col in zip(edited["Колонка"], edited["Персональные данные"], base.columns)
        if bool(flag) != col.is_sensitive
    )
    profile, typed = _profile(raw, filename, sheet, type_overrides, sensitive_overrides)

    with kpi_slot:
        kpi_row(
            [
                KPI(label="Строк", value=f"{profile.row_count:,}".replace(",", " ")),
                KPI(label="Колонок", value=str(profile.column_count)),
                KPI(label="Период данных", value=dataset_period(profile) or "—", hint=", ".join(profile.detected_date_columns) or "нет колонки с датой"),
                KPI(label="Качество данных", value=quality_summary(profile)),
            ]
        )
        st.write("")
        kinds = st.columns(3)
        kinds[0].markdown(f"**Числовые:** {', '.join(profile.detected_numeric_columns) or '—'}")
        kinds[1].markdown(f"**Временные:** {', '.join(profile.detected_date_columns) or '—'}")
        kinds[2].markdown(f"**Категориальные:** {', '.join(profile.detected_category_columns) or '—'}")
    with quality_slot:
        _quality_summary(profile)
    return profile, typed


def _quality_summary(profile: DatasetProfile) -> None:
    st.subheader("Качество данных")
    issues = profile.quality_issues
    problems = [i for i in issues if i.severity is not Severity.INFO]
    notes = [i for i in issues if i.severity is Severity.INFO]

    if not issues:
        st.success("Проблем не найдено: пропусков и дубликатов нет, типы колонок определены однозначно.")
        return
    if not problems:
        st.success("Серьёзных проблем не найдено.")
    for issue in problems:
        show = st.error if issue.severity is Severity.ERROR else st.warning
        show(issue.message, icon=SEVERITY_ICONS[issue.severity])
    if notes:
        with st.expander(f"Замечания ({len(notes)})"):
            for issue in notes:
                st.markdown(f"- {issue.message}")


# --------------------------------------------------------------------------- цель анализа и запуск
def _goal_section(profile: DatasetProfile, typed: pd.DataFrame, pdf_file, filename: str) -> None:
    st.subheader("Что дальше")
    goal = st.text_area(
        GOAL_LABEL, key="user_goal", height=90,
        placeholder="Например: как менялась успешность транзакций и какой канал даёт больше всего операций",
        help="Сформулируйте вопрос своими словами. Если не знаете, с чего начать, нажмите «Предложить анализ автоматически».",
    ).strip()
    left, right = st.columns(2)
    plan_clicked = left.button(PLAN_BUTTON, type="primary")
    auto_clicked = right.button(AUTO_BUTTON)
    st.caption("Данные сохранятся в рабочем хранилище. Расчёты начнутся только после того, как вы подтвердите план анализа.")

    if plan_clicked and not goal:
        st.warning("Опишите, что вы хотите узнать, или нажмите «Предложить анализ автоматически».", icon=":material/edit_note:")
        return
    if plan_clicked or auto_clicked:
        _start_analysis(goal if plan_clicked else AUTO_GOAL, profile, typed, pdf_file, filename)


def _start_analysis(goal: str, profile: DatasetProfile, typed: pd.DataFrame, pdf_file, filename: str) -> None:
    final = profile.model_copy(update={"dataset_id": new_dataset_id()})
    try:
        reference = DatasetStore().save(typed, final)
    except OSError as exc:
        st.error(f"Не удалось сохранить датасет: {exc}")
        return
    notes: list[str] = []
    indexed, index_message = index_profile_safely(final)  # семантическое описание колонок для RAG
    if not indexed:
        notes.append(index_message)
    document = _index_pdf(pdf_file, reference.dataset_id)
    if document:
        notes.append(document[1])
    with st.spinner("Определяем показатели, ищем определения и составляем план анализа…"):
        snapshot = get_runner().start(reference.dataset_id, goal)
    st.session_state[SESSION_KEY] = {
        "thread_id": snapshot.thread_id, "dataset_id": reference.dataset_id, "filename": filename, "goal": goal, "notes": notes,
    }
    st.rerun()


def _index_pdf(pdf_file, dataset_id: str) -> tuple[bool, str] | None:
    """PDF с описанием показателей — в базу знаний, с привязкой к датасету. Сбой не мешает анализу."""
    if pdf_file is None:
        return None
    try:
        return True, get_kb().index_document(pdf_file.getvalue(), pdf_file.name, dataset_id).message
    except PdfError as exc:
        return False, f"Документ не проиндексирован: {exc.user_message}"
    except EmbeddingError as exc:
        return False, f"Документ не проиндексирован: {exc.user_message}"
