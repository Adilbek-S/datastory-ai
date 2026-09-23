"""Страница «Анализ данных»: загрузка → предпросмотр → профиль → качество → подтверждение → анализ."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from datastory.errors import DataLoadError, DatasetNotFoundError
from datastory.evaluation.pipeline import evaluate_result
from datastory.file_processing.loader import detect_format, file_kind, list_sheets, read_table
from datastory.models import ColumnKind, DatasetProfile, KPI, Severity
from datastory.profiler.profiler import build_profile, new_dataset_id
from datastory.storage.store import DatasetStore
from datastory.ui.theme import kpi_row
from datastory.visualization.engine import build_figure
from datastory.workflow.graph import run_analysis_for_dataset

KIND_LABELS = {
    ColumnKind.NUMERIC: "Число",
    ColumnKind.CATEGORICAL: "Категория",
    ColumnKind.DATETIME: "Дата/время",
    ColumnKind.BOOLEAN: "Да/Нет",
    ColumnKind.TEXT: "Текст",
}
LABEL_TO_KIND = {label: kind for kind, label in KIND_LABELS.items()}
SEVERITY_ICONS = {Severity.ERROR: ":material/error:", Severity.WARNING: ":material/warning:", Severity.INFO: ":material/info:"}
INSIGHT_RENDERERS = {"warning": st.warning, "positive": st.success, "info": st.info}
PREVIEW_ROWS = 100


@st.cache_data(show_spinner=False, max_entries=8)
def _sheets(data: bytes) -> list[str]:
    return list_sheets(data)


@st.cache_data(show_spinner=False, max_entries=8)
def _read(data: bytes, filename: str, sheet: str | None) -> pd.DataFrame:
    return read_table(data, filename, sheet)


@st.cache_data(show_spinner=False, max_entries=16)
def _profile(
    data: bytes, filename: str, sheet: str | None,
    type_overrides: tuple, sensitive_overrides: tuple,
) -> tuple[DatasetProfile, pd.DataFrame]:
    return build_profile(
        _read(data, filename, sheet), filename, sheet,
        type_overrides={n: ColumnKind(k) for n, k in type_overrides},
        sensitive_overrides=dict(sensitive_overrides),
        dataset_id="000000000000",  # ID присваивается при сохранении; так кэш не зависит от случайного значения
    )


def render() -> None:
    st.title("Анализ данных")
    st.caption("Загрузите таблицу, проверьте структуру и подтвердите её — затем DataStory AI построит дашборд.")

    left, right = st.columns([3, 2])
    # Формат проверяет detect_format(): так пользователь видит понятное сообщение на русском,
    # а не английскую ошибку фильтра Streamlit.
    table_file = left.file_uploader(
        "Таблица с данными (XLSX или CSV)",
        help="Поддерживаются XLSX и CSV. Изображения с таблицами появятся на следующем этапе.",
    )
    right.file_uploader(
        "PDF с описанием бизнес-показателей (необязательно)", type=["pdf"],
        help="Используется как контекст для анализа (появится на следующем этапе)",
    )

    if table_file is None:
        st.info("Загрузите файл, чтобы начать.", icon=":material/upload_file:")
        return

    data = table_file.getvalue()
    if file_kind(table_file.name) == "image":
        st.warning("Распознавание таблиц на изображениях будет добавлено на следующем этапе.")
        try:
            st.image(data, caption=table_file.name)
        except Exception:  # noqa: BLE001 — повреждённая картинка не должна ронять страницу
            st.error("Не удалось открыть изображение: файл повреждён.")
        return

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

    st.subheader("1. Предпросмотр")
    st.caption(f"Формат: {fmt.upper()}" + (f" · лист «{sheet}»" if sheet else "") + f" · показаны первые {min(PREVIEW_ROWS, len(raw))} строк")
    st.dataframe(raw.head(PREVIEW_ROWS), width="stretch")

    profile, typed, type_overrides, sensitive_overrides = _profile_section(data, table_file.name, sheet, signature)
    _quality_section(profile)
    _confirm_section(profile, typed, signature, type_overrides, sensitive_overrides)
    _saved_datasets()


# --------------------------------------------------------------------------- 2. профиль
def _profile_section(data: bytes, filename: str, sheet: str | None, signature: str):
    st.subheader("2. Профиль датасета")
    base, _ = _profile(data, filename, sheet, (), ())
    kpi_slot = st.container()

    st.markdown("**Типы колонок.** Если тип определён неверно, выберите правильный — профиль пересчитается.")
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
    profile, typed = _profile(data, filename, sheet, type_overrides, sensitive_overrides)

    with kpi_slot:
        kpi_row(
            [
                KPI(label="Строк", value=f"{profile.row_count:,}".replace(",", " ")),
                KPI(label="Колонок", value=str(profile.column_count)),
                KPI(label="Пропущено ячеек", value=str(profile.missing_cell_count)),
                KPI(label="Дубликатов строк", value=str(profile.duplicate_row_count)),
            ]
        )
    st.write("")

    kinds = st.columns(3)
    kinds[0].markdown(f"**Числовые:** {', '.join(profile.detected_numeric_columns) or '—'}")
    kinds[1].markdown(f"**Временные:** {', '.join(profile.detected_date_columns) or '—'}")
    kinds[2].markdown(f"**Категориальные:** {', '.join(profile.detected_category_columns) or '—'}")

    with st.expander("Подробно по колонкам"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Колонка": c.name,
                    "Тип": KIND_LABELS[c.kind] + (" (изменён)" if c.is_overridden else ""),
                    "Тип данных": c.dtype,
                    "Пропусков, %": c.missing_pct,
                    "Уникальных": c.unique_count,
                    "Мин": (c.numeric_stats.min if c.numeric_stats else c.date_min),
                    "Макс": (c.numeric_stats.max if c.numeric_stats else c.date_max),
                    "Среднее": (c.numeric_stats.mean if c.numeric_stats else None),
                }
                for c in profile.columns
            ),
            hide_index=True, width="stretch",
        )
    return profile, typed, type_overrides, sensitive_overrides


# --------------------------------------------------------------------------- 3. качество
def _quality_section(profile: DatasetProfile) -> None:
    st.subheader("3. Качество данных")
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


# --------------------------------------------------------------------------- 4. подтверждение
def _confirm_section(profile, typed, signature, type_overrides, sensitive_overrides) -> None:
    st.subheader("4. Подтверждение структуры")
    state_key = repr((signature, type_overrides, sensitive_overrides))
    description = st.text_area(
        "Описание датасета", value=profile.description, key=f"desc::{state_key}",
        help="Краткое описание попадёт в профиль датасета и будет использоваться при анализе.",
    )
    st.caption("После подтверждения таблица (с выбранными типами колонок) сохраняется в рабочее хранилище приложения.")

    if st.button("Подтвердить структуру и сохранить датасет", type="primary"):
        final = profile.model_copy(update={"dataset_id": new_dataset_id(), "description": description})
        try:
            reference = DatasetStore().save(typed, final)
        except OSError as exc:
            st.error(f"Не удалось сохранить датасет: {exc}")
            return
        st.session_state["confirmed"] = {"key": state_key, "dataset_id": reference.dataset_id}

    confirmed = st.session_state.get("confirmed")
    if not confirmed:
        return
    if confirmed["key"] != state_key:
        st.info("Файл или структура изменились после подтверждения. Подтвердите структуру ещё раз.")
        return

    st.success(f"Датасет сохранён. ID: `{confirmed['dataset_id']}`", icon=":material/check_circle:")
    _analysis_section(confirmed["dataset_id"])


# --------------------------------------------------------------------------- 5. результаты
def _analysis_section(dataset_id: str) -> None:
    store = DatasetStore()
    try:
        result = run_analysis_for_dataset(dataset_id, store)
        df = store.load_dataframe(dataset_id)
    except DatasetNotFoundError as exc:
        st.error(str(exc))
        return

    st.subheader("5. Результаты анализа")
    kpi_row(result.kpis)

    st.markdown("#### Визуализации")
    if result.charts:
        for col, spec in zip(st.columns(2) * len(result.charts), result.charts):
            col.plotly_chart(build_figure(df, spec), width="stretch")
    else:
        st.caption("Для этих данных пока нет подходящих графиков.")

    st.markdown("#### Выводы")
    for insight in result.insights:
        INSIGHT_RENDERERS.get(insight.severity, st.info)(f"**{insight.title}.** {insight.text}")

    with st.expander("Проверка результата (Evaluation)"):
        for case in evaluate_result(result):
            st.write(("✅ " if case.passed else "❌ ") + case.name + (f" — {case.details}" if case.details else ""))


def _saved_datasets() -> None:
    refs = DatasetStore().list()
    if not refs:
        return
    with st.expander(f"Сохранённые датасеты ({len(refs)})"):
        st.dataframe(
            pd.DataFrame(
                {
                    "ID": r.dataset_id,
                    "Файл": r.filename,
                    "Лист": r.sheet_name or "",
                    "Строк": r.row_count,
                    "Колонок": r.column_count,
                    "Сохранён (UTC)": r.created_at.strftime("%Y-%m-%d %H:%M"),
                }
                for r in refs
            ),
            hide_index=True, width="stretch",
        )
