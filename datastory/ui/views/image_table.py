"""Шаг «Распознавание таблицы на изображении»: Vision-модель → редактируемая таблица → подтверждение.

Подтверждённая таблица дальше идёт обычным путём: Dataset Profiler, хранилище, LangGraph. Результат распознавания хранится
в st.session_state по хешу изображения, поэтому перерисовки Streamlit не отправляют картинку в модель повторно.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from datastory.errors import DataLoadError, LLMUnavailableError
from datastory.file_processing.image_loader import RecognizedTable, recognize_table
from datastory.file_processing.loader import clean_frame
from datastory.llm.client import VisionLLM, get_vision_llm

STATE_KEY = "vision_tables"
CONFIRMED_KEY = "vision_confirmed"


def _vision() -> VisionLLM | None:
    try:
        return get_vision_llm()
    except LLMUnavailableError:
        return None


def _digest(frame: pd.DataFrame) -> str:
    return hashlib.sha1(frame.to_json(orient="split", force_ascii=False).encode("utf-8")).hexdigest()


def _recognized(data: bytes, filename: str, key: str) -> dict:
    """Результат распознавания из сессии; при первом обращении вызывает Vision-модель."""
    store = st.session_state.setdefault(STATE_KEY, {})
    if key not in store:
        try:
            with st.spinner("Vision-модель распознаёт таблицу на изображении…"):
                table: RecognizedTable | None = recognize_table(data, filename, _vision())
            store[key] = {"table": table, "error": None}
        except DataLoadError as exc:
            store[key] = {"table": None, "error": exc.user_message}
    return store[key]


def render_image_table(data: bytes, filename: str) -> pd.DataFrame | None:
    """Возвращает подтверждённую пользователем таблицу или None, пока она не готова."""
    key = hashlib.sha1(data).hexdigest()
    st.subheader("Распознавание таблицы на изображении")
    try:
        st.image(data, caption=filename)
    except Exception:  # noqa: BLE001 — повреждённая картинка не должна ронять страницу
        pass

    entry = _recognized(data, filename, key)
    if entry["error"]:
        st.error(entry["error"], icon=":material/image_not_supported:")
        if st.button("Повторить распознавание", key=f"vision-retry::{key}"):
            st.session_state[STATE_KEY].pop(key, None)
            st.rerun()
        return None

    table: RecognizedTable = entry["table"]
    st.caption(f"Таблица распознана моделью {table.model}: {len(table.frame)} строк, {len(table.frame.columns)} колонок. Проверьте значения и при необходимости исправьте.")
    if table.comment:
        st.caption(table.comment)
    for warning in table.warnings:
        st.warning(warning, icon=":material/warning:")

    original = table.frame
    st.markdown("**Заголовки колонок**")
    headers = st.data_editor(
        pd.DataFrame({"Заголовок": [str(c) for c in original.columns]}), key=f"vision-headers::{key}",
        hide_index=True, width="stretch",
    )["Заголовок"].tolist()
    st.markdown("**Строки таблицы** (можно править ячейки, добавлять и удалять строки)")
    body = original.copy()
    body.columns = [f"c{i}" for i in range(len(body.columns))]
    edited = st.data_editor(
        body.astype("object"), key=f"vision-body::{key}", num_rows="dynamic", width="stretch",
        column_config={f"c{i}": st.column_config.TextColumn(str(h)) for i, h in enumerate(headers)},
    )

    final = edited.copy()
    final.columns = [(str(h).strip() or f"Колонка {i + 1}") for i, h in enumerate(headers)]
    final = final.astype("object").where(final.notna(), None)
    final = final.map(lambda v: v.strip() if isinstance(v, str) and v.strip() else (None if isinstance(v, str) else v))
    try:
        final = clean_frame(final)
    except DataLoadError as exc:
        st.error(exc.user_message)
        return None

    digest = f"{key}:{_digest(final)}"
    if st.button("Подтвердить распознанную таблицу", type="primary", key=f"vision-confirm::{key}"):
        st.session_state[CONFIRMED_KEY] = {"digest": digest}
    confirmed = st.session_state.get(CONFIRMED_KEY)
    if not confirmed:
        st.info("Проверьте таблицу и подтвердите её — затем она пройдёт обычное профилирование.", icon=":material/fact_check:")
        return None
    if confirmed["digest"] != digest:
        st.info("Таблица изменилась после подтверждения. Подтвердите распознанную таблицу ещё раз.", icon=":material/edit:")
        return None
    st.success("Таблица подтверждена: дальше она обрабатывается как обычная таблица.", icon=":material/check_circle:")
    return final
