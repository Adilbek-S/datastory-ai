"""Чат по готовому дашборду. История хранится в st.session_state (отдельно для каждой сессии анализа)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from datastory.chat.models import SUPPORTED_QUESTIONS, ChatAnswer
from datastory.ui.workflow import get_chat_service
from datastory.workflow.models import AnalysisResult

MAX_HISTORY = 60  # сообщений в истории одной сессии
INPUT_PLACEHOLDER = "Задайте вопрос по дашборду"


def _history_key(thread_id: str) -> str:
    return f"chat::{thread_id}"


def render_chat(result: AnalysisResult, thread_id: str) -> None:
    st.markdown("#### Чат по дашборду")
    st.caption(
        "Числа берутся из расчётов MCP, определения и события — из документации (RAG). Если нужного расчёта на дашборде нет, "
        "чат запрашивает его у MCP-сервера. Вопросы вне возможностей MVP получают пояснение."
    )
    history: list[dict] = st.session_state.setdefault(_history_key(thread_id), [])

    pending = None
    with st.expander("Примеры вопросов", expanded=not history):
        for index, example in enumerate(SUPPORTED_QUESTIONS):
            if st.button(example, key=f"chat-example::{thread_id}::{index}"):
                pending = example

    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                _show_answer(ChatAnswer.model_validate(message["answer"]))

    question = st.chat_input(INPUT_PLACEHOLDER, key=f"chat-input::{thread_id}") or pending
    if question:
        with st.spinner("Ищем ответ в результатах расчётов и документации…"):
            answer = get_chat_service().ask(question, result, thread_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer.answer, "answer": answer.model_dump(mode="json")})
        del history[:-MAX_HISTORY]
        st.rerun()

    if history and st.button("Очистить историю чата", key=f"chat-clear::{thread_id}"):
        history.clear()
        st.rerun()


def _show_answer(answer: ChatAnswer) -> None:
    if answer.error:
        st.error(answer.answer)
        return
    if not answer.supported:
        st.info(answer.answer, icon=":material/info:")
        return
    st.markdown(answer.answer)
    if answer.limitation:
        st.caption(f":material/warning: Ограничение интерпретации: {answer.limitation}")
    if not (answer.evidence or answer.context_source or answer.tools):
        return
    with st.expander("Числа, источники и вызванные инструменты"):
        if answer.evidence:
            st.dataframe(
                pd.DataFrame({"Число": e.label, "Значение": e.formatted, "Как получено": e.computation} for e in answer.evidence),
                hide_index=True, width="stretch",
            )
        for source in answer.data_sources:
            st.markdown(f"**Источник данных:** {source}")
        if answer.context_source:
            st.markdown(f"**Источник бизнес-контекста:** {answer.context_source}")
        called = ", ".join(answer.tools) or "новых вызовов не потребовалось: использованы результаты дашборда"
        st.caption(f"Вызвано: {called}. Ответ: " + ("языковая модель, проверена на достоверность" if answer.generated_by == "llm" else "детерминированные правила"))
