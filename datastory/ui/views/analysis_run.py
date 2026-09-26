"""Экраны 2 и 3: план анализа (подтверждение пользователем — LangGraph interrupt) и дашборд.

Скрипт Streamlit перезапускается при каждом действии, поэтому в st.session_state хранится только описание сессии анализа
(thread_id, датасет, цель). Каждая перерисовка читает сохранённое состояние (runner.snapshot) и ничего не выполняет
заново; ответ пользователя продолжает граф с точки остановки (runner.resume).
"""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from datastory.errors import DatasetNotFoundError, WorkflowError
from datastory.evaluation.pipeline import evaluate_result
from datastory.mcp_client.analytics_client import TOOLS
from datastory.models import DatasetProfile
from datastory.report.builder import DISCLAIMER, based_on, build_markdown_report, key_findings
from datastory.storage.store import DatasetStore
from datastory.ui.theme import kpi_row
from datastory.ui.views.chat_panel import render_chat
from datastory.ui.workflow import get_runner
from datastory.visualization.engine import build_figure, suggest_charts
from datastory.workflow.methodology import STAGE_TITLES
from datastory.workflow.models import AUTO_GOAL, AnalysisResult, ApprovalDecision
from datastory.workflow.runner import WorkflowSnapshot

SESSION_KEY = "analysis_session"
INSIGHT_RENDERERS = {"warning": st.warning, "positive": st.success, "info": st.info}
CHART_TYPE_LABELS = {"line": "линейный график", "bar": "столбчатый график", "pie": "круговая диаграмма"}
PLACEHOLDER = "— выберите —"
STEPS = ("Загрузка данных", "План анализа", "Дашборд")
INCLUDE, BUILD, RESTART = "Включить", "Построить аналитику", "Начать заново"


def render_stepper(current: int) -> None:
    st.markdown(" · ".join(f"**Шаг {i} из 3 — {title}**" if i == current else f"{i}. {title}" for i, title in enumerate(STEPS, 1)))


def render_session() -> bool:
    """Экран 2 или 3 для текущей сессии анализа. False — сессии нет и нужен экран загрузки."""
    session = st.session_state.get(SESSION_KEY)
    if not session:
        return False
    runner = get_runner()
    try:
        snapshot = runner.snapshot(session["thread_id"])
    except WorkflowError as exc:
        st.session_state.pop(SESSION_KEY, None)
        st.warning(exc.user_message)
        return False

    if snapshot.phase == "awaiting_approval":
        _plan_screen(runner, snapshot, session)
    elif snapshot.phase == "completed":
        _dashboard_screen(snapshot.result, snapshot.thread_id, session)
    elif snapshot.phase == "empty":
        _empty(snapshot)
    elif snapshot.phase == "cancelled":
        render_stepper(1)
        st.info("Анализ отменён. Загрузите данные заново или измените вопрос.")
        _show_warnings(snapshot.warnings)
        _restart(runner, snapshot.thread_id)
    else:
        _failure(runner, snapshot)
    return True


def _restart(runner, thread_id: str, key: str = "restart") -> None:
    if st.button(RESTART, key=f"{key}::{thread_id}"):
        st.session_state.pop(SESSION_KEY, None)
        runner.forget(thread_id)
        st.rerun()


# --------------------------------------------------------------------------- экран 2: план анализа
def _plan_screen(runner, snapshot: WorkflowSnapshot, session: dict) -> None:
    request, plan = snapshot.request, snapshot.request.plan
    render_stepper(2)
    st.subheader("План анализа")
    goal = "автоматический анализ: наиболее значимые тенденции и сравнения" if session.get("goal") in ("", AUTO_GOAL) else session["goal"]
    st.markdown(f"**Цель:** {goal}")
    who = "AI (LLM)" if plan.planner == "llm" else "правила (без LLM)"
    st.caption(f"План составлен: {who}. Расчёты не начнутся, пока вы не нажмёте «{BUILD}». Отключите шаги, которые не нужны.")
    if request.notice:
        st.warning(request.notice, icon=":material/help:")
    for note in session.get("notes", []):
        if "не проиндексирован" in note:  # сбой индексации важен: анализ пойдёт без базы знаний
            st.warning(note, icon=":material/warning:")
        else:
            st.caption(f":material/info: {note}")
    _show_warnings(snapshot.warnings)
    for item in plan.excluded:
        st.warning(f"Исключено: {item.label} — {item.reason}.", icon=":material/block:")
    for note in plan.notes:
        st.caption(f":material/tune: {note}")

    thread, round_number = snapshot.thread_id, request.round
    with st.form(f"plan::{thread}::{round_number}"):
        enabled, choices = [], {}
        for step in plan.charts:
            definition = plan.metric_definition(step.metric)
            with st.container(border=True):
                head, toggle = st.columns([5, 1])
                head.markdown(f"#### {step.title}")
                if toggle.checkbox(INCLUDE, value=True, key=f"step::{thread}::{round_number}::{step.chart_id}"):
                    enabled.append(step)
                st.markdown(f"**Зачем:** {step.rationale or 'Показывает ключевой показатель набора данных.'}")
                metric_col, group_col, chart_col = st.columns(3)
                metric_col.markdown(f"**Показатель**\n\n{definition.label}")
                metric_col.caption(definition.formula)
                if step.mapping.status == "ambiguous":
                    picked = group_col.selectbox(
                        "Группировка (выберите колонку)", [PLACEHOLDER, *step.mapping.candidates],
                        key=f"column::{thread}::{round_number}::{step.chart_id}", help=step.mapping.basis,
                    )
                    if picked != PLACEHOLDER:
                        choices[step.chart_id] = picked
                else:
                    group_col.markdown(f"**Группировка**\n\n{step.x_column}")
                chart_col.markdown(f"**Визуализация**\n\n{CHART_TYPE_LABELS[step.chart_type]}")
                if definition.documented:
                    st.caption(f":material/menu_book: Определение показателя найдено в документации: {definition.definition_source}")
                else:
                    st.caption(
                        f":material/warning: Определения показателя в документации нет. Будет использована формула {definition.formula}: "
                        "включая шаг, вы её подтверждаете."
                    )
        submitted = st.form_submit_button(BUILD, type="primary")
    _restart(runner, thread, key="plan-restart")
    if not submitted:
        return

    confirmed = [s.metric for s in enabled if not plan.metric_definition(s.metric).documented]  # включение шага = подтверждение формулы
    decision = ApprovalDecision(
        action="approve", approved_chart_ids=[s.chart_id for s in enabled], confirmed_metrics=list(dict.fromkeys(confirmed)),
        column_choices=choices,
    )
    try:
        with st.spinner("Выполняем расчёты на MCP-сервере и готовим выводы…"):
            runner.resume(thread, decision)
    except WorkflowError as exc:
        st.session_state.pop(SESSION_KEY, None)
        st.warning(exc.user_message)
        return
    st.rerun()


# --------------------------------------------------------------------------- экран 3: дашборд
def _load_profile(dataset_id: str) -> DatasetProfile | None:
    try:
        return DatasetStore().load_profile(dataset_id)
    except DatasetNotFoundError:
        return None


def _file_stem(filename: str) -> str:
    return re.sub(r"[^\w.-]+", "_", filename.rsplit(".", 1)[0]) or "dataset"


def _dashboard_screen(result: AnalysisResult, thread_id: str, session: dict) -> None:
    dashboard = result.dashboard
    render_stepper(3)
    st.subheader(f"Дашборд: {result.summary.filename}")
    st.warning(DISCLAIMER, icon=":material/smart_toy:")
    kpi_row(dashboard.kpis)
    st.caption(
        f"Расчёты выполнил MCP-сервер аналитики (stdio, инструменты {', '.join(TOOLS)}): числа считает Python-код сервера, а не модель."
    )
    info = result.methodology
    if info and info.stages:
        stages = " и ".join(STAGE_TITLES[s] for s in info.stages)
        st.caption(f":material/rule: Методика: Skill «{info.skill}» передана модели на этапах: {stages}. Правила {', '.join(info.rules)} проверяются кодом.")
    else:
        st.caption(":material/rule: Модель не использовалась: правила методики «datastory-analysis» применены кодом (детерминированный режим).")
    for gap in dashboard.documentation_gaps:
        st.warning(f"Нет документации: {gap}. Использовано только подтверждённое вами правило расчёта.", icon=":material/menu_book:")
    _show_warnings([w for w in dashboard.warnings if w not in dashboard.documentation_gaps])

    if not dashboard.sections:
        st.info("Ни один график не построен: см. предупреждения выше.")
    for row in range(0, len(dashboard.sections), 2):
        for col, section in zip(st.columns(2), dashboard.sections[row : row + 2]):
            with col:
                _chart_block(section)

    st.markdown("#### Ключевые выводы")
    with st.container(border=True):
        findings = key_findings(result)
        for finding in findings:
            st.markdown(f"- {finding}")
        if not findings:
            st.caption("Выводов нет: ни один график не построен.")
    st.caption("Значок калькулятора — результат вычисления по данным, а не предположение модели. Причины изменений по данным не определяются.")

    with st.expander("Использованный контекст"):
        _context_block(result, session)

    report = build_markdown_report(result, _load_profile(result.dataset_id))
    st.download_button(
        "Скачать отчёт", data=report.encode("utf-8"), file_name=f"datastory_report_{_file_stem(result.summary.filename)}.md",
        mime="text/markdown", icon=":material/download:", key=f"download::{thread_id}",
    )

    render_chat(result, thread_id)

    with st.expander("Проверка результата (Evaluation)"):
        for case in evaluate_result(result):
            st.write(("✅ " if case.passed else "❌ ") + case.name + (f" — {case.details}" if case.details else ""))
        for check in result.insight_checks:
            if check.violations:
                status = "заменён расчётным текстом" if check.fallback else "исправлен после повторной попытки"
                st.write(f"⚠️ {check.chart_id}: {status}")
                for violation in check.violations:
                    st.caption(violation)
    _restart(get_runner(), thread_id, key="dashboard-restart")


def _chart_block(section) -> None:
    st.plotly_chart(build_figure(pd.DataFrame(), section.chart), width="stretch")
    insight = section.insight
    if insight is None:
        st.caption("Вывод по этому графику не сформирован.")
        return
    st.markdown(f"**{insight.title}**")
    INSIGHT_RENDERERS.get(insight.severity, st.info)(insight.text, icon=":material/calculate:")
    st.caption(f"На основе: {based_on(insight)}")
    if insight.limitation:
        st.caption(f":material/warning: Ограничение интерпретации: {insight.limitation}")
    with st.expander("Числовые доказательства"):
        st.dataframe(
            pd.DataFrame({"Число": e.label, "Значение": e.formatted, "Как получено": e.computation} for e in insight.evidence),
            hide_index=True, width="stretch",
        )
        st.caption("Текст: " + ("AI, проверен на соответствие расчётам" if insight.generated_by == "llm" else "детерминированные правила"))


def _context_block(result: AnalysisResult, session: dict) -> None:
    if not result.context_sources:
        st.caption(
            "Документы базы знаний не использованы: определения показателей и события не найдены. "
            "Загрузите PDF с описанием показателей на первом экране, чтобы анализ опирался на бизнес-контекст."
        )
    for source in result.context_sources:
        kind = f"определение показателя «{source.metric_label}»" if source.kind == "definition" else "сопроводительный контекст"
        st.markdown(f"**{source.citation}** — {kind}")
        st.markdown("> " + " ".join(source.text.split()))
        if source.used_in:
            st.caption("Использовано в выводах: " + ", ".join(source.used_in))
    for note in session.get("notes", []):
        st.caption(f":material/info: {note}")


# --------------------------------------------------------------------------- прочее
def _show_warnings(warnings: list[str]) -> None:
    for text in warnings:
        st.caption(f":material/info: {text}")


def _empty(snapshot: WorkflowSnapshot) -> None:
    render_stepper(2)
    st.info("Для этого датасета не удалось составить план из поддерживаемых графиков.", icon=":material/info:")
    for item in snapshot.plan.excluded if snapshot.plan else []:
        st.caption(f":material/block: {item.label}: {item.reason}.")
    _show_warnings(snapshot.warnings)
    _overview_charts(snapshot.dataset_id)
    _restart(get_runner(), snapshot.thread_id)


def _failure(runner, snapshot: WorkflowSnapshot) -> None:
    render_stepper(2)
    failure = snapshot.failure
    st.error(failure.message, icon=":material/cable:" if failure.kind == "connection" else ":material/error:")
    if failure.details:
        with st.expander("Технические подробности"):
            st.code(failure.details)
    st.caption("Датасет сохранён. Анализ можно повторить, когда причина сбоя будет устранена.")
    if st.button("Повторить анализ", key=f"retry::{snapshot.thread_id}"):
        runner.retry(snapshot.thread_id)
        st.rerun()
    _restart(runner, snapshot.thread_id)


def _overview_charts(dataset_id: str) -> None:
    """Обзорные графики для датасетов без метрик платёжной системы (строятся по таблице, без расчётов метрик)."""
    store = DatasetStore()
    try:
        specs = suggest_charts(store.load_profile(dataset_id))
        frame = store.load_dataframe(dataset_id)
    except DatasetNotFoundError as exc:
        st.error(str(exc))
        return
    if not specs:
        st.caption("Для этих данных пока нет подходящих графиков.")
        return
    st.caption("В датасете нет колонок Transactions / Successful / Failed / Amount_KZT, поэтому показаны обзорные графики.")
    for col, spec in zip(st.columns(2) * len(specs), specs):
        col.plotly_chart(build_figure(frame, spec), width="stretch")
