"""Панель анализа: запуск, подтверждение плана пользователем (LangGraph interrupt), дашборд.

Скрипт Streamlit перезапускается при каждом действии, поэтому в st.session_state хранится только thread_id сессии
анализа. Каждая перерисовка читает сохранённое состояние (runner.snapshot) и ничего не выполняет заново;
ответ пользователя продолжает граф с точки остановки (runner.resume).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from datastory.errors import DatasetNotFoundError, WorkflowError
from datastory.evaluation.pipeline import evaluate_result
from datastory.mcp_client.analytics_client import TOOLS
from datastory.storage.store import DatasetStore
from datastory.ui.theme import kpi_row
from datastory.ui.workflow import get_runner
from datastory.visualization.engine import build_figure, suggest_charts
from datastory.workflow.catalog import BY_ID
from datastory.workflow.methodology import STAGE_TITLES
from datastory.workflow.models import AnalysisResult, ApprovalDecision
from datastory.workflow.runner import WorkflowSnapshot

SESSION_KEY = "analysis_session"
INSIGHT_RENDERERS = {"warning": st.warning, "positive": st.success, "info": st.info}
CHART_TYPE_LABELS = {"line": "линейный", "bar": "столбчатый", "pie": "круговой"}
PLACEHOLDER = "— выберите —"


def render_analysis(dataset_id: str) -> None:
    st.subheader("5. Анализ")
    runner = get_runner()
    st.text_input(
        "Что вы хотите узнать? (необязательно)", key=f"request::{dataset_id}",
        placeholder="Например: покажи динамику успешности и распределение по каналам",
        help="Без запроса анализируются все поддерживаемые показатели.",
    )
    if st.button("Запустить анализ", type="primary", key=f"start::{dataset_id}"):
        with st.spinner("Определяем показатели, ищем определения и составляем план…"):
            snapshot = runner.start(dataset_id, st.session_state.get(f"request::{dataset_id}", ""))
        st.session_state[SESSION_KEY] = {"thread_id": snapshot.thread_id, "dataset_id": dataset_id}  # новая сессия = новый thread_id

    session = st.session_state.get(SESSION_KEY)
    if not session or session["dataset_id"] != dataset_id:
        st.caption("Нажмите «Запустить анализ»: сначала будет предложен план, а расчёты начнутся после вашего подтверждения.")
        return
    try:
        snapshot = runner.snapshot(session["thread_id"])
    except WorkflowError as exc:
        st.session_state.pop(SESSION_KEY, None)
        st.warning(exc.user_message)
        return

    if snapshot.phase == "awaiting_approval":
        _approval(runner, snapshot)
    elif snapshot.phase == "completed":
        _dashboard(snapshot.result)
    elif snapshot.phase == "empty":
        _empty(snapshot, dataset_id)
    elif snapshot.phase == "cancelled":
        st.info("Анализ отменён. Измените запрос или запустите анализ заново.")
        _show_warnings(snapshot.warnings)
    else:
        _failure(runner, snapshot)


# --------------------------------------------------------------------------- подтверждение плана
def _approval(runner, snapshot: WorkflowSnapshot) -> None:
    request, plan = snapshot.request, snapshot.request.plan
    st.markdown("#### План анализа — требуется ваше подтверждение")
    if request.notice:
        st.warning(request.notice, icon=":material/help:")
    who = "LLM" if plan.planner == "llm" else "правила (без LLM)"
    st.caption(f"План составлен: {who}. Расчёты не начнутся, пока вы не подтвердите его. Максимум графиков в плане — 4.")
    if plan.goal:
        st.markdown(f"**Цель:** {plan.goal}")
    _show_warnings(snapshot.warnings)
    for item in plan.excluded:
        st.warning(f"Исключено: {item.label} — {item.reason}.", icon=":material/block:")
    for note in plan.notes:
        st.caption(f":material/tune: {note}")

    thread = snapshot.thread_id
    with st.form(f"approval::{thread}::{request.round}"):
        approved, confirmed, choices = [], [], {}
        for chart in plan.charts:
            definition = plan.metric_definition(chart.metric)
            checked = st.checkbox(
                f"**{chart.title}** — {definition.label}, {CHART_TYPE_LABELS[chart.chart_type]} график",
                value=True, key=f"chart::{thread}::{request.round}::{chart.chart_id}",
            )
            if checked:
                approved.append(chart.chart_id)
            st.caption(f"Формула: {definition.formula}. {chart.rationale}".strip())
            if definition.documented:
                st.caption(f":material/menu_book: Определение в документации: {definition.definition_source}")
            else:
                st.caption(":material/warning: Бизнес-определение показателя в документации не найдено.")
            if chart.mapping.status == "ambiguous":
                picked = st.selectbox(
                    f"Ось X графика «{chart.title}»", [PLACEHOLDER, *chart.mapping.candidates],
                    key=f"column::{thread}::{request.round}::{chart.chart_id}", help=chart.mapping.basis,
                )
                if picked != PLACEHOLDER:
                    choices[chart.chart_id] = picked
            else:
                st.caption(f"Ось X: «{chart.x_column}» ({chart.mapping.basis}).")
        undocumented = [m for m in plan.metrics if not m.documented]
        if undocumented:
            st.markdown("**Показатели без документации.** Используется только правило расчёта, которое вы подтвердите явно:")
        for metric in undocumented:
            if st.checkbox(
                f"Подтверждаю правило расчёта: {metric.label} = {metric.formula}",
                key=f"formula::{thread}::{request.round}::{metric.metric}",
            ):
                confirmed.append(metric.metric)
        revise_to = st.multiselect(
            "Изменить выбор показателей (для кнопки «Составить план заново»)",
            options=plan.available_analyses, default=[c.analysis for c in plan.charts],
            format_func=lambda a: BY_ID[a].label, key=f"revise::{thread}::{request.round}",
        )
        left, middle, right = st.columns(3)
        approve = left.form_submit_button("Подтвердить и построить дашборд", type="primary")
        revise = middle.form_submit_button("Составить план заново")
        cancel = right.form_submit_button("Отменить")

    action = "approve" if approve else "revise" if revise else "cancel" if cancel else None
    if action is None:
        return
    decision = ApprovalDecision(
        action=action, approved_chart_ids=approved, confirmed_metrics=confirmed, column_choices=choices, selected_analyses=revise_to
    )
    try:
        with st.spinner("Выполняем расчёты на MCP-сервере и готовим выводы…" if action == "approve" else "Продолжаем…"):
            runner.resume(thread, decision)
    except WorkflowError as exc:
        st.session_state.pop(SESSION_KEY, None)
        st.warning(exc.user_message)
        return
    st.rerun()


# --------------------------------------------------------------------------- результат
def _dashboard(result: AnalysisResult) -> None:
    dashboard = result.dashboard
    st.markdown("#### Дашборд")
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

    for row in range(0, len(dashboard.sections), 2):
        for col, section in zip(st.columns(2), dashboard.sections[row : row + 2]):
            with col:
                st.plotly_chart(build_figure(pd.DataFrame(), section.chart), width="stretch")
                insight = section.insight
                if insight is None:
                    continue
                INSIGHT_RENDERERS.get(insight.severity, st.info)(f"**{insight.title}.** {insight.text}", icon=":material/calculate:")
                if insight.limitation:
                    st.caption(f":material/warning: Ограничение интерпретации: {insight.limitation}")
                with st.expander("Числовые доказательства и источники"):
                    st.dataframe(
                        pd.DataFrame(
                            {"Число": e.label, "Значение": e.formatted, "Как получено": e.computation} for e in insight.evidence
                        ),
                        hide_index=True, width="stretch",
                    )
                    st.markdown(f"**Источник данных:** {insight.data_source}")
                    if insight.context_source:
                        st.markdown(f"**Источник бизнес-контекста:** {insight.context_source}")
                    st.caption("Текст: " + ("языковая модель, проверена на достоверность" if insight.generated_by == "llm" else "детерминированные правила"))
    st.caption("Значок калькулятора — результат вычисления по данным, а не предположение модели. Причины изменений по данным не определяются.")

    with st.expander("Проверка результата (Evaluation)"):
        for case in evaluate_result(result):
            st.write(("✅ " if case.passed else "❌ ") + case.name + (f" — {case.details}" if case.details else ""))
        for check in result.insight_checks:
            if check.violations:
                status = "заменён расчётным текстом" if check.fallback else "исправлен после повторной попытки"
                st.write(f"⚠️ {check.chart_id}: {status}")
                for violation in check.violations:
                    st.caption(violation)


def _show_warnings(warnings: list[str]) -> None:
    for text in warnings:
        st.caption(f":material/info: {text}")


def _empty(snapshot: WorkflowSnapshot, dataset_id: str) -> None:
    st.info("Для этого датасета не удалось составить план из поддерживаемых графиков.", icon=":material/info:")
    _show_warnings(snapshot.warnings)
    _overview_charts(dataset_id)


def _failure(runner, snapshot: WorkflowSnapshot) -> None:
    failure = snapshot.failure
    st.error(failure.message, icon=":material/cable:" if failure.kind == "connection" else ":material/error:")
    if failure.details:
        with st.expander("Технические подробности"):
            st.code(failure.details)
    st.caption("Датасет сохранён. Анализ можно повторить, когда причина сбоя будет устранена.")
    if st.button("Повторить анализ", key=f"retry::{snapshot.thread_id}"):
        runner.retry(snapshot.thread_id)
        st.rerun()


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
