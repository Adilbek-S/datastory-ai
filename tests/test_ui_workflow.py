"""Страница «Анализ данных»: запуск, подтверждение плана, дашборд, продолжение после перезапуска скрипта Streamlit."""
from streamlit.testing.v1 import AppTest

from datastory.config import get_settings
from datastory.storage.store import DatasetStore
from datastory.ui.mcp import get_analytics_client
from scripts import generate_demo_data as gen

DEMO_XLSX = (gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes()
DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
START, APPROVE, REVISE, CANCEL = "Запустить анализ", "Подтвердить и построить дашборд", "Составить план заново", "Отменить"


def _analysis_page():
    from datastory.ui.views import analysis

    analysis.render()


def texts(elements) -> str:
    return "\n".join(e.value for e in elements)


def everything(at: AppTest) -> str:
    return texts(at.markdown) + "\n" + texts(at.caption) + "\n" + texts(at.warning) + "\n" + texts(at.info)


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def has_button(at: AppTest, label: str) -> bool:
    return any(b.label == label for b in at.button)


def open_and_upload(name: str, content: bytes, pdf: bytes | None = None) -> AppTest:
    at = AppTest.from_function(_analysis_page, default_timeout=90).run()
    at.file_uploader[0].upload(name, content)
    if pdf is not None:
        at.file_uploader[1].upload("business_metrics.pdf", pdf)
    return at.run()


def confirmed(name="transactions_2026.xlsx", content=DEMO_XLSX, pdf=None) -> AppTest:
    at = open_and_upload(name, content, pdf)
    return next(b for b in at.button if b.label.startswith("Подтвердить структуру")).click().run()


def started(**kwargs) -> AppTest:
    return button(confirmed(**kwargs), START).click().run()


def confirm_formulas(at: AppTest) -> None:
    for box in at.checkbox:
        if box.label.startswith("Подтверждаю правило расчёта"):
            box.check()


def approved(**kwargs) -> AppTest:
    at = started(**kwargs)
    confirm_formulas(at)
    return button(at, APPROVE).click().run()


# ================================================================== запуск и подтверждение
def test_analysis_does_not_start_by_itself():
    at = confirmed()
    assert not at.exception and not at.error
    assert has_button(at, START) and not at.get("plotly_chart")
    assert "сначала будет предложен план" in texts(at.caption)
    assert get_analytics_client().calls == []  # без запроса пользователя MCP не вызывался


def test_start_shows_the_plan_and_waits_for_approval():
    at = started()
    assert not at.exception and not at.error
    body = everything(at)
    assert "План анализа — требуется ваше подтверждение" in body and "Расчёты не начнутся, пока вы не подтвердите" in body
    charts = [b for b in at.checkbox if not b.label.startswith("Подтверждаю")]
    assert [c.value for c in charts] == [True] * 4 and "Динамика количества операций" in charts[0].label
    assert "Динамика успешности" in charts[2].label and "Распределение по каналам" in charts[3].label
    assert not at.get("plotly_chart")
    assert get_analytics_client().calls == ["profile_dataset"]  # до подтверждения ничего не рассчитано


def test_missing_documentation_is_flagged_and_requires_formula_confirmation():
    at = started()
    body = everything(at)
    assert "Бизнес-определение показателя в документации не найдено" in body
    formulas = [b for b in at.checkbox if b.label.startswith("Подтверждаю правило расчёта")]
    assert len(formulas) == 3 and not any(f.value for f in formulas)
    at = button(at, APPROVE).click().run()  # формулы не подтверждены: график не строится, вопрос повторяется
    assert not at.exception and not at.get("plotly_chart")
    assert "не подтверждено" in everything(at) or "Подтвердите формулу" in everything(at)
    assert "calculate_metrics" not in get_analytics_client().calls


def test_approval_builds_the_dashboard_from_mcp_results():
    at = approved()
    assert not at.exception and not at.error
    assert len(at.get("plotly_chart")) == 4
    body = everything(at)
    for label in ("Количество транзакций", "Success Rate", "Объём транзакций"):
        assert label in body
    assert "Расчёты выполнил MCP-сервер аналитики" in body
    assert "Модель не использовалась: правила методики «datastory-analysis» применены кодом" in body
    assert "91,98%" in body and "2026-03" in body  # март — период минимума Success Rate, число посчитано MCP
    assert "Причина изменения по данным не установлена" in body
    assert "Нет документации: Success Rate" in body
    client = get_analytics_client()
    assert client.calls.count("calculate_metrics") == 4 and client.calls.count("create_chart_spec") == 4
    assert any("Числовые доказательства и источники" in e.label for e in at.expander)


def test_business_document_gives_definitions_and_sources():
    at = started(pdf=DEMO_PDF)
    assert not any(b.label.startswith("Подтверждаю") for b in at.checkbox)  # определения найдены: формулы подтверждать не нужно
    assert "business_metrics.pdf" in everything(at)
    at = button(at, APPROVE).click().run()
    assert not at.exception and len(at.get("plotly_chart")) == 4
    body = everything(at) + texts(at.expander[0].markdown) if at.expander else everything(at)
    assert "Источник бизнес-контекста" in "\n".join(m.value for e in at.expander for m in e.markdown)
    assert "Нет документации" not in body


def test_reruns_continue_the_same_session_without_repeating_steps():
    at = started()
    thread = at.session_state["analysis_session"]["thread_id"]
    client = get_analytics_client()
    for _ in range(3):  # любое действие пользователя перезапускает скрипт
        at = at.run()
        assert not at.exception and at.session_state["analysis_session"]["thread_id"] == thread
    assert client.calls == ["profile_dataset"]  # план не пересоздаётся, расчётов нет

    confirm_formulas(at)
    at = button(at, APPROVE).click().run()
    calls = list(client.calls)
    for _ in range(3):
        at = at.run()
        assert not at.exception and len(at.get("plotly_chart")) == 4
    assert client.calls == calls  # дашборд читается из чекпоинта, а не считается заново


def test_every_start_gets_its_own_thread():
    at = started()
    first = at.session_state["analysis_session"]["thread_id"]
    at = button(at, START).click().run()
    assert at.session_state["analysis_session"]["thread_id"] != first


def test_rejecting_the_plan_allows_changing_the_metrics():
    at = started()
    at.multiselect[0].set_value(["success_dynamics"])
    at = button(at, REVISE).click().run()
    assert not at.exception
    charts = [b for b in at.checkbox if not b.label.startswith("Подтверждаю")]
    assert len(charts) == 1 and "Динамика успешности" in charts[0].label
    assert "calculate_metrics" not in get_analytics_client().calls
    confirm_formulas(at)
    at = button(at, APPROVE).click().run()
    assert len(at.get("plotly_chart")) == 1


def test_cancel_ends_the_analysis():
    at = button(started(), CANCEL).click().run()
    assert "Анализ отменён" in texts(at.info) and not at.get("plotly_chart")
    assert get_analytics_client().calls == ["profile_dataset"]


def test_ambiguous_column_is_a_question_to_the_user():
    csv = "Month;Created;Transactions;Successful\n" + "\n".join(
        f"2026-0{m};2026-0{m}-15;{100 + m};{90 + m}" for m in range(1, 5)
    )
    at = started(name="ambiguous.csv", content=csv.encode("utf-8"))
    assert not at.exception
    boxes = [s for s in at.selectbox if s.label.startswith("Ось X графика")]
    assert boxes and set(boxes[0].options) == {"— выберите —", "Month", "Created"}
    confirm_formulas(at)
    at = button(at, APPROVE).click().run()  # колонка не выбрана: система спрашивает снова, а не угадывает
    assert "Уточните колонку" in everything(at) and not at.get("plotly_chart")

    for box in [s for s in at.selectbox if s.label.startswith("Ось X графика")]:
        box.select("Month")
    confirm_formulas(at)
    at = button(at, APPROVE).click().run()
    assert not at.exception and len(at.get("plotly_chart")) == 2  # динамика количества и успешности


# ================================================================== ошибки
def test_connection_failure_shows_a_clear_error_and_keeps_the_dataset(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_MODULE", "no_such_module_xyz")
    monkeypatch.setenv("MCP_STARTUP_TIMEOUT", "10")
    get_settings.cache_clear()

    at = started()
    assert not at.exception  # страница не падает
    assert "Не удалось подключиться к MCP-серверу «datastory-analytics»" in texts(at.error)
    assert "no_such_module_xyz" in texts(at.error)
    assert "Датасет сохранён" in texts(at.caption)
    assert len(DatasetStore().list()) == 1  # сохранение не зависит от MCP
    assert any("Технические подробности" in e.label for e in at.expander)
    assert "No module named" in texts(at.code)
    assert not at.get("plotly_chart") and has_button(at, "Повторить анализ")  # молчаливой подмены расчётами нет


def test_dataset_without_payment_metrics_shows_overview_charts():
    csv = "city;score;visits\n" + "\n".join(f"{c};{s};{v}" for c, s, v in [("a", 1.5, 10), ("b", 2.5, 20), ("a", 3.5, 30), ("b", 4.5, 40), ("a", 5.5, 50), ("b", 6.5, 60)])
    at = started(name="generic.csv", content=csv.encode("utf-8"))
    assert not at.exception and not at.error
    assert "нет колонок Transactions" in texts(at.caption)  # объяснение, почему показаны обзорные графики
    assert len(at.get("plotly_chart")) >= 2
    assert get_analytics_client().calls == ["profile_dataset"]  # метрики платёжной системы не запрашивались
