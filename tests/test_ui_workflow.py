"""Сценарий в три экрана: загрузка → план анализа (подтверждение) → дашборд; продолжение после перезапуска скрипта."""
from streamlit.testing.v1 import AppTest

from datastory.config import get_settings
from datastory.report.builder import DISCLAIMER
from datastory.storage.store import DatasetStore
from datastory.ui.mcp import get_analytics_client
from datastory.workflow.models import AUTO_GOAL
from scripts import generate_demo_data as gen

DEMO_XLSX = (gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes()
DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
GOAL_LABEL = "Что вы хотите узнать из этих данных?"
PLAN, AUTO, BUILD, INCLUDE, RESTART = (
    "Составить план анализа", "Предложить анализ автоматически", "Построить аналитику", "Включить", "Начать заново",
)


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


def uploaded(name="transactions_2026.xlsx", content=DEMO_XLSX, pdf=None) -> AppTest:
    """Экран 1 после загрузки файла."""
    return open_and_upload(name, content, pdf)


def started(goal: str | None = None, **kwargs) -> AppTest:
    """Экран 2: автоматический анализ или анализ по цели пользователя."""
    at = uploaded(**kwargs)
    if goal is None:
        return button(at, AUTO).click().run()
    at.text_area[0].set_value(goal).run()
    return button(at, PLAN).click().run()


def approved(goal: str | None = None, **kwargs) -> AppTest:
    """Экран 3: план подтверждён без изменений."""
    return button(started(goal, **kwargs), BUILD).click().run()


def steps(at: AppTest) -> list:
    return [c for c in at.checkbox if c.label == INCLUDE]


# ================================================================== экран 1: загрузка
def test_upload_screen_shows_the_overview_and_the_goal_field():
    at = uploaded()
    assert not at.exception and not at.error
    assert [s.value for s in at.subheader] == ["Предпросмотр", "Качество данных", "Что дальше"]
    body = texts(at.markdown)
    assert "Шаг 1 из 3 — Загрузка данных" in body
    for card in ("Строк</div><div class=\"value\">18", "Колонок</div><div class=\"value\">6", "Период данных</div><div class=\"value\">2026-01 — 2026-06", "Качество данных</div><div class=\"value\">Без замечаний"):
        assert card in body
    assert "Проблем не найдено" in texts(at.success)
    assert at.text_area[0].label == GOAL_LABEL and at.text_area[0].value == ""
    assert has_button(at, PLAN) and has_button(at, AUTO)
    assert DatasetStore().list() == [] and get_analytics_client().calls == []  # без действия пользователя ничего не сохраняется


def test_own_goal_is_required_for_the_plan_button():
    at = button(uploaded(), PLAN).click().run()
    assert not at.exception and "Опишите, что вы хотите узнать" in texts(at.warning)
    assert DatasetStore().list() == [] and "analysis_session" not in at.session_state


def test_automatic_analysis_uses_the_fixed_goal():
    at = started()
    assert at.session_state["analysis_session"]["goal"] == AUTO_GOAL
    assert "Найди наиболее значимые тенденции, сравнения и аномалии" in AUTO_GOAL and "не более четырёх" in AUTO_GOAL
    assert "автоматический анализ" in texts(at.markdown)
    saved = DatasetStore().list()
    assert len(saved) == 1 and saved[0].row_count == 18


def test_users_own_goal_reaches_the_workflow():
    goal = "Как менялась успешность транзакций?"
    at = started(goal)
    session = at.session_state["analysis_session"]
    assert session["goal"] == goal and f"**Цель:** {goal}" in texts(at.markdown)
    from datastory.ui.workflow import get_runner

    state = get_runner().graph.get_state({"configurable": {"thread_id": session["thread_id"]}})
    assert state.values["user_request"] == goal


# ================================================================== экран 2: план анализа
def test_plan_screen_shows_a_card_for_every_step():
    at = started()
    assert not at.exception and not at.error
    body = texts(at.markdown)
    assert "Шаг 2 из 3 — План анализа" in body and [s.value for s in at.subheader] == ["План анализа"]
    for title in ("Динамика количества операций", "Динамика объёма", "Динамика успешности", "Распределение по каналам"):
        assert f"#### {title}" in body
    assert body.count("**Зачем:**") == 4
    assert "**Показатель**\n\nSuccess Rate" in body and "**Группировка**\n\nMonth" in body and "**Группировка**\n\nChannel" in body
    assert body.count("**Визуализация**\n\nлинейный график") == 3 and "**Визуализация**\n\nкруговая диаграмма" in body
    assert [c.value for c in steps(at)] == [True] * 4 and has_button(at, BUILD)
    assert not at.get("plotly_chart")
    assert get_analytics_client().calls == ["profile_dataset"]  # до подтверждения ничего не рассчитано


def test_missing_documentation_is_stated_on_the_cards():
    body = everything(started())
    assert body.count("Определения показателя в документации нет") == 4 and "включая шаг, вы её подтверждаете" in body


def test_documentation_found_in_the_uploaded_pdf_is_shown_on_the_cards():
    body = everything(started(pdf=DEMO_PDF))
    assert "Определение показателя найдено в документации: business_metrics.pdf" in body
    assert "Определения показателя в документации нет" not in body


def test_disabled_steps_are_not_built():
    at = started()
    steps(at)[0].uncheck()
    steps(at)[1].uncheck()
    at = button(at, BUILD).click().run()
    assert not at.exception and len(at.get("plotly_chart")) == 2
    assert get_analytics_client().calls.count("calculate_metrics") == 2


def test_disabling_every_step_keeps_the_plan_and_explains():
    at = started()
    for box in steps(at):
        box.uncheck()
    at = button(at, BUILD).click().run()
    assert not at.exception and "Не выбрано ни одного графика" in everything(at) and not at.get("plotly_chart")
    assert has_button(at, BUILD)


def test_ambiguous_column_is_a_question_on_the_card():
    csv = "Month;Created;Transactions;Successful\n" + "\n".join(f"2026-0{m};2026-0{m}-15;{100 + m};{90 + m}" for m in range(1, 5))
    at = started(name="ambiguous.csv", content=csv.encode("utf-8"))
    boxes = [s for s in at.selectbox if s.label.startswith("Группировка")]
    assert boxes and set(boxes[0].options) == {"— выберите —", "Month", "Created"}
    at = button(at, BUILD).click().run()  # колонка не выбрана: система спрашивает снова, а не угадывает
    assert "Уточните колонку" in everything(at) and not at.get("plotly_chart")

    for box in [s for s in at.selectbox if s.label.startswith("Группировка")]:
        box.select("Month")
    at = button(at, BUILD).click().run()
    assert not at.exception and len(at.get("plotly_chart")) == 2  # динамика количества и успешности


def test_restart_returns_to_the_upload_screen():
    at = button(started(), RESTART).click().run()
    assert "analysis_session" not in at.session_state and "Загрузите файл, чтобы начать" in texts(at.info)


# ================================================================== экран 3: дашборд
def test_dashboard_has_the_required_blocks():
    at = approved(pdf=DEMO_PDF)
    assert not at.exception and not at.error
    body = everything(at)
    assert "Шаг 3 из 3 — Дашборд" in body and [s.value for s in at.subheader][0] == "Дашборд: transactions_2026.xlsx"
    assert DISCLAIMER in texts(at.warning)  # предупреждение об автоматической генерации выводов
    for label in ("Количество транзакций", "Success Rate", "Объём транзакций"):
        assert label in body  # KPI
    assert len(at.get("plotly_chart")) == 4
    assert "#### Ключевые выводы" in body
    assert any(e.label == "Использованный контекст" for e in at.expander)
    assert len(at.get("download_button")) == 1 and at.get("download_button")[0].proto.label == "Скачать отчёт"


def test_every_chart_has_a_title_an_insight_and_its_basis():
    at = approved(pdf=DEMO_PDF)
    body = texts(at.markdown)
    for title in ("Динамика количества операций", "Динамика объёма", "Динамика успешности", "Распределение по каналам"):
        assert f"**{title}**" in body  # заголовок вывода под графиком
    based = [c.value for c in at.caption if c.value.startswith("На основе:")]
    assert len(based) == 4 and all("MCP calculate_metrics" in b and "business_metrics.pdf" in b for b in based)
    assert "91,98%" in texts(at.info) + texts(at.warning)  # март — минимум Success Rate, число посчитано MCP
    client = get_analytics_client()
    assert client.calls.count("calculate_metrics") == 4 and client.calls.count("create_chart_spec") == 4


def test_key_findings_block_lists_the_main_conclusions():
    at = approved(pdf=DEMO_PDF)
    findings = [m.value for m in at.markdown if m.value.startswith("- **")]
    assert findings and findings[0].startswith("- **Динамика успешности.**")  # предупреждение — первым
    assert "2026-02 → 2026-03" in findings[0]
    assert any("Причины изменений по данным не установлены" in m.value for m in at.markdown)


def test_used_context_shows_the_knowledge_base_sources():
    at = approved(pdf=DEMO_PDF)
    context = next(e for e in at.expander if e.label == "Использованный контекст")
    body = texts(context.markdown)
    assert "business_metrics.pdf, стр. 1" in body and "определение показателя «Success Rate»" in body
    assert "Использовано в выводах" in texts(context.caption)


def test_used_context_says_so_when_no_documents_were_used():
    at = approved()
    context = next(e for e in at.expander if e.label == "Использованный контекст")
    assert "Документы базы знаний не использованы" in texts(context.caption)
    assert "Нет документации: Success Rate" in everything(at)


def test_dashboard_without_a_model_says_the_rules_were_applied_by_code():
    assert "Модель не использовалась: правила методики «datastory-analysis» применены кодом" in everything(approved())


# ================================================================== перезапуски скрипта и сессии
def test_reruns_continue_the_same_session_without_repeating_steps():
    at = started()
    thread = at.session_state["analysis_session"]["thread_id"]
    client = get_analytics_client()
    for _ in range(3):  # любое действие пользователя перезапускает скрипт
        at = at.run()
        assert not at.exception and at.session_state["analysis_session"]["thread_id"] == thread
    assert client.calls == ["profile_dataset"]  # план не пересоздаётся, расчётов нет

    at = button(at, BUILD).click().run()
    calls = list(client.calls)
    for _ in range(3):
        at = at.run()
        assert not at.exception and len(at.get("plotly_chart")) == 4
    assert client.calls == calls  # дашборд читается из чекпоинта, а не считается заново


def test_every_start_gets_its_own_thread():
    at = started()
    first = at.session_state["analysis_session"]["thread_id"]
    at = button(at, RESTART).click().run()
    at.file_uploader[0].upload("transactions_2026.xlsx", DEMO_XLSX)
    at = button(at.run(), AUTO).click().run()
    assert at.session_state["analysis_session"]["thread_id"] != first


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


def test_dataset_without_measures_shows_overview_charts():
    """Ни колонок платёжной системы, ни числовых колонок: план из поддерживаемых анализов не составить, показаны обзорные графики."""
    csv = "city;segment\n" + "\n".join(f"{c};{g}" for c, g in [("a", "x"), ("b", "y"), ("a", "x"), ("b", "y"), ("a", "y"), ("b", "x")])
    at = started(name="generic.csv", content=csv.encode("utf-8"))
    assert not at.exception and not at.error
    assert "нет колонок Transactions" in texts(at.caption)  # объяснение, почему показаны обзорные графики
    assert len(at.get("plotly_chart")) >= 1
    assert get_analytics_client().calls == ["profile_dataset"]  # метрики платёжной системы не запрашивались
    assert has_button(at, RESTART)


def test_sales_like_dataset_gets_a_plan_of_sums_instead_of_payment_analyses():
    rows = [f"2026-0{m};{r};{100 * m + i}" for m in range(1, 5) for i, r in enumerate(("Almaty", "Astana"))]
    csv = "Month;Region;Revenue_KZT\n" + "\n".join(rows)
    at = started(name="mini_sales.csv", content=csv.encode("utf-8"))
    assert not at.exception and not at.error
    body = texts(at.markdown)
    assert "#### Динамика показателя: Revenue" in body and "#### Сравнение по категориям: Revenue" in body
    assert "Транзакц" not in body
