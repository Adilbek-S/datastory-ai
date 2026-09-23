"""Страница «Анализ данных» поверх MCP: результаты, жизненный цикл подключения при перезапусках Streamlit, ошибки."""
import pytest
from streamlit.testing.v1 import AppTest

from datastory.config import get_settings
from datastory.mcp_client.connection import McpToolError
from datastory.storage.store import DatasetStore
from datastory.ui.mcp import get_analytics_client
from scripts import generate_demo_data as gen

DEMO_XLSX = (gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes()


def _analysis_page():
    from datastory.ui.views import analysis

    analysis.render()


def texts(elements) -> str:
    return "\n".join(e.value for e in elements)


def confirm_button(at: AppTest):
    return next(b for b in at.button if "Подтвердить" in b.label)


def open_and_upload(name: str, content: bytes) -> AppTest:
    at = AppTest.from_function(_analysis_page, default_timeout=90).run()
    at.file_uploader[0].upload(name, content)
    return at.run()


def confirmed_demo() -> AppTest:
    return confirm_button(open_and_upload("transactions_2026.xlsx", DEMO_XLSX)).click().run()


# ================================================================== результаты приходят с MCP-сервера
def test_results_are_computed_by_the_mcp_server():
    at = confirmed_demo()
    assert not at.exception and not at.error
    body = texts(at.markdown) + texts(at.caption)
    for label in ("Количество транзакций", "Success Rate", "Объём транзакций", "Средняя сумма транзакции"):
        assert label in body
    assert "SUM(Successful) / SUM(Transactions) × 100" in body
    assert "Расчёты выполнил MCP-сервер аналитики" in body
    for tool in ("profile_dataset", "calculate_metrics", "create_chart_spec"):
        assert tool in body


def test_charts_are_rendered_from_chart_specs():
    at = confirmed_demo()
    assert len(at.get("plotly_chart")) == 4


def test_march_insight_is_shown_as_a_computation():
    at = confirmed_demo()
    warnings = texts(at.warning)
    assert "Снижение Success Rate: 2026-03" in warnings and "не объясняют его причину" in warnings
    assert "результат вычисления по данным" in texts(at.caption)


def test_streamlit_tool_calls_really_go_through_the_mcp_client():
    confirmed_demo()
    client = get_analytics_client()
    assert client.calls[0] == "profile_dataset"
    assert client.calls.count("calculate_metrics") == 9 and client.calls.count("create_chart_spec") == 4
    assert client.connection.is_connected


# ================================================================== жизненный цикл при перезапусках скрипта
def test_reruns_reuse_the_same_server_process():
    at = confirmed_demo()
    client = get_analytics_client()
    assert client.connection.start_count == 1

    for _ in range(3):  # любое действие пользователя перезапускает скрипт
        at = at.run()
        assert not at.exception
    at = confirm_button(at).click().run()
    assert client.connection.start_count == 1  # один дочерний процесс на все перерисовки
    assert client.calls.count("profile_dataset") >= 2 and get_analytics_client() is client


def test_closed_client_is_replaced_on_the_next_render():
    at = confirmed_demo()
    old = get_analytics_client()
    old.close()  # например, подключение закрыли вручную
    at = confirm_button(at).click().run()
    new = get_analytics_client()
    assert new is not old and not new.is_closed and new.connection.is_connected
    assert not at.exception and not at.error


def test_dead_server_process_is_restarted_transparently():
    at = confirmed_demo()
    client = get_analytics_client()
    client.connection.disconnect()  # процесс сервера остановлен (как при падении)
    at = confirm_button(at).click().run()
    assert not at.exception and not at.error
    assert client.connection.start_count == 2 and client.connection.is_connected


# ================================================================== ошибки
def test_connection_failure_shows_a_clear_error_and_keeps_the_dataset(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_MODULE", "no_such_module_xyz")
    monkeypatch.setenv("MCP_STARTUP_TIMEOUT", "10")
    get_settings.cache_clear()

    at = confirmed_demo()
    assert not at.exception  # страница не падает
    assert "Не удалось подключиться к MCP-серверу «datastory-analytics»" in texts(at.error)
    assert "no_such_module_xyz" in texts(at.error)
    assert "Датасет сохранён" in texts(at.caption)
    assert len(DatasetStore().list()) == 1  # сохранение не зависит от MCP
    assert any("Технические подробности" in e.label for e in at.expander)
    assert "No module named" in texts(at.code)  # диагностика сервера доступна, но спрятана
    assert not at.get("plotly_chart")  # молчаливой подмены локальными расчётами нет


def test_tool_error_is_shown_with_the_tool_name(monkeypatch):
    import datastory.ui.views.analysis as analysis

    def refuse(dataset_id, client):
        raise McpToolError("calculate_metrics", "group_by: колонка «X» не найдена в датасете.")

    monkeypatch.setattr(analysis, "run_analysis_for_dataset", refuse)
    at = confirmed_demo()
    assert not at.exception
    assert "Инструмент calculate_metrics отклонил запрос: group_by: колонка «X» не найдена" in texts(at.error)


def test_dataset_without_payment_metrics_shows_overview_charts():
    csv = "city;score;visits\n" + "\n".join(f"{c};{s};{v}" for c, s, v in [("a", 1.5, 10), ("b", 2.5, 20), ("a", 3.5, 30), ("b", 4.5, 40), ("a", 5.5, 50), ("b", 6.5, 60)])
    at = confirm_button(open_and_upload("generic.csv", csv.encode("utf-8"))).click().run()
    assert not at.exception and not at.error
    assert "нет колонок Transactions" in texts(at.caption)  # объяснение, почему показаны обзорные графики
    assert len(at.get("plotly_chart")) >= 2
    client = get_analytics_client()
    assert client.calls == ["profile_dataset"]  # метрики платёжной системы не запрашивались
