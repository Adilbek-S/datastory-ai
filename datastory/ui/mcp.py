"""Доступ интерфейса к аналитическому MCP-серверу.

Жизненный цикл при перезапусках скрипта Streamlit. Клиент кэшируется через st.cache_resource:
перерисовки страницы не создают новых дочерних процессов, а все сессии приложения делят одно подключение.
validate отбрасывает закрытый клиент (например, после остановки в тестах), и создаётся новый. Обрыв связи
клиент обрабатывает сам: перезапускает сервер и повторяет вызов. При выходе из процесса подключения
закрывает atexit (datastory.mcp_client.connection).
"""
from __future__ import annotations

import streamlit as st

from datastory.config import get_settings
from datastory.mcp_client.analytics_client import AnalyticsMcpClient


@st.cache_resource(show_spinner=False, validate=lambda client: not client.is_closed)
def _cached_client(server_module: str, workspace_dir: str, startup_timeout: float, call_timeout: float) -> AnalyticsMcpClient:
    # Аргументы — ключ кэша: смена сервера или каталога хранилища даёт новое подключение.
    return AnalyticsMcpClient.from_settings(get_settings())


def get_analytics_client() -> AnalyticsMcpClient:
    settings = get_settings()
    return _cached_client(
        settings.mcp_server_module, str(settings.workspace_dir), settings.mcp_startup_timeout, settings.mcp_call_timeout
    )
