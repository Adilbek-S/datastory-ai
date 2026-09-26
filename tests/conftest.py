"""Изоляция тестов: без сети, без ключа OpenAI и без записи в боевые каталоги.

В .env лежит настоящий ключ, поэтому переменные окружения (у них приоритет над .env)
принудительно переопределяются для каждого теста. «Живые» тесты помечены @pytest.mark.live
и запускаются только с RUN_LIVE_TESTS=1.
"""
import os

import pandas as pd
import pytest
import streamlit as st

from datastory.config import Settings, get_settings
from datastory.file_processing.loader import read_table
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.profiler.profiler import build_profile
from datastory.storage.store import DatasetStore
from scripts import generate_demo_data as gen
from datastory.mcp_client.connection import close_all_connections, live_connections
from datastory.rag.api import get_default_knowledge_base


def _clear_caches() -> None:
    get_settings.cache_clear()
    get_default_knowledge_base.cache_clear()


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")  # тесты не отправляют трассы в облако, даже если в .env настоящий ключ
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    _clear_caches()
    before = live_connections()  # подключения уровня модуля (общие фикстуры) тест не закрывает
    yield
    close_all_connections(live_connections() - before)  # но и не оставляет после себя процессов MCP-сервера
    st.cache_resource.clear()
    _clear_caches()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="живой тест: запустите с RUN_LIVE_TESTS=1 (тратит запросы OpenAI)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


# ------------------------------------------------------------------ общий MCP-сервер для интеграционных тестов
@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("mcp_workspace")


@pytest.fixture(scope="module")
def datasets(workspace) -> dict[str, str]:
    """Три подтверждённых датасета в отдельном хранилище: платежи (демо), с персональными данными и обычный."""
    store = DatasetStore(workspace)
    raw = read_table((gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes(), "transactions_2026.xlsx", "transactions")
    frames = {
        "demo": (raw, "transactions_2026.xlsx", "transactions"),
        "clients": (
            pd.DataFrame({"email": ["a@b.kz", "c@d.kz", "e@f.kz"], "Transactions": [1, 2, 3], "Successful": [1, 2, 3]}),
            "clients.csv", None,
        ),
        "generic": (pd.DataFrame({"city": ["a", "b", "a"], "score": [1.5, 2.5, 3.5]}), "generic.csv", None),
    }
    ids = {}
    for key, (frame, filename, sheet) in frames.items():
        profile, typed = build_profile(frame, filename, sheet)
        store.save(typed, profile)
        ids[key] = profile.dataset_id
    return ids


@pytest.fixture(scope="module")
def client(workspace, datasets):
    """Настоящий клиент MCP: сервер запускается дочерним процессом (stdio) один раз на модуль."""
    instance = AnalyticsMcpClient.from_settings(Settings(workspace_dir=workspace, _env_file=None))
    yield instance
    instance.close()
