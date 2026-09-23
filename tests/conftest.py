"""Изоляция тестов: без сети, без ключа OpenAI и без записи в боевые каталоги.

В .env лежит настоящий ключ, поэтому переменные окружения (у них приоритет над .env)
принудительно переопределяются для каждого теста. «Живые» тесты помечены @pytest.mark.live
и запускаются только с RUN_LIVE_TESTS=1.
"""
import os

import pytest

from datastory.config import get_settings
from datastory.rag.api import get_default_knowledge_base


def _clear_caches() -> None:
    get_settings.cache_clear()
    get_default_knowledge_base.cache_clear()


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    _clear_caches()
    yield
    _clear_caches()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="живой тест: запустите с RUN_LIVE_TESTS=1 (тратит запросы OpenAI)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
