"""MCP Server (FastMCP): инструменты анализа данных для LLM-агентов.

Инструменты возвращают только краткий профиль (без строк таблицы и без персональных данных).

Запуск (stdio):  python -m datastory.mcp_server.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from datastory.file_processing.loader import read_table
from datastory.profiler.llm_profile import build_llm_profile
from datastory.profiler.profiler import build_profile
from datastory.rag.api import get_default_knowledge_base
from datastory.storage.store import DatasetStore

mcp = FastMCP("datastory-ai")


@mcp.tool()
def profile_file(path: str, sheet_name: str | None = None) -> dict:
    """Краткий профиль таблицы (CSV/XLSX): размер, типы колонок, статистика, качество данных."""
    with open(path, "rb") as fh:
        df = read_table(fh.read(), path, sheet_name)
    profile, _ = build_profile(df, path, sheet_name)
    return build_llm_profile(profile).model_dump(mode="json")


@mcp.tool()
def list_datasets() -> list[dict]:
    """Подтверждённые датасеты в рабочем хранилище (dataset_id, файл, размер)."""
    return [ref.model_dump(mode="json", exclude={"storage_path"}) for ref in DatasetStore().list()]


@mcp.tool()
def get_dataset_profile(dataset_id: str) -> dict:
    """Краткий профиль подтверждённого датасета по его dataset_id."""
    return build_llm_profile(DatasetStore().load_profile(dataset_id)).model_dump(mode="json")


@mcp.tool()
def search_business_context(query: str, top_k: int = 3, dataset_id: str | None = None) -> dict:
    """Top-k фрагментов загруженных PDF по смыслу. Каждый фрагмент — факт из документа со ссылкой на источник."""
    return get_default_knowledge_base().search_business_context(query, top_k, dataset_id=dataset_id).model_dump(mode="json")


@mcp.tool()
def find_columns(query: str, dataset_id: str | None = None) -> dict:
    """Сопоставляет фразу («количество операций») с колонкой. При статусе ambiguous нужно подтверждение пользователя."""
    return get_default_knowledge_base().resolve_column(query, dataset_id=dataset_id).model_dump(mode="json")


@mcp.tool()
def get_source(chunk_id: str) -> dict:
    """Источник (файл, страница, раздел или датасет и колонка) по chunk_id."""
    return get_default_knowledge_base().get_source_reference(chunk_id).model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()
