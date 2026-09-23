"""MCP Server (FastMCP): инструменты анализа данных для LLM-агентов.

Инструменты возвращают только краткий профиль (без строк таблицы и без персональных данных).

Запуск (stdio):  python -m datastory.mcp_server.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from datastory.file_processing.loader import read_table
from datastory.profiler.llm_profile import build_llm_profile
from datastory.profiler.profiler import build_profile
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


if __name__ == "__main__":
    mcp.run()
