"""MCP Server (FastMCP): инструменты анализа данных для LLM-агентов.

Запуск (stdio):  python -m datastory.mcp_server.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from datastory.file_processing.loader import load_table
from datastory.profiler.profiler import profile_dataframe

mcp = FastMCP("datastory-ai")


@mcp.tool()
def profile_file(path: str) -> dict:
    """Возвращает профиль таблицы (CSV/Excel): размер, типы столбцов, пропуски."""
    with open(path, "rb") as fh:
        df = load_table(fh.read(), path)
    return profile_dataframe(df).model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()
