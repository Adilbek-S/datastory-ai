"""Тестовый MCP-сервер для проверки жизненного цикла подключения (не часть приложения)."""
import os
import time

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-lifecycle", log_level="WARNING")


@mcp.tool()
def pid() -> int:
    """PID процесса сервера: по нему видно, переиспользуется ли процесс или он перезапущен."""
    return os.getpid()


@mcp.tool()
def sleep(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


@mcp.tool()
def die() -> str:
    """Имитация падения сервера посреди вызова."""
    os._exit(3)


@mcp.tool()
def fail() -> str:
    raise ValueError("ожидаемая ошибка инструмента")


if __name__ == "__main__":
    mcp.run()
