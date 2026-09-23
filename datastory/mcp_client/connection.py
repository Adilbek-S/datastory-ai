"""Подключение к MCP-серверу по stdio с управляемым жизненным циклом.

Зачем отдельный класс. Streamlit перезапускает скрипт при каждом действии пользователя, а потоки
скриптов сменяются. Поэтому подключение:
  * живёт в собственном потоке со своим циклом asyncio и не привязано к потоку Streamlit;
  * запускается лениво, при первом вызове, и переиспользуется (один дочерний процесс на приложение);
  * при обрыве (процесс завершился, канал закрыт) перезапускается и вызов повторяется один раз —
    инструменты только читают данные, повтор безопасен;
  * ограничено таймаутами: зависший сервер не подвешивает приложение;
  * закрывается явно (close) и при выходе из процесса (atexit), не оставляя сирот.

Все ошибки подключения превращаются в McpConnectionError с понятным сообщением на русском.
"""
from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import re
import tempfile
import threading
import weakref
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_TOOL_ERROR_PREFIX = re.compile(r"^Error executing tool \w+:\s*")
_STDERR_TAIL = 2000


class McpConnectionError(RuntimeError):
    """Не удалось подключиться к MCP-серверу или связь с ним потеряна."""

    def __init__(self, user_message: str, details: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.details = details  # технические подробности (stderr сервера) — для раскрывающегося блока


class McpToolError(RuntimeError):
    """Сервер принял вызов, но инструмент вернул ошибку (например, неверный group_by)."""

    def __init__(self, tool: str, message: str):
        super().__init__(message)
        self.tool = tool
        self.message = message


@dataclass(frozen=True)
class McpServerConfig:
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None
    name: str = "mcp"


@dataclass
class _State:
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None
    session: ClientSession | None = None
    stop: asyncio.Event | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    stderr: Any = None


class McpConnection:
    """Потокобезопасное синхронное подключение к одному MCP-серверу."""

    def __init__(self, config: McpServerConfig, *, startup_timeout: float = 30.0, call_timeout: float = 60.0):
        self.config = config
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self.start_count = 0  # сколько раз запускался процесс сервера (для диагностики и тестов)
        self._lock = threading.RLock()
        self._state: _State | None = None
        self._closed = False
        _LIVE.add(self)

    # ------------------------------------------------------------------ состояние
    @property
    def is_connected(self) -> bool:
        state = self._state
        return bool(state and state.session is not None and state.thread and state.thread.is_alive())

    @property
    def is_closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------ запуск и остановка
    def ensure_started(self) -> None:
        with self._lock:
            if self._closed:
                raise McpConnectionError("Подключение к MCP-серверу закрыто.")
            if not self.is_connected:
                self._teardown()
                self._start()

    def _start(self) -> None:
        state = _State(stderr=tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace"))
        state.loop = asyncio.new_event_loop()
        state.thread = threading.Thread(target=self._run, args=(state,), name=f"mcp-{self.config.name}", daemon=True)
        self._state = state
        state.thread.start()

        if not state.ready.wait(self.startup_timeout + 5):
            details = self._stderr_tail(state)
            self._teardown()
            raise McpConnectionError(
                f"MCP-сервер «{self.config.name}» не запустился за {self.startup_timeout:g} с. Проверьте, что сервер стартует командой "
                f"«{self.config.command} {' '.join(self.config.args)}».",
                details,
            )
        if state.error is not None or state.session is None:
            details = self._stderr_tail(state)
            reason = _describe(state.error)
            self._teardown()
            raise McpConnectionError(
                f"Не удалось подключиться к MCP-серверу «{self.config.name}»: {reason}. Проверьте, что сервер запускается "
                f"командой «{self.config.command} {' '.join(self.config.args)}».",
                details,
            )
        self.start_count += 1

    def _run(self, state: _State) -> None:
        asyncio.set_event_loop(state.loop)
        try:
            state.loop.run_until_complete(self._serve(state))
        finally:
            state.ready.set()
            state.loop.close()

    async def _serve(self, state: _State) -> None:
        params = StdioServerParameters(
            command=self.config.command, args=list(self.config.args), env=self.config.env, cwd=self.config.cwd
        )
        state.stop = asyncio.Event()
        try:
            # Контексты открываются и закрываются в одной задаче (требование anyio), поэтому всё живёт здесь.
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params, errlog=state.stderr))
                session = await stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), self.startup_timeout)
                state.session = session
                state.ready.set()
                await state.stop.wait()
        except BaseException as exc:  # noqa: BLE001 — любая причина обрыва сохраняется для сообщения пользователю
            if state.session is None:
                state.error = exc
        finally:
            state.session = None

    def _teardown(self) -> None:
        state, self._state = self._state, None
        if state is None:
            return
        if state.loop and state.stop is not None and state.thread and state.thread.is_alive():
            try:
                state.loop.call_soon_threadsafe(state.stop.set)
            except RuntimeError:  # цикл уже закрыт
                pass
            state.thread.join(timeout=10)
        try:
            state.stderr.close()
        except Exception:  # noqa: BLE001
            pass

    def disconnect(self) -> None:
        """Останавливает процесс сервера; следующий вызов запустит его заново."""
        with self._lock:
            self._teardown()

    def close(self) -> None:
        """Окончательно закрывает подключение. Безопасно вызывать повторно."""
        with self._lock:
            self._closed = True
            self._teardown()

    def __enter__(self) -> "McpConnection":
        self.ensure_started()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _stderr_tail(state: _State) -> str:
        try:
            state.stderr.flush()
            state.stderr.seek(0)
            return state.stderr.read()[-_STDERR_TAIL:].strip()
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------ вызовы
    def list_tools(self) -> list[str]:
        return [tool.name for tool in self._request(lambda s: s.list_tools(), "list_tools").tools]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        """Вызывает инструмент и возвращает структурированный результат (dict)."""
        result = self._request(lambda s: s.call_tool(name, arguments or {}), name, timeout)
        if result.isError:
            text = next((c.text for c in result.content if getattr(c, "type", "") == "text"), "Инструмент вернул ошибку.")
            raise McpToolError(name, _TOOL_ERROR_PREFIX.sub("", text))
        if result.structuredContent is not None:
            return result.structuredContent
        text = next((c.text for c in result.content if getattr(c, "type", "") == "text"), None)
        try:
            return json.loads(text) if text is not None else None
        except json.JSONDecodeError:
            return text

    def _request(self, make_call, label: str, timeout: float | None = None):
        timeout = timeout or self.call_timeout
        last_error: BaseException | None = None
        details = ""
        for _ in (1, 2):  # второй заход — после перезапуска сервера
            self.ensure_started()
            state = self._state
            if state is None or state.session is None:
                continue
            future = asyncio.run_coroutine_threadsafe(_with_timeout(make_call(state.session), timeout), state.loop)
            try:
                return future.result(timeout + 5)
            except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
                future.cancel()
                self._reset(state)  # зависший сервер перезапустится при следующем вызове
                raise McpConnectionError(
                    f"MCP-сервер «{self.config.name}» не ответил на вызов «{label}» за {timeout:g} с. Соединение будет установлено заново."
                ) from None
            except Exception as exc:  # noqa: BLE001
                if _is_tool_level(exc):
                    raise
                last_error, details = exc, self._stderr_tail(state)
                self._reset(state)  # обрыв связи: перезапуск и повтор
        raise McpConnectionError(
            f"Связь с MCP-сервером «{self.config.name}» потеряна и не восстановилась: {_describe(last_error)}.", details
        )

    def _reset(self, failed: _State) -> None:
        with self._lock:
            if self._state is failed:
                self._teardown()


async def _with_timeout(coro, timeout: float):
    return await asyncio.wait_for(coro, timeout)


def _is_tool_level(exc: BaseException) -> bool:
    """Ошибки, которые не связаны с обрывом связи и не лечатся перезапуском."""
    return isinstance(exc, (McpToolError, McpConnectionError, ValueError, TypeError))


def _describe(exc: BaseException | None) -> str:
    if exc is None:
        return "неизвестная причина"
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:  # anyio оборачивает ошибки в группы
        exc = exc.exceptions[0]
    if isinstance(exc, FileNotFoundError):
        return "исполняемый файл сервера не найден"
    if isinstance(exc, (asyncio.TimeoutError, concurrent.futures.TimeoutError)):
        return "сервер не ответил вовремя"
    text = str(exc).strip() or exc.__class__.__name__
    if "closed" in text.lower():
        return "процесс сервера завершился сразу после запуска"
    return text[:200]


# --------------------------------------------------------------------------- очистка при выходе
_LIVE: "weakref.WeakSet[McpConnection]" = weakref.WeakSet()


def live_connections() -> set["McpConnection"]:
    return set(_LIVE)


def close_all_connections(only: "set[McpConnection] | None" = None) -> None:
    """Закрывает все подключения процесса (atexit, а также teardown в тестах)."""
    for connection in list(only if only is not None else _LIVE):
        connection.close()


atexit.register(close_all_connections)
