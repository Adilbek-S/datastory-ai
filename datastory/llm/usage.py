"""Учёт токенов LLM: сколько токенов промпта и ответа потрачено (для evaluation и диагностики).

Токены берутся из ответов OpenAI (usage_metadata), а не оцениваются по длине текста.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler


@dataclass(frozen=True)
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageRecorder:
    """Накопитель токенов; безопасен для вызовов из нескольких потоков."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls = self._input = self._output = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._calls += 1
            self._input += input_tokens
            self._output += output_tokens

    def snapshot(self) -> Usage:
        with self._lock:
            return Usage(self._calls, self._input, self._output)

    def reset(self) -> Usage:
        """Возвращает накопленное и обнуляет счётчики."""
        with self._lock:
            usage = Usage(self._calls, self._input, self._output)
            self._calls = self._input = self._output = 0
            return usage


class UsageHandler(BaseCallbackHandler):
    """Callback LangChain: складывает usage_metadata каждого ответа модели в UsageRecorder."""

    def __init__(self, recorder: UsageRecorder):
        self.recorder = recorder

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.recorder.add(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))
