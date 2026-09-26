"""Запуск и продолжение анализа: одна сессия анализа = один thread_id.

Streamlit перезапускает скрипт при каждом действии, поэтому граф и чекпоинтер живут в объекте AnalysisRunner
(кэшируется на процесс), а интерфейс хранит только thread_id. При каждой перерисовке интерфейс вызывает snapshot():
он читает сохранённое состояние и ничего не выполняет заново. Ответ пользователя — resume(), он продолжает граф
ровно с точки остановки (interrupt в human_approval).
"""
from __future__ import annotations

import logging
import uuid
from collections import OrderedDict

from langgraph.types import Command
from pydantic import BaseModel, Field

from datastory.errors import WorkflowError
from datastory.mcp_client.connection import McpConnectionError, McpToolError
from datastory.observability import clip, trace_config
from datastory.workflow.graph import WorkflowDeps, build_graph, make_checkpointer
from datastory.workflow.models import AnalysisPlan, AnalysisResult, ApprovalDecision, ApprovalRequest, Phase

logger = logging.getLogger("datastory.workflow")

MAX_THREADS = 20  # сколько сессий анализа держим в памяти процесса


class Failure(BaseModel):
    message: str
    details: str = ""
    kind: str = "internal"  # connection | tool | internal


class WorkflowSnapshot(BaseModel):
    thread_id: str
    dataset_id: str
    phase: Phase
    request: ApprovalRequest | None = None
    plan: AnalysisPlan | None = None
    result: AnalysisResult | None = None
    warnings: list[str] = Field(default_factory=list)
    failure: Failure | None = None


def new_thread_id() -> str:
    return f"analysis-{uuid.uuid4().hex}"


class AnalysisRunner:
    def __init__(self, deps: WorkflowDeps, checkpointer=None, max_threads: int = MAX_THREADS):
        self.deps = deps
        self.checkpointer = checkpointer or make_checkpointer()
        self.graph = build_graph(deps, self.checkpointer)
        self.max_threads = max_threads
        self._threads: OrderedDict[str, str] = OrderedDict()  # thread_id -> dataset_id
        self._failures: dict[str, Failure] = {}

    # ------------------------------------------------------------------ управление сессиями
    def start(self, dataset_id: str, request: str = "", thread_id: str | None = None) -> WorkflowSnapshot:
        """Новая сессия анализа (новый thread_id). Выполняется до подтверждения плана пользователем или до конца."""
        thread_id = thread_id or new_thread_id()
        self._threads[thread_id] = dataset_id
        while len(self._threads) > self.max_threads:  # старые сессии забываем
            old, _ = self._threads.popitem(last=False)
            self.checkpointer.delete_thread(old)
            self._failures.pop(old, None)
        self._run(
            {"dataset_id": dataset_id, "user_request": request, "revision": 0, "approval_round": 0}, thread_id,
            "datastory.analysis.plan", dataset_id, {"goal": clip(request)},
        )
        return self.snapshot(thread_id)

    def resume(self, thread_id: str, decision: ApprovalDecision) -> WorkflowSnapshot:
        """Ответ пользователя на запрос подтверждения. Граф продолжает с точки interrupt."""
        if self.snapshot(thread_id).phase != "awaiting_approval":
            raise WorkflowError("Этот анализ сейчас не ожидает подтверждения: возможно, он уже выполнен или отменён.")
        self._run(
            Command(resume=decision.model_dump(mode="json")), thread_id, f"datastory.analysis.{decision.action}", None,
            {"approved_steps": len(decision.approved_chart_ids)},
        )
        return self.snapshot(thread_id)

    def retry(self, thread_id: str) -> WorkflowSnapshot:
        """Повтор после сбоя (например, MCP-сервер недоступен): граф продолжает с узла, на котором остановился."""
        if thread_id not in self._failures:
            raise WorkflowError("У этого анализа нет сбоя, который можно повторить.")
        self._failures.pop(thread_id)
        self._run(None, thread_id, "datastory.analysis.retry", None, {})
        return self.snapshot(thread_id)

    def forget(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)
        self._failures.pop(thread_id, None)
        self.checkpointer.delete_thread(thread_id)

    # ------------------------------------------------------------------ чтение состояния
    def snapshot(self, thread_id: str) -> WorkflowSnapshot:
        state = self.graph.get_state(self._config(thread_id))
        values = state.values
        failure = self._failures.get(thread_id)
        if not values and failure is None:
            raise WorkflowError("Сессия анализа не найдена: начните анализ заново.")
        base = {
            "thread_id": thread_id, "dataset_id": values.get("dataset_id") or self._threads.get(thread_id, ""),
            "plan": values.get("plan"), "warnings": values.get("warnings", []),
        }
        if failure:
            return WorkflowSnapshot(phase="failed", failure=failure, **base)
        pending = next((i for task in state.tasks for i in task.interrupts), None)
        if pending is not None:
            return WorkflowSnapshot(phase="awaiting_approval", request=ApprovalRequest.model_validate(pending.value), **base)
        status = values.get("status")
        if status == "completed":
            return WorkflowSnapshot(phase="completed", result=values["result"], **base)
        if status == "cancelled":
            return WorkflowSnapshot(phase="cancelled", **base)
        if status == "empty":
            return WorkflowSnapshot(phase="empty", **base)
        raise WorkflowError("Анализ находится в неожиданном состоянии. Начните его заново.")

    # ------------------------------------------------------------------ внутреннее
    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _run(self, payload, thread_id: str, run_name: str, dataset_id: str | None, metadata: dict) -> None:
        """Один запуск графа = один корневой run в LangSmith; все запуски сессии связаны метаданными thread_id."""
        config = trace_config(run_name, thread_id=thread_id, dataset_id=dataset_id or self._threads.get(thread_id), tags=("analysis",), metadata=metadata)
        try:
            self.graph.invoke(payload, config)
        except McpConnectionError as exc:
            self._failures[thread_id] = Failure(message=exc.user_message, details=exc.details or "", kind="connection")
        except McpToolError as exc:
            self._failures[thread_id] = Failure(message=f"Инструмент {exc.tool} отклонил запрос: {exc.message}", kind="tool")
        except Exception as exc:  # noqa: BLE001 — любой сбой узла показываем пользователем, а не роняем страницу
            logger.exception("Сбой воркфлоу (thread %s)", thread_id)
            self._failures[thread_id] = Failure(message="Внутренняя ошибка анализа.", details=f"{type(exc).__name__}: {exc}")
