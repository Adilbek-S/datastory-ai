"""Демо-сценарий трассировки LangSmith: sales_2026.xlsx + вопрос про выручку и регионы, без интерфейса.

    python scripts/run_sales_scenario.py

Скрипт выполняет настоящий сценарий (OpenAI, MCP-сервер, RAG) и затем проверяет, что трассы ДЕЙСТВИТЕЛЬНО появились в LangSmith:
запрашивает у LangSmith API корневые run-ы проекта и выводит дерево последнего. Проверка идёт через API облака, а не через
локальные счётчики. Если трассировка не включена (нет LANGSMITH_TRACING=true или LANGSMITH_API_KEY), скрипт так и скажет и
завершится с кодом 2: успешная интеграция не имитируется. Без OPENAI_API_KEY код возврата 3.
"""
from __future__ import annotations

import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datastory.chat.graph import ChatService  # noqa: E402
from datastory.config import PROJECT_ROOT, Settings, get_settings  # noqa: E402
from datastory.errors import LLMUnavailableError  # noqa: E402
from datastory.file_processing.loader import read_table  # noqa: E402
from datastory.llm.client import get_llm  # noqa: E402
from datastory.mcp_client.analytics_client import AnalyticsMcpClient  # noqa: E402
from datastory.observability import tracing_status  # noqa: E402
from datastory.profiler.profiler import build_profile  # noqa: E402
from datastory.rag.knowledge_base import KnowledgeBase  # noqa: E402
from datastory.storage.store import DatasetStore  # noqa: E402
from datastory.workflow.graph import WorkflowDeps  # noqa: E402
from datastory.workflow.models import ApprovalDecision  # noqa: E402
from datastory.workflow.runner import AnalysisRunner  # noqa: E402

SALES_FILE = PROJECT_ROOT / "data" / "demo" / "sales_2026.xlsx"
GOAL = "Покажи динамику выручки по месяцам, сравни регионы и найди основные изменения"
CHAT_QUESTION = "Какие изменения заслуживают внимания?"
WAIT_SECONDS = 60


def run_scenario(work: Path) -> str:
    """Возвращает thread_id сессии анализа."""
    settings = get_settings()
    raw = read_table(SALES_FILE.read_bytes(), SALES_FILE.name, "sales")
    profile, typed = build_profile(raw, SALES_FILE.name, "sales")
    DatasetStore(work / "workspace").save(typed, profile)

    client = AnalyticsMcpClient.from_settings(Settings(workspace_dir=work / "workspace", _env_file=None))
    try:
        kb = KnowledgeBase()
        deps = WorkflowDeps.of(client, get_llm(settings), kb)
        runner = AnalysisRunner(deps)
        snapshot = runner.start(profile.dataset_id, GOAL)
        plan = snapshot.request.plan
        print(f"План ({plan.planner}): " + "; ".join(f"{c.title} [{c.chart_type}, {c.x_column}]" for c in plan.charts))
        done = runner.resume(
            snapshot.thread_id,
            ApprovalDecision(action="approve", approved_chart_ids=[c.chart_id for c in plan.charts], confirmed_metrics=[m.metric for m in plan.metrics]),
        )
        for insight in done.result.insights:
            print(f"- {insight.title}: {insight.text}")
        answer = ChatService(deps).ask(CHAT_QUESTION, done.result, snapshot.thread_id)
        print(f"Чат ({CHAT_QUESTION}): {answer.answer}")
        return snapshot.thread_id
    finally:
        client.close()


def verify_traces(thread_id: str, started: datetime) -> bool:
    """Ищет в LangSmith корневые run-ы этой сессии по thread_id; ждёт, пока фоновая отправка закончится."""
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import Client

    wait_for_all_tracers()  # дожидаемся отправки буфера
    settings = get_settings()
    client = Client()
    deadline = time.time() + WAIT_SECONDS
    roots = []
    while time.time() < deadline:
        roots = [
            r for r in client.list_runs(project_name=settings.langsmith_project, is_root=True, start_time=started - timedelta(minutes=1))
            if (r.extra or {}).get("metadata", {}).get("thread_id") == thread_id
        ]
        if {"datastory.analysis.plan", "datastory.analysis.approve", "datastory.chat.ask"} <= {r.name for r in roots}:
            break
        time.sleep(3)
    if not roots:
        print(f"В проекте «{settings.langsmith_project}» трассы сессии {thread_id} НЕ найдены за {WAIT_SECONDS} с.")
        return False
    print(f"\nНайдено корневых трасс в LangSmith (проект «{settings.langsmith_project}», thread_id={thread_id}):")
    for root in sorted(roots, key=lambda r: r.start_time):
        children = list(client.list_runs(project_name=settings.langsmith_project, trace_id=root.trace_id))
        kinds = sorted({c.name for c in children if c.name.startswith(("mcp.", "rag.", "llm.", "analysis_plan", "insight."))})
        print(f"  • {root.name}: {len(children)} run-ов, статус {root.status}")
        print(f"    {', '.join(kinds)}")
        print(f"    {client.get_run_url(run=root, project_name=settings.langsmith_project)}")
    return True


def main() -> int:
    status = tracing_status(get_settings())
    print(f"LangSmith: {status.label}. {status.reason}")
    try:
        get_llm()
    except LLMUnavailableError as exc:
        print(f"Сценарий не запущен: {exc.user_message}")
        return 3
    if not SALES_FILE.exists():
        print(f"Нет файла {SALES_FILE}. Создайте его: python scripts/generate_sales_demo.py")
        return 3

    started = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        thread_id = run_scenario(Path(tmp))
    if not status.enabled:
        print("\nТрассировка не включена: в LangSmith ничего не отправлено. Задайте LANGSMITH_TRACING=true и LANGSMITH_API_KEY (см. LANGSMITH_SETUP.md).")
        return 2
    return 0 if verify_traces(thread_id, started) else 1


if __name__ == "__main__":
    raise SystemExit(main())
