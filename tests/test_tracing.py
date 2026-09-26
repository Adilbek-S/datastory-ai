"""Трассировка LangSmith: конфигурация из окружения и span-ы workflow, LLM, RAG, MCP, чата и Vision.

Span-ы собирает клиент-рекордер в памяти (tests/tracing_fakes.py): тесты доказывают, что код отправляет нужные события с нужной
вложенностью и метаданными. Доставку в облако LangSmith они не проверяют — для этого нужен реальный ключ.
"""
import os

import pytest
from langsmith import tracing_context

from datastory.chat.graph import ChatService
from datastory.chat.models import ChatAnswerDraft, QuestionPlan
from datastory.config import Settings, get_settings
from datastory.file_processing.image_loader import recognize_table
from datastory.llm.client import OpenAIStructuredLLM
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.mcp_client.connection import McpConnection, McpServerConfig
from datastory.observability import clip, configure_tracing, span, trace_config, tracing_status
from datastory.rag.embeddings import OpenAIEmbeddings
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.models import AUTO_GOAL
from datastory.workflow.runner import AnalysisRunner
from scripts import generate_demo_data as gen
from tests.tracing_fakes import FakeChat, RecordingClient
from tests.vision_fakes import screenshot_extraction
from tests.workflow_helpers import approve, good_insight, plan_all

DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
SCREENSHOT = (gen.DEFAULT_OUT / "transactions_screenshot.png").read_bytes()
NODES_BEFORE_APPROVAL = {"analyze_intent", "retrieve_context", "build_analysis_plan", "validate_analysis_plan", "human_approval"}
NODES_AFTER_APPROVAL = {"execute_analytics", "generate_insights", "verify_insights", "prepare_dashboard"}


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


@pytest.fixture
def recorder():
    return RecordingClient()


def fake_llm(**extra) -> OpenAIStructuredLLM:
    responders = {"PlanDraft": plan_all, "InsightDraft": good_insight, **extra}
    return OpenAIStructuredLLM("test", "gpt-4o-mini", chat=FakeChat(responders))


@pytest.fixture
def kb():
    knowledge = KnowledgeBase()
    knowledge.index_document(DEMO_PDF, "business_metrics.pdf")
    return knowledge


def analyze(client, dataset_id, llm, kb, recorder):
    runner = AnalysisRunner(WorkflowDeps.of(client, llm, kb))
    with tracing_context(enabled=True, client=recorder):
        snapshot = runner.start(dataset_id, AUTO_GOAL)
        done = runner.resume(snapshot.thread_id, approve(snapshot))
    return runner, snapshot, done


# ================================================================== конфигурация из переменных окружения
def test_tracing_needs_both_the_flag_and_a_key():
    on = tracing_status(settings(langsmith_tracing=True, langsmith_api_key="lsv2_x", langsmith_project="demo"))
    assert on.enabled and on.requested and on.label == "Включена" and "demo" in on.reason

    no_key = tracing_status(settings(langsmith_tracing=True))
    assert not no_key.enabled and no_key.requested and no_key.label == "Нет ключа"
    assert "LANGSMITH_API_KEY не задан" in no_key.reason and "НЕ включена" in no_key.reason

    off = tracing_status(settings(langsmith_api_key="lsv2_x"))
    assert not off.enabled and not off.requested and off.label == "Выключена"


def test_settings_are_read_from_environment_variables(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "from-env")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
    loaded = Settings(_env_file=None)
    assert loaded.langsmith_tracing and loaded.langsmith_project == "from-env"
    assert tracing_status(loaded).endpoint == "https://eu.api.smith.langchain.com"


def test_enabled_tracing_exports_the_variables_langsmith_reads(monkeypatch):
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT", "LANGSMITH_WORKSPACE_ID"):
        monkeypatch.setenv(name, "")  # значения вернутся после теста
    configure_tracing(settings(langsmith_tracing=True, langsmith_api_key="lsv2_x", langsmith_project="demo", langsmith_workspace_id="ws-1"))
    assert os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_API_KEY"] == "lsv2_x"
    assert os.environ["LANGSMITH_PROJECT"] == "demo" and os.environ["LANGSMITH_WORKSPACE_ID"] == "ws-1"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"


def test_missing_key_switches_tracing_off_and_warns_instead_of_pretending(monkeypatch, caplog):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    with caplog.at_level("WARNING", logger="datastory.tracing"):
        status = configure_tracing(settings(langsmith_tracing=True))
    assert not status.enabled and os.environ["LANGSMITH_TRACING"] == "false"  # LangChain не пытается отправлять трассы
    assert any("трассировка НЕ включена" in r.getMessage() for r in caplog.records)


def test_get_settings_applies_the_tracing_configuration(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    get_settings.cache_clear()
    get_settings()
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_trace_config_names_the_run_and_groups_by_thread():
    config = trace_config("datastory.analysis.plan", thread_id="analysis-1", dataset_id="d1", tags=("analysis",), metadata={"goal": "x"})
    assert config["run_name"] == "datastory.analysis.plan" and config["tags"] == ["datastory", "analysis"]
    assert config["configurable"] == {"thread_id": "analysis-1"}
    assert config["metadata"] == {"goal": "x", "thread_id": "analysis-1", "dataset_id": "d1"}
    assert clip("а" * 400).endswith("…") and len(clip("а" * 400)) == 301


# ================================================================== workflow: узлы, план, выводы
def test_workflow_graph_nodes_are_traced_under_named_roots(client, datasets, kb, recorder):
    runner, snapshot, _ = analyze(client, datasets["demo"], fake_llm(), kb, recorder)
    plan_root = recorder.named("datastory.analysis.plan")[0]
    build_root = recorder.named("datastory.analysis.approve")[0]
    assert plan_root["metadata"]["thread_id"] == build_root["metadata"]["thread_id"] == snapshot.thread_id  # одна сессия — один thread
    assert plan_root["metadata"]["dataset_id"] == datasets["demo"] and "analysis" in plan_root["tags"]
    for name in NODES_BEFORE_APPROVAL:
        assert recorder.ancestors(recorder.named(name)[0])[-1] == "datastory.analysis.plan", name
    for name in NODES_AFTER_APPROVAL:
        assert recorder.ancestors(recorder.named(name)[0])[-1] == "datastory.analysis.approve", name


def test_analysis_plan_generation_has_its_own_span_and_llm_run(client, datasets, kb, recorder):
    analyze(client, datasets["demo"], fake_llm(), kb, recorder)
    span_run = recorder.named("analysis_plan.generate")[0]
    assert "build_analysis_plan" in recorder.ancestors(span_run)
    assert span_run["outputs"]["planner"] == "llm" and len(span_run["outputs"]["steps"]) == 4
    assert span_run["inputs"]["goal"].startswith("Найди наиболее значимые тенденции")
    llm_run = recorder.named("llm.PlanDraft")[0]
    assert "analysis_plan.generate" in recorder.ancestors(llm_run) and "llm" in llm_run["tags"]
    assert llm_run["metadata"]["schema"] == "PlanDraft" and llm_run["metadata"]["ls_provider"] == "openai"


def test_insight_generation_and_verification_are_traced_per_chart(client, datasets, kb, recorder):
    analyze(client, datasets["demo"], fake_llm(), kb, recorder)
    generated, verified = recorder.named("insight.generate"), recorder.named("insight.verify")
    assert len(generated) == 4 and len(verified) == 4 and len(recorder.named("llm.InsightDraft")) == 4
    assert {r["metadata"]["chart_id"] for r in generated} == {"count_dynamics", "volume_dynamics", "success_dynamics", "channel_distribution"}
    assert all(r["outputs"]["generated_by"] == "llm" and r["outputs"]["evidence"] >= 1 for r in generated)
    assert all(r["outputs"]["passed"] for r in verified)
    assert all("insight.generate" in recorder.ancestors(r) for r in recorder.named("llm.InsightDraft"))


def test_rules_mode_still_traces_the_plan_and_insight_steps(client, datasets, kb, recorder):
    analyze(client, datasets["demo"], None, kb, recorder)
    assert recorder.named("analysis_plan.generate")[0]["outputs"]["planner"] == "rules"
    assert {r["outputs"]["generated_by"] for r in recorder.named("insight.generate")} == {"rules"}
    assert not recorder.named("llm.PlanDraft")


# ================================================================== MCP: инструмент, длительность, успех/ошибка
def test_every_mcp_call_is_a_tool_span_with_name_duration_and_status(client, datasets, kb, recorder):
    analyze(client, datasets["demo"], None, kb, recorder)
    calls = [r for r in recorder.records.values() if r["name"].startswith("mcp.")]
    assert sorted(r["name"] for r in calls) == ["mcp.calculate_metrics"] * 4 + ["mcp.create_chart_spec"] * 4 + ["mcp.profile_dataset"]
    for run in calls:
        assert run["run_type"] == "tool" and "mcp" in run["tags"] and run["error"] is None
        assert run["metadata"]["mcp.tool"] == run["name"].removeprefix("mcp.") and run["metadata"]["mcp.server"] == "datastory-analytics"
        assert run["metadata"]["mcp.status"] == "success" and run["metadata"]["mcp.duration_ms"] >= 0
        assert run["outputs"]["status"] == "success" and run["outputs"]["duration_ms"] == run["metadata"]["mcp.duration_ms"]
        assert run["inputs"]["tool"] == run["metadata"]["mcp.tool"]


def test_mcp_spans_carry_summaries_not_table_data(client, datasets, kb, recorder):
    analyze(client, datasets["demo"], None, kb, recorder)
    metrics = next(r for r in recorder.named("mcp.calculate_metrics") if r["inputs"]["arguments"]["metric"] == "success_rate")
    assert metrics["inputs"]["arguments"]["group_by"] == ["Month"] and metrics["outputs"]["groups"] == 6
    chart = recorder.named("mcp.create_chart_spec")[0]
    assert set(chart["inputs"]["arguments"]) == {"chart_type", "title", "x_axis", "metric", "points"}  # без строк результата
    assert "rows" not in metrics["outputs"] and recorder.named("mcp.profile_dataset")[0]["outputs"]["rows"] == 18


def test_a_failed_mcp_call_is_marked_as_an_error_span(datasets, recorder):
    broken = AnalyticsMcpClient(McpConnection(McpServerConfig(command="python", args=("-m", "no_such_module_xyz"), name="datastory-analytics"), startup_timeout=10))
    try:
        runner = AnalysisRunner(WorkflowDeps.of(broken, None, None))
        with tracing_context(enabled=True, client=recorder):
            snapshot = runner.start(datasets["demo"])
    finally:
        broken.close()
    assert snapshot.phase == "failed"
    failed = recorder.named("mcp.profile_dataset")[0]
    assert failed["error"] and failed["metadata"]["mcp.status"] == "error" and failed["metadata"]["mcp.error_type"] == "McpConnectionError"
    assert failed["metadata"]["mcp.duration_ms"] >= 0 and failed["outputs"] is None
    assert recorder.named("datastory.analysis.plan")[0]["error"]  # корневой run тоже помечен ошибкой


# ================================================================== RAG
def test_rag_retrieval_is_a_retriever_span_with_documents_and_sources(client, datasets, kb, recorder):
    analyze(client, datasets["demo"], None, kb, recorder)
    searches = recorder.named("rag.search_business_context")
    assert len(searches) >= 4  # определения трёх показателей и сопроводительный контекст
    assert all(r["run_type"] == "retriever" and "rag" in r["tags"] for r in searches)
    first = searches[0]
    assert first["inputs"]["query"] and first["inputs"]["dataset_id"] == datasets["demo"]
    documents = first["outputs"]["documents"]
    assert documents and {"page_content", "metadata"} <= set(documents[0]) and "business_metrics.pdf" in documents[0]["metadata"]["source"]
    assert "retrieve_context" in recorder.ancestors(first)


def test_embedding_span_reports_counts_and_never_the_texts(recorder):
    class Client:
        class embeddings:  # noqa: N801 — имитация openai.OpenAI().embeddings
            @staticmethod
            def create(model, input):
                from types import SimpleNamespace

                return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.1, 0.2]) for i, _ in enumerate(input)])

    embedder = OpenAIEmbeddings("key", client=Client())
    with tracing_context(enabled=True, client=recorder):
        embedder.embed(["секретный текст документа", "второй"])
    run = recorder.named("openai.embeddings")[0]
    assert run["run_type"] == "embedding" and run["inputs"] == {"texts": 2} and run["outputs"] == {"vectors": 2}
    assert "секретный" not in str(recorder.records)


# ================================================================== чат и Vision
def chat_llm() -> OpenAIStructuredLLM:
    def plan(system, user, n):
        return QuestionPlan(intent="extreme_period", metric=None, comment="Максимум.")

    def answer(system, user, n):
        facts = {p.split(" | ")[0]: p.split(" | ")[2] for p in user.splitlines() if " | " in p}
        return ChatAnswerDraft(
            answer=f"Максимум Success Rate — {facts['success_rate.max']}.", evidence_ids=["success_rate.max"], limitation=None, citations=[],
        )

    return OpenAIStructuredLLM("test", "gpt-4o-mini", chat=FakeChat({"QuestionPlan": plan, "ChatAnswerDraft": answer}))


def test_chat_questions_are_traced_and_joined_to_the_analysis_thread(client, datasets, kb, recorder):
    runner, snapshot, done = analyze(client, datasets["demo"], None, kb, recorder)
    chat = ChatService(WorkflowDeps.of(client, chat_llm(), kb))
    with tracing_context(enabled=True, client=recorder):
        answer = chat.ask("Какой месяц имеет максимальное значение?", done.result, snapshot.thread_id)
    assert answer.generated_by == "llm"
    root = recorder.named("datastory.chat.ask")[0]
    assert root["metadata"]["thread_id"] == snapshot.thread_id and "chat" in root["tags"] and root["metadata"]["question"].startswith("Какой месяц")
    for node in ("classify_question", "gather_evidence", "compose_answer", "verify_answer"):
        assert "datastory.chat.ask" in recorder.ancestors(recorder.named(node)[0]), node
    assert {"llm.QuestionPlan", "llm.ChatAnswerDraft"} <= recorder.names()


def test_chat_new_calculation_is_an_mcp_span_inside_the_chat_trace(client, datasets, kb, recorder):
    _, snapshot, done = analyze(client, datasets["demo"], None, kb, recorder)
    recorder.records.clear()
    with tracing_context(enabled=True, client=recorder):
        ChatService(WorkflowDeps.of(client, None, kb)).ask("Какой канал имеет наибольший объём транзакций?", done.result)
    mcp = recorder.named("mcp.calculate_metrics")[0]
    assert "datastory.chat.ask" in recorder.ancestors(mcp) and mcp["metadata"]["mcp.status"] == "success"


def test_vision_recognition_is_traced_without_the_image(recorder):
    llm = OpenAIStructuredLLM("test", "gpt-4o-mini", chat=FakeChat({"TableExtraction": lambda s, u, n: screenshot_extraction()}))
    with tracing_context(enabled=True, client=recorder):
        table = recognize_table(SCREENSHOT, "transactions_screenshot.png", llm)
    assert len(table.frame) == 6
    assert recorder.names() == {"vision.recognize_table"}  # вызов модели с изображением исключён из трассы
    run = recorder.named("vision.recognize_table")[0]
    assert run["inputs"]["filename"] == "transactions_screenshot.png" and run["inputs"]["image_bytes"] == len(SCREENSHOT)
    assert run["outputs"]["rows"] == 6 and "base64" not in str(recorder.records) and "image_url" not in str(recorder.records)


# ================================================================== выключенная трассировка
def test_disabled_tracing_sends_nothing(client, datasets, kb, recorder):
    runner = AnalysisRunner(WorkflowDeps.of(client, fake_llm(), kb))
    with tracing_context(enabled=False, client=recorder):
        snapshot = runner.start(datasets["demo"])
        runner.resume(snapshot.thread_id, approve(snapshot))
    assert recorder.records == {}


def test_span_helper_records_exceptions_and_reraises(recorder):
    with tracing_context(enabled=True, client=recorder):
        with pytest.raises(ValueError):
            with span("failing", inputs={"a": 1}):
                raise ValueError("boom")
    assert recorder.named("failing")[0]["error"].endswith("ValueError: boom") or "boom" in recorder.named("failing")[0]["error"]


# ================================================================== интерфейс
def test_home_page_shows_the_tracing_status(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def page():
        from datastory.ui.views import home

        home.render()

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    get_settings.cache_clear()
    at = AppTest.from_function(page, default_timeout=60).run()
    body = "\n".join(m.value for m in at.markdown)
    assert not at.exception and "LangSmith</div><div class=\"value\">Нет ключа" in body
    assert "LANGSMITH_API_KEY не задан" in "\n".join(w.value for w in at.warning)

    monkeypatch.setenv("LANGSMITH_TRACING", "true")  # без ключа configure_tracing принудительно выставил false
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setattr("datastory.observability.configure_tracing", lambda s: tracing_status(s))  # без записи в os.environ
    get_settings.cache_clear()
    at = AppTest.from_function(page, default_timeout=60).run()
    assert "LangSmith</div><div class=\"value\">Включена" in "\n".join(m.value for m in at.markdown)
    get_settings.cache_clear()
