"""AI Chat по готовому дашборду на демонстрационном датасете: MCP для расчётов, RAG для контекста, границы возможностей."""
import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from datastory.analytics.formatting import format_metric_value
from datastory.chat import classify
from datastory.chat.classify import classify_by_rules, detect_metric, has_period_filter
from datastory.chat.graph import ChatService, strip_fact_ids
from datastory.chat.models import SUPPORTED_QUESTIONS, ChatAnswerDraft, Citation, QuestionPlan
from datastory.errors import LLMError
from datastory.insights.generator import CAUSE_UNKNOWN, NO_DEFINITION
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.mcp_client.connection import McpConnection, McpServerConfig
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.skills import load_skill
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.methodology import methodology_prompt
from datastory.workflow.runner import AnalysisRunner
from scripts import generate_demo_data as gen
from tests.helpers import ScriptedLLM
from tests.workflow_helpers import approve

ROWS = gen.build_rows()
DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
CHAT_DIR = Path(__file__).resolve().parent.parent / "datastory" / "chat"
Q_DEFINITION, Q_CHANGE, Q_EXTREME, Q_TOP, Q_NOTABLE, Q_EVENTS = SUPPORTED_QUESTIONS


def rate(rows) -> float:
    return sum(r["Successful"] for r in rows) / sum(r["Transactions"] for r in rows) * 100


def by(key, value) -> list:
    return [r for r in ROWS if r[key] == value]


def run_analysis(client, dataset_id, kb):
    runner = AnalysisRunner(WorkflowDeps.of(client, None, kb))
    snapshot = runner.start(dataset_id)
    return runner.resume(snapshot.thread_id, approve(snapshot)).result


@pytest.fixture
def kb_with_docs():
    kb = KnowledgeBase()
    kb.index_document(DEMO_PDF, "business_metrics.pdf")
    return kb


@pytest.fixture
def demo(client, datasets, kb_with_docs):
    result = run_analysis(client, datasets["demo"], kb_with_docs)
    return SimpleNamespace(result=result, client=client, kb=kb_with_docs, chat=ChatService(WorkflowDeps.of(client, None, kb_with_docs)))


def chat_with(demo, llm) -> ChatService:
    return ChatService(WorkflowDeps.of(demo.client, llm, demo.kb))


# ================================================================== шесть типов вопросов (без LLM)
def test_definition_question_uses_rag_and_names_the_source(demo):
    demo.client.calls.clear()
    answer = demo.chat.ask(Q_DEFINITION, demo.result)
    assert answer.intent == "metric_definition" and answer.supported and answer.tools == ["RAG search_business_context"]
    assert "business_metrics.pdf" in answer.context_source and "Success Rate" in answer.answer
    assert "SUM(Successful) / SUM(Transactions) × 100" in answer.answer  # формула из MCP-определений
    assert demo.client.calls == []  # для определения расчёт не нужен


def test_change_question_uses_dashboard_numbers_and_says_the_cause_is_unknown(demo):
    demo.client.calls.clear()
    answer = demo.chat.ask("Как изменился Success Rate?", demo.result)
    assert answer.intent == "metric_change" and answer.metrics == ["success_rate"] and answer.tools == []
    assert format_metric_value(rate(by("Month", "2026-01")), "%") in answer.answer
    assert format_metric_value(rate(by("Month", "2026-06")), "%") in answer.answer
    assert CAUSE_UNKNOWN in answer.limitation and answer.evidence
    assert demo.client.calls == []  # расчёт уже был на дашборде: MCP не вызывался повторно
    assert "результат дашборда" in answer.data_sources[0]


def test_change_without_a_named_metric_covers_every_dashboard_metric(demo):
    answer = demo.chat.ask(Q_CHANGE, demo.result)
    assert answer.metrics == ["transaction_count", "transaction_volume", "success_rate"]
    assert all(label in answer.answer for label in ("Количество транзакций", "Объём транзакций", "Success Rate"))


def test_extreme_period_question(demo):
    answer = demo.chat.ask(Q_EXTREME, demo.result)
    assert answer.intent == "extreme_period"
    march, june = rate(by("Month", "2026-03")), rate(by("Month", "2026-06"))
    assert f"максимум — {format_metric_value(june, '%')} (период 2026-06)" in answer.answer
    assert f"минимум — {format_metric_value(march, '%')} (период 2026-03)" in answer.answer


def test_top_channel_question(demo):
    demo.client.calls.clear()
    answer = demo.chat.ask(Q_TOP, demo.result)
    mobile = sum(r["Transactions"] for r in by("Channel", "Mobile"))
    assert answer.intent == "top_category" and "«Mobile»" in answer.answer and format_metric_value(mobile, "шт.") in answer.answer
    assert answer.answer.index("Mobile") < answer.answer.index("Web") < answer.answer.index("API")
    assert demo.client.calls == []  # распределение по каналам уже на дашборде


def test_notable_changes_question_points_at_the_march_drop(demo):
    answer = demo.chat.ask(Q_NOTABLE, demo.result)
    drop = rate(by("Month", "2026-03")) - rate(by("Month", "2026-02"))
    assert answer.intent == "notable_changes" and "2026-02 → 2026-03" in answer.answer
    assert f"{abs(drop):.2f}".replace(".", ",") + " п.п." in answer.answer
    assert CAUSE_UNKNOWN in answer.limitation


def test_events_question_uses_rag_and_cites_the_document(demo):
    demo.client.calls.clear()
    answer = demo.chat.ask(Q_EVENTS, demo.result)
    assert answer.intent == "context_events" and "RAG search_business_context" in answer.tools
    assert "плановое обновление инфраструктуры" in answer.answer and "стр. 2" in answer.context_source
    assert "не доказывает причинную связь" in answer.answer
    assert demo.client.calls == []
    for causal in ("вызвано обновлением", "из-за обновления"):
        assert causal not in answer.answer


def test_cause_question_admits_the_cause_is_unknown(demo):
    answer = demo.chat.ask("Почему упала успешность?", demo.result)
    assert answer.intent == "cause_question" and "не определяется" in answer.answer
    assert "остаётся гипотезой" in answer.answer and CAUSE_UNKNOWN in answer.limitation


# ================================================================== новый расчёт → MCP
def test_a_metric_missing_from_the_dashboard_is_calculated_by_mcp(demo):
    demo.client.calls.clear()
    answer = demo.chat.ask("Какой канал имеет наибольший объём транзакций?", demo.result)
    web = sum(r["Amount_KZT"] for r in by("Channel", "Web"))
    mobile = sum(r["Amount_KZT"] for r in by("Channel", "Mobile"))
    assert demo.client.calls == ["calculate_metrics"] and answer.tools == ["MCP calculate_metrics"]
    assert "«Mobile»" in answer.answer and format_metric_value(mobile, "KZT") in answer.answer and format_metric_value(web, "KZT") in answer.answer
    assert answer.data_sources[0].startswith("MCP calculate_metrics")


def test_a_new_time_series_is_calculated_by_mcp_too(demo):
    demo.client.calls.clear()
    answer = demo.chat.ask("Как изменилась средняя сумма транзакции?", demo.result)
    assert answer.metrics == ["average_transaction_amount"] and demo.client.calls == ["calculate_metrics"]
    first = by("Month", "2026-01")
    expected = sum(r["Amount_KZT"] for r in first) / sum(r["Transactions"] for r in first)
    assert format_metric_value(expected, "KZT") in answer.answer


def test_definition_of_a_metric_outside_the_dashboard_comes_from_rag(demo):
    answer = demo.chat.ask("Как считается средняя сумма транзакции?", demo.result)
    assert answer.metrics == ["average_transaction_amount"] and "Average Transaction Amount" in answer.answer
    assert "SUM(Amount_KZT) / SUM(Transactions)" in answer.answer


# ================================================================== границы MVP
@pytest.mark.parametrize(
    "question",
    ["Сделай прогноз на июль", "Напиши python код для регрессии", "Удали строки с ошибками", "Какая погода завтра?", "Как изменился объём в марте?", "Какой канал лидирует в январе?"],
)
def test_out_of_scope_questions_get_an_explanation_and_no_tools(demo, question):
    demo.client.calls.clear()
    answer = demo.chat.ask(question, demo.result)
    assert not answer.supported and answer.intent == "unsupported" and answer.tools == [] and answer.evidence == []
    assert all(q in answer.answer for q in SUPPORTED_QUESTIONS)  # что чат умеет, перечислено
    assert demo.client.calls == []


def test_period_filters_are_reported_as_a_limitation(demo):
    answer = demo.chat.ask("Как изменился объём в марте?", demo.result)
    assert "фильтром по месяцу" in answer.answer


def test_a_question_without_a_metric_and_dashboard_series_asks_for_clarification(demo):
    plan = demo.result.plan.model_copy(update={"charts": [c for c in demo.result.plan.charts if c.mapping.role == "category"]})
    bare = demo.result.model_copy(update={"plan": plan})
    answer = demo.chat.ask(Q_CHANGE, bare)
    assert answer.supported and answer.intent == "metric_change" and "Уточните показатель" in answer.answer and "Success Rate" in answer.answer


def test_empty_and_overlong_questions_are_rejected_politely(demo):
    assert "Введите вопрос" in demo.chat.ask("   ", demo.result).answer
    assert "слишком длинный" in demo.chat.ask("а" * 600, demo.result).answer


def test_no_documentation_is_reported_instead_of_invented(client, datasets):
    empty_kb = KnowledgeBase()
    result = run_analysis(client, datasets["demo"], empty_kb)
    chat = ChatService(WorkflowDeps.of(client, None, empty_kb))
    definition = chat.ask("Что означает Success Rate?", result)
    assert "в документации не найдено" in definition.answer and definition.context_source is None
    assert NO_DEFINITION.format(formula="SUM(Successful) / SUM(Transactions) × 100") in definition.limitation
    events = chat.ask(Q_EVENTS, result)
    assert "события не найдены" in events.answer and events.context_source is None


def test_unavailable_mcp_server_gives_an_error_answer_not_a_crash(client, datasets, kb_with_docs, demo):
    broken = AnalyticsMcpClient(McpConnection(McpServerConfig(command="python", args=("-m", "no_such_module_xyz"), name="datastory-analytics"), startup_timeout=10))
    try:
        chat = ChatService(WorkflowDeps.of(broken, None, kb_with_docs))
        answer = chat.ask("Как изменилась средняя сумма транзакции?", demo.result)  # нужен новый расчёт
        assert answer.error and "Не удалось выполнить расчёт" in answer.answer and "Не удалось подключиться к MCP-серверу" in answer.answer
    finally:
        broken.close()


def test_chat_has_no_code_generation_or_execution():
    forbidden_names = {"eval", "exec", "compile", "__import__"}  # re.compile — это метод, он допустим
    forbidden_methods = {"system", "popen", "Popen", "run", "eval", "exec"}
    forbidden_modules = {"subprocess", "os", "pandas", "code", "runpy"}
    for path in CHAT_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        methods = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not names & forbidden_names, (path.name, names & forbidden_names)
        assert not methods & forbidden_methods, (path.name, methods & forbidden_methods)
        assert not {m.split(".")[0] for m in modules} & forbidden_modules, path.name


# ================================================================== разбор вопросов
@pytest.mark.parametrize(
    "question, intent",
    [
        ("Что означает Success Rate?", "metric_definition"), ("Как считается объём транзакций?", "metric_definition"),
        ("Как изменилось количество транзакций?", "metric_change"), ("Какая динамика успешности?", "metric_change"),
        ("Какой месяц имеет максимальное значение?", "extreme_period"), ("В какой месяц было минимальное значение?", "extreme_period"),
        ("Какой канал имеет наибольшее количество операций?", "top_category"), ("Какая категория лидирует?", "top_category"),
        ("Какие изменения заслуживают внимания?", "notable_changes"), ("Есть ли аномалии?", "notable_changes"),
        ("Какие события упоминаются в документации?", "context_events"), ("Почему упала успешность?", "cause_question"),
        ("Сделай прогноз", "unsupported"), ("Привет", "unsupported"),
    ],
)
def test_rule_based_intents(question, intent):
    assert classify_by_rules(question)[0] == intent


def test_metric_is_recognized_only_if_available():
    available = ["transaction_count", "success_rate", "transaction_volume"]
    assert detect_metric("Как изменилась успешность?", available) == "success_rate"
    assert detect_metric("Каков объём?", available) == "transaction_volume"
    assert detect_metric("Сколько операций?", available) == "transaction_count"
    assert detect_metric("Какая средняя сумма?", available) is None  # такого показателя в датасете нет
    assert detect_metric("Что происходит?", available) is None


def test_period_filter_detection():
    assert has_period_filter("Как изменился объём в марте?") and has_period_filter("Показатель за 2026-03")
    assert not has_period_filter("Какой месяц имеет максимальное значение?")
    assert classify.FILTER_SENSITIVE == ("metric_change", "extreme_period", "top_category")


def test_service_ids_are_removed_from_the_user_text():
    assert strip_fact_ids("Рост на 34 470 (transaction_count.change, transaction_count.change_pct).") == "Рост на 34 470."


# ================================================================== LLM: структурные ответы и проверка
def fact_lines(user: str) -> dict[str, str]:
    return {m.group(1): m.group(3).strip() for m in re.finditer(r"^([\w.]+) \| (.*) \| (.*)$", user, re.MULTILINE)}


def scripted(answer, classify_as=None, **kwargs) -> ScriptedLLM:
    def plan(system, user, n):
        return classify_as or QuestionPlan(intent="metric_change", metric="success_rate", comment="Изменение успешности.")

    return ScriptedLLM(QuestionPlan=plan, ChatAnswerDraft=answer)


def change_answer(system, user, n, extra=""):
    facts = fact_lines(user)
    return ChatAnswerDraft(
        answer=f"Success Rate был {facts['success_rate.first']}, стал {facts['success_rate.last']}: изменение {facts['success_rate.change']}.{extra}",
        evidence_ids=["success_rate.first", "success_rate.last", "success_rate.change"], limitation=None, citations=[],
    )


def test_llm_classifies_the_question_and_writes_a_checked_answer(demo):
    llm = scripted(change_answer)
    answer = chat_with(demo, llm).ask("Как изменилась успешность?", demo.result)
    assert answer.generated_by == "llm" and answer.classified_by == "llm" and answer.violations == []
    assert format_metric_value(rate(by("Month", "2026-06")), "%") in answer.answer
    assert [e.id for e in answer.evidence] == ["success_rate.first", "success_rate.last", "success_rate.change"]
    assert CAUSE_UNKNOWN in answer.limitation  # обязательное ограничение добавляет код, а не модель


def test_the_skill_methodology_is_sent_to_the_chat_stage(demo):
    llm = scripted(change_answer)
    chat_with(demo, llm).ask("Как изменилась успешность?", demo.result)
    skill = load_skill("datastory-analysis")
    system = llm.prompts("ChatAnswerDraft")[0][0]
    assert methodology_prompt(skill, "chat") in system and "**A3.**" in system and "**V4.**" not in system
    assert "datastory-analysis" not in llm.prompts("QuestionPlan")[0][0]  # разбор вопроса методику не получает


def test_invented_numbers_are_rejected_then_regenerated_with_feedback(demo):
    def answer(system, user, n):
        return change_answer(system, user, n, " Успешность выросла до 99,9%." if n == 1 else "")

    llm = scripted(answer)
    result = chat_with(demo, llm).ask("Как изменилась успешность?", demo.result)
    assert result.generated_by == "llm" and "99,9" not in result.answer
    assert any("[numbers-grounded]" in v and "правило методики A1" in v for v in result.violations)
    assert any("Предыдущий ответ отклонён" in u and "99,9" in u for _, u in llm.prompts("ChatAnswerDraft"))


def test_persistently_invalid_answers_fall_back_to_calculated_text(demo):
    llm = scripted(lambda s, u, n: change_answer(s, u, n, " Успешность выросла до 99,9%."))
    result = chat_with(demo, llm).ask("Как изменилась успешность?", demo.result)
    assert result.generated_by == "rules" and "99,9" not in result.answer and result.violations
    assert len(llm.prompts("ChatAnswerDraft")) == 2


def test_causal_claims_are_rejected(demo):
    def answer(system, user, n):
        return change_answer(system, user, n, " Снижение вызвано обновлением инфраструктуры.")

    result = chat_with(demo, scripted(answer)).ask("Как изменилась успешность?", demo.result)
    assert result.generated_by == "rules" and any("[no-causal-claim]" in v for v in result.violations)
    assert "вызвано" not in result.answer


def test_a_metric_invented_by_the_llm_is_dropped(demo):
    llm = scripted(change_answer, QuestionPlan(intent="metric_change", metric="profit_margin", comment="Маржа."))
    answer = chat_with(demo, llm).ask(Q_CHANGE, demo.result)  # показатель не назван: отвечаем по всем показателям дашборда
    assert answer.metrics == ["transaction_count", "transaction_volume", "success_rate"]


def test_llm_failures_fall_back_to_rules(demo):
    def failing(system, user, n):
        return LLMError("Превышен лимит запросов OpenAI. Повторите позже.")

    llm = ScriptedLLM(QuestionPlan=failing, ChatAnswerDraft=failing)
    answer = chat_with(demo, llm).ask(Q_EXTREME, demo.result)
    assert answer.classified_by == "rules" and answer.generated_by == "rules" and answer.intent == "extreme_period"
    assert "2026-06" in answer.answer


def test_llm_can_mark_a_question_as_unsupported(demo):
    llm = scripted(change_answer, QuestionPlan(intent="unsupported", metric=None, comment="Вопрос о погоде не относится к дашборду."))
    answer = chat_with(demo, llm).ask("Какая завтра погода?", demo.result)
    assert not answer.supported and "Вопрос о погоде" in answer.answer and llm.prompts("ChatAnswerDraft") == []


def test_llm_period_filters_are_caught_even_if_the_model_misses_them(demo):
    answer = chat_with(demo, scripted(change_answer)).ask("Как изменилась успешность в марте?", demo.result)
    assert not answer.supported and "фильтром по месяцу" in answer.answer


def test_document_quotes_must_be_verbatim(demo):
    hits = demo.kb.search_business_context("плановое обновление инфраструктуры")
    event = next(h for h in hits.hits if "плановое обновление" in h.text)

    def cited(quote):
        def answer(system, user, n):
            return ChatAnswerDraft(
                answer="Документ упоминает плановое обновление инфраструктуры.", evidence_ids=[], limitation=None,
                citations=[Citation(chunk_id=event.source.chunk_id, quote=quote)],
            )

        return answer

    ask = lambda quote: chat_with(demo, scripted(cited(quote), QuestionPlan(intent="context_events", metric=None, comment=""))).ask(Q_EVENTS, demo.result)  # noqa: E731
    good = ask("плановое обновление инфраструктуры обработки транзакций")
    assert good.generated_by == "llm" and "стр. 2" in good.context_source
    bad = ask("обновление привело к падению на 40%")
    assert bad.generated_by == "rules" and any("[context-not-in-source]" in v for v in bad.violations)


# ================================================================== настоящая модель (тратит запросы OpenAI)
@pytest.mark.live
def test_live_llm_answers_the_supported_questions_on_the_demo_dataset(demo):
    from tests.helpers import live_openai

    chat = chat_with(demo, live_openai())
    for question in SUPPORTED_QUESTIONS:
        answer = chat.ask(question, demo.result)
        assert answer.supported and not answer.error and answer.answer, question
        assert answer.classified_by == "llm"
    assert not chat.ask("Сделай прогноз на июль", demo.result).supported
