"""Специализированный маршрут LangGraph для чата по готовому дашборду.

    classify_question → gather_evidence → compose_answer → verify_answer
                     ↘ (вопрос вне MVP или нужно уточнение) → ответ без LLM

Те же зависимости, что у основного воркфлоу (WorkflowDeps): MCP-клиент, LLM, база знаний, Skill. Тип вопроса — один из
фиксированного набора; произвольный код не генерируется и не выполняется. Числа берутся из результатов MCP (готовых
или свежих), контекст — из RAG; текст проверяет тот же verifier, что и выводы дашборда.
"""
from __future__ import annotations

import logging
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from datastory.chat.classify import FILTER_SENSITIVE, PERIOD_FILTER_REASON, classify_by_rules, detect_metric, has_period_filter
from datastory.chat.evidence import Bundle, ChatDeps, answer_from_rules, chat_limitations, gather, sentences
from datastory.chat.prompts import CLASSIFY_SYSTEM, chat_prompt, chat_system, classify_prompt
from datastory.chat.models import MAX_QUESTION, SUPPORTED_QUESTIONS, ChatAnswer, ChatAnswerDraft, QuestionPlan
from datastory.errors import LLMError
from datastory.insights.generator import quote_is_verbatim
from datastory.insights.verifier import Violation, check_insight
from datastory.mcp_client.connection import McpConnectionError, McpToolError
from datastory.observability import clip, trace_config
from datastory.models import EvidenceType, Insight
from datastory.rag.api import search_business_context
from datastory.skills import SkillError
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.models import AnalysisResult

logger = logging.getLogger("datastory.chat")

MAX_ATTEMPTS = 2  # первая попытка LLM и одна повторная с замечаниями проверки


class ChatState(TypedDict, total=False):
    question: str
    result: AnalysisResult
    intent: str
    metric: str | None
    classified_by: str
    note: str
    bundle: Bundle
    draft: ChatAnswer
    issues: list[str]
    feedback: list[str]
    history: list[str]  # замечания всех отклонённых попыток
    attempt: int
    answer: ChatAnswer


def _unsupported(question: str, text: str, classified_by: str, supported: bool = False) -> ChatAnswer:
    return ChatAnswer(question=question, intent="unsupported", answer=text, supported=supported, classified_by=classified_by)


def limits_message(reason: str) -> str:
    listing = "\n".join(f"- {q}" for q in SUPPORTED_QUESTIONS)
    return f"{reason} Сейчас чат отвечает на вопросы такого рода:\n{listing}".strip()


FACT_IDS = re.compile(r"\s*\((?:[a-z_]+\.[a-z_0-9]+(?:,\s*)?)+\)")


def strip_fact_ids(text: str) -> str:
    """Служебные id доказательств «(success_rate.max)» в тексте для пользователя не нужны."""
    return FACT_IDS.sub("", text)


def strip_quotes(text: str, bundle: Bundle) -> str:
    """Дословные предложения из документа — документальный факт с источником; проверка чисел и причин к ним не применяется."""
    for fragment in bundle.fragments.values():
        for sentence in sorted(sentences(fragment.text), key=len, reverse=True):
            text = text.replace(sentence, " ")
    return text


def build_chat_graph(deps: WorkflowDeps):
    def classify_question(state: ChatState) -> ChatState:
        question, result = state["question"], state["result"]
        available = result.summary.available_metrics
        llm = deps.llm()
        intent, note, by = *classify_by_rules(question), "rules"
        metric = detect_metric(question, available)
        if llm is not None:
            try:
                plan = llm.generate(QuestionPlan, system=CLASSIFY_SYSTEM, user=classify_prompt(question, result))
                intent, note, by = plan.intent, plan.comment, "llm"
                metric = plan.metric if plan.metric in available else metric  # выдуманный показатель отбрасывается
            except LLMError as exc:
                logger.warning("Классификация вопроса по правилам: %s", exc.user_message)
        if intent in FILTER_SENSITIVE and has_period_filter(question):
            intent, note = "unsupported", PERIOD_FILTER_REASON  # чат не считает по фильтрам: честно сообщаем об ограничении
        return {"intent": intent, "metric": metric, "classified_by": by, "note": note, "attempt": 0, "history": []}

    def route_after_classification(state: ChatState) -> str:
        return "answer_unsupported" if state["intent"] == "unsupported" else "gather_evidence"

    def answer_unsupported(state: ChatState) -> ChatState:
        reason = state.get("note") or "Этот вопрос выходит за возможности текущего MVP."
        return {"answer": _unsupported(state["question"], limits_message(reason), state["classified_by"])}

    def gather_evidence(state: ChatState) -> ChatState:
        kb = deps.kb()
        result = state["result"]

        def search(query: str):
            return search_business_context(query, dataset_id=result.dataset_id, kb=kb)

        chat_deps = ChatDeps(client=deps.client(), search=search, result=result)
        bundle = gather(state["intent"], state["metric"], state["question"], chat_deps)  # MCP и RAG — только здесь
        return {"bundle": bundle}

    def route_after_gathering(state: ChatState) -> str:
        return "clarify" if state["bundle"].clarification else "compose_answer"

    def clarify(state: ChatState) -> ChatState:
        bundle: Bundle = state["bundle"]
        answer = _unsupported(state["question"], bundle.clarification, state["classified_by"], supported=True)
        return {"answer": answer.model_copy(update={"intent": state["intent"]})}

    def compose_answer(state: ChatState) -> ChatState:
        bundle: Bundle = state["bundle"]
        question, llm = state["question"], deps.llm()
        attempt = state.get("attempt", 0) + 1
        if llm is not None:
            try:
                system = chat_system(deps.skill())
                draft = llm.generate(ChatAnswerDraft, system=system, user=chat_prompt(question, bundle, state.get("feedback")))
                answer, found = assemble_chat(draft, question, bundle, state["classified_by"])
                return {"draft": answer, "issues": [str(v) for v in found], "attempt": attempt, "feedback": []}
            except (LLMError, SkillError) as exc:
                logger.warning("Ответ по правилам: %s", getattr(exc, "user_message", exc))
        return {"draft": answer_from_rules(question, bundle, state["classified_by"]), "issues": [], "attempt": attempt, "feedback": []}

    def verify_answer(state: ChatState) -> ChatState:
        bundle, draft = state["bundle"], state["draft"]
        found = state.get("issues", []) + [str(v) for v in verify_chat_answer(draft, bundle)]
        history = [*state.get("history", []), *found]
        if not found:
            return {"answer": draft.model_copy(update={"violations": history}), "feedback": [], "history": history}
        if draft.generated_by == "llm" and state["attempt"] < MAX_ATTEMPTS:
            return {"feedback": found, "history": history}
        fallback = answer_from_rules(state["question"], bundle, state["classified_by"]) if draft.generated_by == "llm" else draft
        return {"answer": fallback.model_copy(update={"violations": history}), "feedback": [], "history": history}

    def route_after_verification(state: ChatState) -> str:
        return "compose_answer" if state.get("feedback") else END

    graph = StateGraph(ChatState)
    for node in (classify_question, answer_unsupported, gather_evidence, clarify, compose_answer, verify_answer):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "classify_question")
    graph.add_conditional_edges("classify_question", route_after_classification, {"answer_unsupported": "answer_unsupported", "gather_evidence": "gather_evidence"})
    graph.add_edge("answer_unsupported", END)
    graph.add_conditional_edges("gather_evidence", route_after_gathering, {"clarify": "clarify", "compose_answer": "compose_answer"})
    graph.add_edge("clarify", END)
    graph.add_edge("compose_answer", "verify_answer")
    graph.add_conditional_edges("verify_answer", route_after_verification, {"compose_answer": "compose_answer", END: END})
    return graph.compile()


# --------------------------------------------------------------------------- LLM-ответ: сборка и проверка
def assemble_chat(draft: ChatAnswerDraft, question: str, bundle: Bundle, classified_by: str) -> tuple[ChatAnswer, list[Violation]]:
    """Ответ LLM → ChatAnswer: доказательства, источники и ограничения берутся из кода, а не из текста модели."""
    from datastory.insights.verifier import CONTEXT_NOT_IN_SOURCE, EVIDENCE_REQUIRED, UNKNOWN_EVIDENCE

    issues: list[Violation] = []
    facts = {f.id: f for f in bundle.facts}
    evidence = []
    for fact_id in dict.fromkeys(draft.evidence_ids):
        if fact_id in facts:
            evidence.append(facts[fact_id])
        elif fact_id not in bundle.fragments:  # chunk_id, попавший в evidence_ids по ошибке, — не выдумка; он учитывается ниже
            issues.append(Violation(UNKNOWN_EVIDENCE, f"доказательство {fact_id!r} не входит в набор фактов"))
    if not evidence and bundle.facts and not any(v.rule == UNKNOWN_EVIDENCE for v in issues):
        issues.append(Violation(EVIDENCE_REQUIRED, "не указано ни одного числового доказательства"))

    cited: list = []
    for citation in (draft.citations if bundle.fragments else []):  # документов в запросе не было: ссылки модели не на что проверять
        fragment = bundle.fragments.get(citation.chunk_id)
        if fragment is None:
            issues.append(Violation(CONTEXT_NOT_IN_SOURCE, f"фрагмент {citation.chunk_id!r} не был найден в базе знаний"))
        elif quote_is_verbatim(fragment, citation.quote.strip()):
            cited.append(fragment)
        else:
            issues.append(Violation(CONTEXT_NOT_IN_SOURCE, "цитата не найдена дословно в указанном фрагменте документа"))

    sources = [f for f in [*(x for x in bundle.definitions.values() if x), *cited] if f is not None]
    if bundle.intent == "context_events":
        sources = list(cited)
    own = [draft.limitation.strip()] if draft.limitation and draft.limitation.strip() else []
    limits = [*own, *chat_limitations(bundle, event_used=bool(cited) and bundle.intent != "metric_definition")]
    answer = ChatAnswer(
        question=question, intent=bundle.intent, answer=strip_fact_ids(draft.answer).strip(), metrics=bundle.metrics, evidence=evidence,
        data_sources=list(dict.fromkeys(bundle.data_sources)),
        context_source="; ".join(dict.fromkeys(f.citation for f in sources)) or None,
        limitation=" ".join(dict.fromkeys(limits)) or None, tools=list(dict.fromkeys(bundle.tools)),
        generated_by="llm", classified_by=classified_by,
    )
    return answer, issues


def verify_chat_answer(answer: ChatAnswer, bundle: Bundle) -> list[Violation]:
    """Те же проверки, что у выводов дашборда: числа, причины, рост/снижение, периоды."""
    needs_numbers = bool(bundle.facts)
    insight = Insight(
        title="", text=strip_quotes(answer.answer, bundle), evidence=answer.evidence, evidence_type=EvidenceType.COMPUTED,
        limitation=strip_quotes(answer.limitation or "", bundle) or None,
    )
    return check_insight(insight, bundle.facts, bundle.mask(), bundle.labels, require_evidence=needs_numbers)


class ChatService:
    """Чат по готовому дашборду: один вопрос — один проход по маршруту. История хранится интерфейсом (st.session_state)."""

    def __init__(self, deps: WorkflowDeps):
        self.graph = build_chat_graph(deps)

    def ask(self, question: str, result: AnalysisResult, thread_id: str | None = None) -> ChatAnswer:
        question = (question or "").strip()
        if not question:
            return _unsupported(question, "Введите вопрос.", "rules", supported=True)
        if len(question) > MAX_QUESTION:
            return _unsupported(question, f"Вопрос слишком длинный: не более {MAX_QUESTION} символов.", "rules", supported=True)
        try:
            config = trace_config(
                "datastory.chat.ask", thread_id=thread_id, dataset_id=result.dataset_id, tags=("chat",), metadata={"question": clip(question)}
            )
            return self.graph.invoke({"question": question, "result": result}, config)["answer"]
        except McpConnectionError as exc:
            return _error(question, f"Не удалось выполнить расчёт: {exc.user_message}")
        except McpToolError as exc:
            return _error(question, f"Инструмент {exc.tool} отклонил запрос: {exc.message}")
        except Exception:  # noqa: BLE001 — сбой чата не должен ронять страницу дашборда
            logger.exception("Сбой чата")
            return _error(question, "Внутренняя ошибка чата. Повторите вопрос.")


def _error(question: str, text: str) -> ChatAnswer:
    return ChatAnswer(question=question, intent="unsupported", answer=text, supported=True, error=True)
