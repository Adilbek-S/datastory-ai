"""Чат на странице готового дашборда: история в session_state, вызовы MCP и RAG, границы MVP."""
from datastory.chat.models import SUPPORTED_QUESTIONS
from datastory.ui.mcp import get_analytics_client
from tests.test_ui_workflow import DEMO_PDF, approved, button, everything, texts

Q_DEFINITION, Q_CHANGE, Q_EXTREME, Q_TOP, Q_NOTABLE, Q_EVENTS = SUPPORTED_QUESTIONS


def history_of(at) -> list[dict]:
    return at.session_state[f"chat::{at.session_state['analysis_session']['thread_id']}"]


def ask(at, question: str):
    at.chat_input[0].set_value(question).run()
    return at


def dashboard(pdf=DEMO_PDF):
    return approved(pdf=pdf)


def test_chat_appears_on_the_finished_dashboard_only():
    at = dashboard()
    assert not at.exception and len(at.chat_input) == 1
    assert "Чат по дашборду" in texts(at.markdown) and not at.chat_message
    assert any("Примеры вопросов" in e.label for e in at.expander)


def test_question_gets_an_answer_from_dashboard_numbers():
    at = ask(dashboard(), "Как изменился Success Rate?")
    assert not at.exception and not at.error
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    reply = texts(at.chat_message[1].markdown)
    assert "Success Rate" in reply and "97,66%" in reply and "97,88%" in reply
    assert "Причина изменения по данным не установлена" in texts(at.chat_message[1].caption)


def test_history_is_kept_in_session_state_across_questions_and_reruns():
    at = ask(dashboard(), Q_TOP)
    at = ask(at, Q_EXTREME)
    for _ in range(2):  # перерисовки не теряют историю
        at = at.run()
        assert not at.exception
    history = history_of(at)
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == Q_TOP and history[2]["content"] == Q_EXTREME
    assert len(at.chat_message) == 4 and "Mobile" in texts(at.chat_message[1].markdown)


def test_answers_that_reuse_the_dashboard_do_not_call_mcp_again():
    at = dashboard()
    client = get_analytics_client()
    before = list(client.calls)
    ask(at, Q_TOP)
    assert client.calls == before


def test_a_calculation_missing_from_the_dashboard_goes_to_mcp():
    at = dashboard()
    client = get_analytics_client()
    ask(at, "Какой канал имеет наибольший объём транзакций?")
    assert client.calls.count("calculate_metrics") == 5  # 4 расчёта дашборда и один новый
    assert "MCP calculate_metrics" in texts(at.chat_message[1].caption)


def test_events_are_answered_from_the_document_with_a_source():
    at = ask(dashboard(), Q_EVENTS)
    reply = texts(at.chat_message[1].markdown)
    assert "плановое обновление инфраструктуры" in reply and "business_metrics.pdf" in reply and "не доказывает причинную связь" in reply
    expander = [e for e in at.expander if "Числа, источники" in e.label][0]
    assert "стр. 2" in texts(expander.markdown)


def test_example_buttons_ask_the_prepared_questions():
    at = dashboard()
    at = button(at, Q_DEFINITION).click().run()
    assert history_of(at)[0]["content"] == Q_DEFINITION
    assert "Формула расчёта в системе" in texts(at.chat_message[1].markdown)


def test_out_of_scope_question_gets_a_clear_limitation():
    at = ask(dashboard(), "Сделай прогноз на июль")
    assert not at.exception
    assert "требует расчётов или действий, которых нет в текущем MVP" in texts(at.chat_message[1].markdown) + texts(at.info)
    assert Q_EVENTS in texts(at.info)  # перечислено, что чат умеет


def test_history_can_be_cleared():
    at = ask(dashboard(), Q_CHANGE)
    at = button(at, "Очистить историю чата").click().run()
    assert history_of(at) == [] and not at.chat_message


def test_a_new_analysis_session_starts_with_an_empty_chat():
    at = ask(dashboard(), Q_CHANGE)
    first = at.session_state["analysis_session"]["thread_id"]
    at = button(at, "Запустить анализ").click().run()
    second = at.session_state["analysis_session"]["thread_id"]
    assert first != second and not at.chat_message
    assert len(at.session_state[f"chat::{first}"]) == 2  # прежняя история сохранена под прежним thread_id


def test_chat_works_without_documentation_and_says_so():
    at = ask(dashboard(pdf=None), Q_EVENTS)
    assert "события не найдены" in texts(at.chat_message[1].markdown) and not at.exception
