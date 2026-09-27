"""План по запросу пользователя (набор без колонок платёжной системы): проверка шагов LLM по данным, фильтры, группировки, отказ."""
import pandas as pd
import pytest

from datastory.analytics.formatting import format_metric_value
from datastory.errors import LLMError
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.workflow.models import AUTO_GOAL, ApprovalDecision, FilterDraft, PlanDraft, StepDraft
from scripts import generate_sales_demo as sales
from tests.helpers import ScriptedLLM
from tests.workflow_helpers import approve, make_runner, save_dataset

ROWS = sales.build_rows()
AOV, RATE, REVENUE, ORDERS = "ratio:Revenue_KZT/Orders", "rate:Cancelled_Orders/Orders", "sum:Revenue_KZT", "sum:Orders"


@pytest.fixture
def sales_id(workspace, client):
    return save_dataset(workspace, pd.DataFrame(ROWS), "sales_2026.xlsx")


@pytest.fixture
def sales_kb(tmp_path):
    knowledge = KnowledgeBase()
    for path in sales.generate_all(tmp_path).values():
        if path.suffix == ".pdf":
            knowledge.index_document(path.read_bytes(), path.name)
    return knowledge


def step(metric=REVENUE, group_by=("Month",), chart_type="line", filters=(), title="Шаг", rationale="Потому что.") -> StepDraft:
    return StepDraft(title=title, metric=metric, group_by=list(group_by), filters=list(filters), chart_type=chart_type, rationale=rationale)


def planner(*steps, unsupported=(), ambiguous=False) -> ScriptedLLM:
    def plan(system, user, n):
        return PlanDraft(goal="Запрос", steps=list(steps), unsupported=list(unsupported), ambiguous=ambiguous)

    def no_insight(system, user, n):
        return LLMError("выводы по правилам")  # тексты выводов строят правила: проверяются числа расчётов

    return ScriptedLLM(PlanDraft=plan, InsightDraft=no_insight)


def plan_for(client, sales_id, llm, kb=None, query="Покажи что-нибудь"):
    return make_runner(client, llm, kb).start(sales_id, query)


# ================================================================== допустимые шаги
def test_valid_step_becomes_a_card_with_metric_grouping_and_chart(client, sales_id, sales_kb):
    snapshot = plan_for(client, sales_id, planner(step(AOV, ("Channel",), "bar", title="Средний чек по каналам")), sales_kb, "Какой средний чек по каналам?")
    plan = snapshot.request.plan
    card = plan.charts[0]
    assert (card.analysis, card.metric, card.group_by, card.x_column, card.chart_type) == ("custom", AOV, ["Channel"], "Channel", "bar")
    assert card.mapping.status == "confident" and card.filters == [] and plan.planner == "llm"
    definition = plan.metric_definition(AOV)
    assert definition.formula == "SUM(Revenue_KZT) / SUM(Orders)" and definition.documented  # определение AOV найдено в sales_metrics.pdf
    assert "sales_metrics.pdf" in definition.definition_source


def test_revenue_definition_is_found_in_the_documents(client, sales_id, sales_kb):
    plan = plan_for(client, sales_id, planner(step(REVENUE, ("Region",), "bar")), sales_kb).request.plan
    assert plan.metric_definition(REVENUE).documented and plan.documentation_gaps == []


def test_metric_without_a_document_definition_is_marked_undocumented(client, sales_id):
    plan = plan_for(client, sales_id, planner(step(REVENUE, ("Region",), "bar"))).request.plan  # база знаний пуста
    assert not plan.metric_definition(REVENUE).documented and plan.documentation_gaps


def test_the_planner_gets_the_metrics_dimensions_and_query_documents(client, sales_id, sales_kb):
    llm = planner(step(AOV, ("Channel",), "bar"))
    plan_for(client, sales_id, llm, sales_kb, "Какой средний чек по каналам?")
    system, user = llm.prompts("PlanDraft")[0]
    assert "ratio:Revenue_KZT/Orders | Revenue / Orders | SUM(Revenue_KZT) / SUM(Orders) | KZT" in user
    assert "Channel: категория, значения: Web, Mobile, Offline" in user and "Month: дата/период, от 2026-01" in user
    assert "sales_metrics.pdf" in user and "Average Order Value" in user  # определение найдено по запросу (RAG)
    assert "НЕ подменяй" in system and "Методика анализа" in system  # правила плана и Skill
    assert llm.prompts("IntentDraft") == []  # отдельного разбора намерения по каталогу нет


def test_no_group_by_means_dynamics_over_time(client, sales_id):
    card = plan_for(client, sales_id, planner(step(RATE, (), "bar"))).request.plan.charts[0]
    assert (card.group_by, card.chart_type, card.mapping.role) == (["Month"], "line", "time")


def test_time_and_category_give_a_multi_series_line(client, sales_id):
    card = plan_for(client, sales_id, planner(step(REVENUE, ("Category", "Month"), "bar"))).request.plan.charts[0]
    assert card.group_by == ["Month", "Category"] and card.x_column == "Month" and card.chart_type == "line"  # дата — ось X


# ================================================================== правила визуализации
@pytest.mark.parametrize(
    "metric, group, requested, expected",
    [
        (REVENUE, "Month", "pie", "line"),  # V4: круговая для динамики запрещена
        (REVENUE, "Region", "pie", "pie"),  # V3: доли аддитивного показателя по небольшому числу категорий
        (REVENUE, "Region", "line", "bar"),  # V2: сравнение категорий — столбцы
        (AOV, "Region", "pie", "bar"),  # отношение нельзя показывать долями целого
        (RATE, "Channel", "pie", "bar"),
    ],
)
def test_chart_type_follows_the_visualization_rules(client, sales_id, metric, group, requested, expected):
    card = plan_for(client, sales_id, planner(step(metric, (group,), requested))).request.plan.charts[0]
    assert card.chart_type == expected


def test_two_categories_are_reduced_to_one_with_a_note(client, sales_id):
    plan = plan_for(client, sales_id, planner(step(REVENUE, ("Region", "Channel"), "bar", title="Регион и канал"))).request.plan
    assert plan.charts[0].group_by == ["Region"] and any("две категории" in n for n in plan.notes)


# ================================================================== фильтры
def test_filter_values_are_matched_to_the_spelling_in_the_data(client, sales_id):
    filters = [FilterDraft(column="channel", op="in", values=["web", "MOBILE"])]
    card = plan_for(client, sales_id, planner(step(REVENUE, ("Channel",), "bar", filters))).request.plan.charts[0]
    assert [(f.column, f.op.value, f.value) for f in card.filters] == [("Channel", "in", ["Web", "Mobile"])]


def test_period_filter_is_accepted_and_a_malformed_one_rejected(client, sales_id):
    march = [FilterDraft(column="Month", op="eq", values=["2026-03"])]
    card = plan_for(client, sales_id, planner(step(REVENUE, ("Region",), "bar", march))).request.plan.charts[0]
    assert card.filters[0].value == "2026-03"
    bad = [FilterDraft(column="Month", op="eq", values=["март"])]
    plan = plan_for(client, sales_id, planner(step(REVENUE, ("Region",), "bar", bad))).plan
    assert plan.charts == [] and "не похоже на период" in plan.excluded[0].reason


def test_a_filter_value_that_does_not_exist_rejects_the_step(client, sales_id):
    filters = [FilterDraft(column="Channel", op="eq", values=["Online"])]
    snapshot = plan_for(client, sales_id, planner(step(REVENUE, ("Region",), "bar", filters)))
    assert snapshot.phase == "empty" and "«Online» нет в колонке «Channel» (есть: Web, Mobile, Offline)" in snapshot.plan.excluded[0].reason


def test_a_filter_on_a_numeric_or_missing_column_is_rejected(client, sales_id):
    filters = [FilterDraft(column="Orders", op="eq", values=["10"])]
    plan = plan_for(client, sales_id, planner(step(REVENUE, ("Region",), "bar", filters))).plan
    assert plan.charts == [] and "такой колонки-измерения нет" in plan.excluded[0].reason


# ================================================================== запросы, которые нельзя выполнить
def test_unknown_metric_is_never_invented(client, sales_id):
    snapshot = plan_for(client, sales_id, planner(step("ebitda", ("Month",)), unsupported=["EBITDA"]), None, "Покажи EBITDA")
    plan = snapshot.plan
    assert snapshot.phase == "empty" and plan.charts == [] and plan.metrics == []
    assert "показатель «ebitda» отсутствует" in plan.excluded[0].reason
    assert "EBITDA" in plan.unsupported and any("ebitda" in u for u in plan.unsupported)


def test_a_model_that_only_reports_unsupported_gives_an_empty_plan(client, sales_id):
    snapshot = plan_for(client, sales_id, planner(unsupported=["прогноз выручки"]), None, "Сделай прогноз")
    assert snapshot.phase == "empty" and snapshot.plan.unsupported == ["прогноз выручки"]


def test_unknown_group_column_rejects_the_step(client, sales_id):
    plan = plan_for(client, sales_id, planner(step(REVENUE, ("Store",), "bar"))).plan
    assert plan.charts == [] and "колонка группировки «Store» не найдена" in plan.excluded[0].reason


def test_ambiguous_grouping_is_a_question_to_the_user(client, sales_id):
    runner = make_runner(client, planner(step(REVENUE, (), "bar", title="Выручка по группам"), ambiguous=True))
    snapshot = runner.start(sales_id, "Покажи выручку по группам")
    card = snapshot.request.plan.charts[0]
    assert card.mapping.status == "ambiguous" and card.mapping.candidates == ["Month", "Region", "Channel", "Category"] and card.x_column is None
    again = runner.resume(snapshot.thread_id, approve(snapshot))
    assert again.phase == "awaiting_approval" and "Уточните колонку" in again.request.notice
    done = runner.resume(snapshot.thread_id, approve(snapshot, column_choices={"custom-1": "Region"}))
    assert done.phase == "completed" and done.result.metrics[0].group_by == ["Region"]


def test_at_most_four_steps_are_kept(client, sales_id):
    many = [step(REVENUE, (col,), "bar", title=f"Шаг {i}") for i, col in enumerate(("Region", "Channel", "Category", "Month", "Region"))]
    plan = plan_for(client, sales_id, planner(*many)).request.plan
    assert len(plan.charts) <= 4


def test_duplicate_steps_are_dropped(client, sales_id):
    plan = plan_for(client, sales_id, planner(step(REVENUE, ("Region",), "bar"), step(REVENUE, ("Region",), "bar", title="Ещё раз"))).request.plan
    assert len(plan.charts) == 1


# ================================================================== выполнение и выводы
def test_filtered_and_multi_series_steps_run_through_mcp_with_verified_insights(client, sales_id, sales_kb):
    web_mobile = [FilterDraft(column="Channel", op="in", values=["Web", "Mobile"])]
    llm = planner(
        step(REVENUE, ("Channel",), "bar", web_mobile, title="Web и Mobile"),
        step(REVENUE, ("Month", "Category"), "line", title="Рост по категориям"),
    )
    runner = make_runner(client, llm, sales_kb)
    snapshot = runner.start(sales_id, "Сравни выручку Web и Mobile и покажи рост категорий")
    client.calls.clear()
    done = runner.resume(snapshot.thread_id, approve(snapshot))
    assert done.phase == "completed" and client.calls.count("calculate_metrics") == 2
    channels, categories = done.result.metrics
    assert [(f.column, f.value) for f in channels.filters] == [("Channel", ["Web", "Mobile"])]
    assert {r.group["Channel"]: r.value for r in channels.rows} == {
        ch: sum(r["Revenue_KZT"] for r in ROWS if r["Channel"] == ch) for ch in ("Web", "Mobile")
    }
    assert categories.group_by == ["Month", "Category"] and len(categories.rows) == 24
    assert all(c.passed for c in done.result.insight_checks)

    web_text = done.result.insights[0].data_source
    assert "фильтры: Channel in ['Web', 'Mobile']" in web_text

    def growth(category):
        first = sum(r["Revenue_KZT"] for r in ROWS if r["Category"] == category and r["Month"] == "2026-01")
        last = sum(r["Revenue_KZT"] for r in ROWS if r["Category"] == category and r["Month"] == "2026-06")
        return (last - first) / first * 100

    leader = max(sales.CATEGORIES, key=growth)
    text = done.result.insights[1].text
    assert leader == "Electronics" and f"«{leader}»" in text.split(".")[0]
    assert f"{growth(leader):.1f}".replace(".", ",") + "%" in text


def test_category_rules_text_uses_only_calculated_numbers(client, sales_id):
    runner = make_runner(client, planner(step(AOV, ("Channel",), "bar", title="Средний чек")))
    snapshot = runner.start(sales_id, "Какой средний чек по каналам?")
    result = runner.resume(snapshot.thread_id, approve(snapshot)).result
    web = sum(r["Revenue_KZT"] for r in ROWS if r["Channel"] == "Web") / sum(r["Orders"] for r in ROWS if r["Channel"] == "Web")
    values = {r.group["Channel"]: r.value for r in result.metrics[0].rows}
    assert values["Web"] == pytest.approx(web) and result.metrics[0].unit == "KZT"
    assert all(c.passed for c in result.insight_checks)
    text = result.insights[0].text
    assert "доля" not in text  # у среднего чека долей нет: сравниваются значения
    top, bottom = max(values, key=values.get), min(values, key=values.get)
    assert f"наибольшее значение у «{top}» — {format_metric_value(values[top], 'KZT')}" in text
    assert f"Наименьшее — у «{bottom}»: {format_metric_value(values[bottom], 'KZT')}" in text


# ================================================================== резервные режимы
def test_without_a_model_the_catalog_plan_is_used(client, sales_id):
    plan = make_runner(client).start(sales_id, "Какой средний чек по каналам?").request.plan
    assert plan.planner == "rules" and {c.analysis for c in plan.charts} <= {"measure_dynamics", "measure_comparison"}


def test_automatic_goal_uses_the_catalog_even_with_a_model(client, sales_id):
    llm = planner(step(REVENUE, ("Region",), "bar"))
    plan_for(client, sales_id, llm, None, AUTO_GOAL)
    assert "измерения" not in llm.prompts("PlanDraft")[0][1].lower() or "Запрос пользователя" in llm.prompts("PlanDraft")[0][1]
    assert "Доступные показатели (id |" not in llm.prompts("PlanDraft")[0][1]  # автоанализ идёт по каталогу


# ================================================================== защита от подмены показателя и правила запроса
def named(term, **kwargs) -> StepDraft:
    return step(**kwargs).model_copy(update={"requested_term": term})


def test_a_documented_but_unavailable_metric_is_not_replaced_by_a_similar_one(client, sales_id, sales_kb):
    """Return Rate описан в документе, но колонок для него нет; доля отмен — другой показатель."""
    snapshot = plan_for(client, sales_id, planner(named("Return Rate", metric=RATE, group_by=("Category",), chart_type="bar")), sales_kb)
    assert snapshot.phase == "empty" and "«Return Rate» описан в документах" in snapshot.plan.excluded[0].reason
    assert "sales_metrics.pdf" in snapshot.plan.excluded[0].reason and snapshot.plan.charts == []


def test_a_term_that_matches_its_definition_is_accepted(client, sales_id, sales_kb):
    for term, metric in (("cancellation rate", RATE), ("Average Order Value", AOV), ("выручка", REVENUE)):
        plan = plan_for(client, sales_id, planner(named(term, metric=metric, group_by=("Channel",), chart_type="bar")), sales_kb).request.plan
        assert plan.charts[0].metric == metric, term


def test_a_term_absent_from_the_documents_cannot_be_checked_and_is_accepted(client, sales_id, sales_kb):
    plan = plan_for(client, sales_id, planner(named("оборот", metric=REVENUE, group_by=("Region",), chart_type="bar")), sales_kb).request.plan
    assert plan.charts[0].metric == REVENUE


def test_a_period_outside_the_data_is_rejected_as_a_forecast(client, sales_id):
    filters = [FilterDraft(column="Month", op="in", values=["2026-07"])]
    snapshot = plan_for(client, sales_id, planner(step(REVENUE, ("Month",), "line", filters)), None, "Покажи выручку на июль")
    assert snapshot.phase == "empty" and "вне диапазона данных (2026-01 — 2026-06)" in snapshot.plan.excluded[0].reason


def test_growth_words_add_the_time_axis(client, sales_id):
    card = plan_for(client, sales_id, planner(step(REVENUE, ("Category",), "bar")), None, "Какая категория показала наибольший рост?").request.plan.charts[0]
    assert card.group_by == ["Month", "Category"] and card.chart_type == "line"


def test_no_time_axis_is_added_to_a_plain_comparison(client, sales_id):
    card = plan_for(client, sales_id, planner(step(REVENUE, ("Region",), "bar")), None, "Сравни регионы по выручке").request.plan.charts[0]
    assert card.group_by == ["Region"]


def test_vague_grouping_words_turn_into_a_question(client, sales_id):
    card = plan_for(client, sales_id, planner(step(REVENUE, ("Category",), "bar")), None, "Покажи выручку по группам").request.plan.charts[0]
    assert card.mapping.status == "ambiguous" and card.group_by == []


def test_placeholder_metrics_are_skipped_silently(client, sales_id):
    snapshot = plan_for(client, sales_id, planner(step("unsupported", ("Month",)), unsupported=["Return Rate"]), None, "Покажи Return Rate")
    assert snapshot.phase == "empty" and snapshot.plan.excluded == [] and snapshot.plan.unsupported == ["Return Rate"]


# ================================================================== устойчивость планировщика
def test_an_empty_model_answer_is_retried_once_with_a_reminder(client, sales_id):
    def plan(system, user, n):
        if n == 1:
            return PlanDraft(goal="x", steps=[], unsupported=[])  # пустой ответ
        return PlanDraft(goal="x", steps=[step(REVENUE, ("Month",))], unsupported=[])

    llm = ScriptedLLM(PlanDraft=plan)
    snapshot = plan_for(client, sales_id, llm, None, "Покажи динамику выручки по месяцам")
    assert snapshot.phase == "awaiting_approval" and len(llm.prompts("PlanDraft")) == 2
    assert "Предыдущий ответ был пустым" in llm.prompts("PlanDraft")[1][1]


def test_a_persistently_empty_answer_is_reported_not_left_silent(client, sales_id):
    llm = ScriptedLLM(PlanDraft=lambda s, u, n: PlanDraft(goal="x", steps=[], unsupported=[]))
    snapshot = plan_for(client, sales_id, llm, None, "Покажи что-то")
    assert snapshot.phase == "empty" and snapshot.plan.unsupported == ["не удалось составить план по запросу: назовите показатель и группировку точнее"]
    assert len(llm.prompts("PlanDraft")) == 2


def test_the_prompt_forbids_substituting_related_financial_metrics(client, sales_id):
    llm = planner(step(REVENUE, ("Region",)))
    plan_for(client, sales_id, llm, None, "Покажи прибыль по регионам")
    system = llm.prompts("PlanDraft")[0][0]
    assert "выручка (Revenue) — не прибыль" in system and "шаг не строится" in system
