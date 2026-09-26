"""Правила планирования и поиск бизнес-контекста без MCP: чистые функции."""
from types import SimpleNamespace

from datastory.analytics.models import ColumnSummary, DatasetSummary
from datastory.rag.models import ContextHit, ContextSearchResult, SourceKind, SourceReference
from datastory.workflow.catalog import BY_ID, PIE_MAX_CATEGORIES, evaluate_candidates, fit_chart_type
from datastory.workflow.context import BusinessContext, retrieve_business_context
from datastory.workflow.models import ApprovalDecision, ChartDraft, PlanDraft
from datastory.workflow.planning import apply_decision, finalize_plan, resolve_mapping, rules_draft


def summary(columns: dict[str, tuple[str, int]], available=None) -> DatasetSummary:
    """columns: имя -> (тип, число уникальных значений)."""
    kinds = {n: k for n, (k, _) in columns.items()}
    return DatasetSummary(
        dataset_id="d", filename="t.csv", row_count=10, column_count=len(columns), column_names=list(columns), column_types=kinds,
        columns=[ColumnSummary(name=n, kind=k, dtype="x", unique_count=u) for n, (k, u) in columns.items()],
        numeric_measures=[n for n, k in kinds.items() if k == "numeric"],
        dimensions=[n for n, k in kinds.items() if k in ("categorical", "datetime")],
        available_metrics=available if available is not None else ["transaction_count", "success_rate", "transaction_volume"],
    )


PAYMENTS = summary({"Month": ("datetime", 6), "Channel": ("categorical", 3), "Transactions": ("numeric", 18), "Successful": ("numeric", 18), "Amount_KZT": ("numeric", 18)})


def plan_for(draft: PlanDraft, data=PAYMENTS, selected=None, choices=None, resolver=None):
    candidates = evaluate_candidates(data)
    selected = selected or [c.id for c in candidates]
    return finalize_plan("d", draft, "llm", data, candidates, selected, BusinessContext(), choices, resolver)


# ================================================================== доступность
def test_candidates_report_why_an_analysis_is_unavailable():
    data = summary({"Month": ("datetime", 6), "Transactions": ("numeric", 6), "Successful": ("numeric", 6)}, available=["transaction_count", "success_rate"])
    found = {c.id: c for c in evaluate_candidates(data)}
    assert found["success_dynamics"].available and found["success_dynamics"].dimension_candidates == ["Month"]
    assert "нет колонки «Amount_KZT»" in found["volume_dynamics"].reasons[0]
    assert "нет категориальной колонки" in found["channel_distribution"].reasons[0]


def test_dynamics_need_a_time_column_and_categories_a_reasonable_number_of_values():
    no_time = summary({"Channel": ("categorical", 3), "Transactions": ("numeric", 3), "Successful": ("numeric", 3)}, available=["transaction_count", "success_rate"])
    assert "нет колонки с датой или периодом" in evaluate_candidates(no_time)[0].reasons
    too_many = summary({"Month": ("datetime", 6), "City": ("categorical", 40), "Transactions": ("numeric", 3)}, available=["transaction_count"])
    assert not {c.id: c for c in evaluate_candidates(too_many)}["channel_distribution"].available


# ================================================================== правила визуализации
def test_line_for_time_pie_only_for_few_categories():
    assert fit_chart_type(BY_ID["success_dynamics"], "line", None) == ("line", "")
    chart, note = fit_chart_type(BY_ID["success_dynamics"], "pie", None)
    assert chart == "line" and "не применяется к динамике" in note
    assert fit_chart_type(BY_ID["channel_distribution"], "pie", PIE_MAX_CATEGORIES) == ("pie", "")
    chart, note = fit_chart_type(BY_ID["channel_distribution"], "pie", PIE_MAX_CATEGORIES + 1)
    assert chart == "bar" and "не более чем для" in note
    assert fit_chart_type(BY_ID["channel_distribution"], None, 3)[0] == "pie"
    assert fit_chart_type(BY_ID["channel_distribution"], "line", 3)[0] == "pie"


def test_every_chart_has_a_single_metric_so_units_are_never_mixed_on_an_axis():
    candidates = evaluate_candidates(PAYMENTS)
    plan = plan_for(rules_draft(candidates, [c.id for c in candidates]))
    assert all(isinstance(c.metric, str) for c in plan.charts) and len(plan.charts) <= 4


def test_pie_becomes_bar_when_the_category_column_has_many_values():
    data = summary({"Month": ("datetime", 6), "Channel": ("categorical", 9), "Transactions": ("numeric", 9)}, available=["transaction_count"])
    draft = PlanDraft(goal="", charts=[ChartDraft(analysis="channel_distribution", title="Каналы", chart_type="pie", x_column="Channel", rationale="")])
    plan = plan_for(draft, data)
    chart = next(c for c in plan.charts if c.analysis == "channel_distribution")
    assert chart.chart_type == "bar" and any("круговая диаграмма" in n for n in plan.notes)


# ================================================================== сопоставление колонок
def test_one_candidate_is_confident_several_need_agreement_or_the_user():
    kind = BY_ID["count_dynamics"]
    assert resolve_mapping(kind, ["Month"], None, None, None).status == "confident"

    ask = resolve_mapping(kind, ["Month", "Created"], "Month", None, None)
    assert ask.status == "ambiguous" and ask.column is None and "Модель предложила «Month»" in ask.basis

    agree = SimpleNamespace(status="confident", chosen=SimpleNamespace(column_name="Month"))
    assert resolve_mapping(kind, ["Month", "Created"], "Month", None, lambda phrase: agree).status == "confident"
    disagree = SimpleNamespace(status="confident", chosen=SimpleNamespace(column_name="Created"))
    assert resolve_mapping(kind, ["Month", "Created"], "Month", None, lambda phrase: disagree).status == "ambiguous"
    unsure = SimpleNamespace(status="ambiguous", chosen=None)
    assert resolve_mapping(kind, ["Month", "Created"], "Month", None, lambda phrase: unsure).status == "ambiguous"

    chosen = resolve_mapping(kind, ["Month", "Created"], None, "Created", None)
    assert (chosen.status, chosen.column) == ("confirmed", "Created")
    assert resolve_mapping(kind, ["Month", "Created"], None, "Amount", None).status == "ambiguous"  # чужая колонка не принимается


def test_rules_never_guess_between_several_columns():
    data = summary({"Month": ("datetime", 6), "Created": ("datetime", 6), "Transactions": ("numeric", 6)}, available=["transaction_count"])
    candidates = evaluate_candidates(data)
    plan = plan_for(rules_draft(candidates, ["count_dynamics"]), data, ["count_dynamics"])
    assert plan.charts[0].mapping.status == "ambiguous" and plan.clarifications[0].candidates == ["Month", "Created"]


# ================================================================== решение пользователя
def test_decision_confirms_formulas_only_for_the_chosen_metrics():
    selected = ["count_dynamics", "success_dynamics"]
    plan = plan_for(rules_draft(evaluate_candidates(PAYMENTS), selected), selected=selected)
    both = ApprovalDecision(action="approve", approved_chart_ids=selected, confirmed_metrics=["success_rate"])
    outcome = apply_decision(plan, both, PAYMENTS)
    assert outcome.error is None and [c.chart_id for c in outcome.plan.charts] == ["success_dynamics"]
    assert [e.analysis for e in outcome.plan.excluded] == ["count_dynamics"]
    assert outcome.plan.metrics[0].formula_confirmed_by_user

    none = apply_decision(plan, both.model_copy(update={"confirmed_metrics": []}), PAYMENTS)
    assert none.error and "не подтверждено" in none.error and len(none.plan.charts) == 2  # план не искажается ошибкой


# ================================================================== бизнес-контекст (RAG)
def hit(text, relevant=True, chunk="c1") -> ContextHit:
    source = SourceReference(kind=SourceKind.DOCUMENT, chunk_id=chunk, filename="doc.pdf", page=1, section="Раздел")
    return ContextHit(rank=1, score=0.9, relevant=relevant, text=text, page=1, source=source)


def search_returning(*hits):
    return lambda query: ContextSearchResult(query=query, provider="test", hits=list(hits), found=bool(hits))


def test_a_fragment_is_a_definition_only_if_it_names_the_metric():
    context = retrieve_business_context(["success_dynamics"], search_returning(hit("Погода в Астане в марте была тёплой.")))
    assert "success_rate" not in context.definitions  # релевантный по близости, но о другом: определения нет

    context = retrieve_business_context(["success_dynamics"], search_returning(hit("Success Rate — доля успешных транзакций.")))
    assert context.definitions["success_rate"].citation == "doc.pdf, стр. 1, раздел «Раздел»"


def test_irrelevant_hits_are_not_used():
    context = retrieve_business_context(["success_dynamics"], search_returning(hit("Success Rate — доля успешных.", relevant=False)))
    assert context.definitions == {} and context.events == []


def test_events_exclude_definition_fragments_and_are_limited():
    hits = [hit("Success Rate — доля", chunk="def")] + [hit(f"Событие {i}", chunk=f"e{i}") for i in range(5)]
    context = retrieve_business_context(["success_dynamics"], search_returning(*hits))
    assert "def" not in {e.chunk_id for e in context.events} and len(context.events) == 2


def test_rag_failure_is_reported_not_raised():
    def broken(query):
        raise RuntimeError("индекс повреждён")

    context = retrieve_business_context(["success_dynamics"], broken)
    assert "База знаний недоступна" in context.error and context.definitions == {}
