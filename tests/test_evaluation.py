"""Evaluation pipeline: golden cases, эталонные расчёты, метрики, отчёт. Настоящие LLM и OpenAI здесь не вызываются."""
import json
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from datastory.evaluation import __main__ as cli
from datastory.evaluation import report, scoring
from datastory.evaluation.golden import GOLDEN_PATH, GROUPS, RESULTS_PATH, ExpectedFilter, ExpectedNumeric, ExpectedRow, dump_golden, file_sha256, load_golden
from datastory.evaluation.golden_builder import AOV, RATE, REVENUE, build_golden, reference
from datastory.evaluation.runner import Environment, predict, run_case, summarize
from datastory.llm.usage import UsageHandler, UsageRecorder
from datastory.rag.embeddings import OpenAIEmbeddings
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.models import FilterDraft, PlanDraft, StepDraft
from datastory.workflow.runner import AnalysisRunner
from scripts import generate_sales_demo as sales
from tests.helpers import ScriptedLLM
from tests.workflow_helpers import save_dataset

GOLDEN = load_golden()
ROWS = sales.build_rows()
BY_ID = {c.id: c for c in GOLDEN.cases}


# ================================================================== golden cases
def test_golden_set_has_thirty_cases_in_six_groups_of_five():
    assert len(GOLDEN.cases) == 30 and len({c.id for c in GOLDEN.cases}) == 30
    for group in GROUPS:
        assert sum(c.group == group for c in GOLDEN.cases) == 5, group


def test_every_case_has_the_required_fields():
    for case in GOLDEN.cases:
        assert case.id and case.user_query.strip() and case.expected_behavior in ("answer", "reject", "clarify")
        if case.expected_behavior == "answer":
            assert case.expected_metric and case.expected_columns and case.expected_group_by and case.expected_chart_type in ("line", "bar", "pie")
            assert case.expected_numeric is not None and case.expected_rag_document
        else:  # отказ и уточнение: ничего не выдумывается
            assert case.expected_metric is None and case.expected_numeric is None and case.expected_group_by == []


def test_invalid_group_covers_missing_documented_and_ambiguous_requests():
    invalid = {c.id: c for c in GOLDEN.cases if c.group == "invalid_ambiguous"}
    assert [c.expected_behavior for c in invalid.values()] == ["reject", "reject", "reject", "clarify", "reject"]
    assert "EBITDA" in invalid["iv-01"].user_query and invalid["iv-01"].expected_rag_document is None  # нет ни в данных, ни в базе знаний
    assert invalid["iv-02"].expected_rag_document == "sales_metrics.pdf"  # описан в документе, но данных нет


def test_golden_examples_from_the_task_are_present():
    queries = {c.user_query for c in GOLDEN.cases}
    for expected in ("Какой средний чек по каналам?", "Покажи динамику выручки по месяцам", "Какая категория показала наибольший рост выручки?",
                     "Сравни выручку Web и Mobile", "Покажи cancellation rate", "Покажи EBITDA"):
        assert expected in queries


def test_the_committed_golden_file_is_up_to_date():
    assert GOLDEN_PATH.read_text(encoding="utf-8") == dump_golden(build_golden())


def test_expected_columns_follow_the_expected_step():
    case = BY_ID["fl-03"]  # средний чек Electronics по каналам
    assert set(case.expected_columns) == {"Revenue_KZT", "Orders", "Channel", "Category"} and case.expected_filters[0].value == "Electronics"


# ================================================================== эталонный расчёт (чистый Python)
def test_reference_is_computed_from_raw_rows_sum_to_sum():
    rows = [
        {"g": "a", "n": 10, "d": 100, "c": 2}, {"g": "a", "n": 20, "d": 100, "c": 3}, {"g": "b", "n": 30, "d": 50, "c": 5},
    ]
    ratio = reference(rows, "ratio:n/d", ["g"], [])
    assert {r.group["g"]: r.value for r in ratio.rows} == {"a": 0.15, "b": 0.6} and ratio.overall == pytest.approx(60 / 250)  # не среднее долей
    rate = reference(rows, "rate:c/d", ["g"], [])
    assert {r.group["g"]: r.value for r in rate.rows} == {"a": 2.5, "b": 10.0}
    filtered = reference(rows, "sum:n", ["g"], [ExpectedFilter(column="g", op="in", value=["a"])])
    assert [(r.group, r.value) for r in filtered.rows] == [({"g": "a"}, 30.0)] and filtered.overall == 30.0


def test_reference_totals_match_the_generator():
    total = sum(r["Revenue_KZT"] for r in ROWS)
    assert BY_ID["ts-01"].expected_numeric.overall == total
    assert BY_ID["fl-01"].expected_numeric.overall == sum(r["Revenue_KZT"] for r in ROWS if r["Channel"] in ("Web", "Mobile"))
    assert BY_ID["rg-01"].expected_numeric.overall == pytest.approx(sum(r["Cancelled_Orders"] for r in ROWS) / sum(r["Orders"] for r in ROWS) * 100)


# ================================================================== Numeric Accuracy: MCP против Python на всех случаях
@pytest.fixture
def sales_dataset(workspace, client):
    return save_dataset(workspace, pd.DataFrame(ROWS), "sales_2026.xlsx")


def test_mcp_matches_the_python_reference_for_every_golden_case(client, sales_dataset):
    checked = 0
    for case in GOLDEN.cases:
        spec = case.expected_numeric
        if spec is None:
            continue
        result = client.calculate_metrics(sales_dataset, spec.metric, spec.group_by, [f.model_dump() for f in spec.filters] or None)
        verdict = scoring.score_numeric(spec, [{"group": r.group, "value": r.value} for r in result.rows], result.overall.value)
        assert verdict.ok, (case.id, verdict.detail)
        assert verdict.max_rel_error < 1e-9
        checked += 1
    assert checked == 25


# ================================================================== метрики
HITS = [
    {"document": "sales_metrics.pdf", "section": "4. Средний чек (Average Order Value, AOV)", "score": 0.7},
    {"document": "sales_channels_regions.pdf", "section": "1. Каналы продаж", "score": 0.4},
    {"document": "sales_calendar_2026.pdf", "section": "1. События 2026 года", "score": 0.3},
    {"document": "sales_metrics.pdf", "section": "2. Выручка (Revenue)", "score": 0.2},
]


def test_hit_at_3_looks_only_at_the_top_three():
    assert scoring.retrieval_hit(HITS, "sales_channels_regions.pdf") and scoring.retrieval_hit(HITS, "sales_calendar_2026.pdf")
    assert not scoring.retrieval_hit(HITS[:3], "sales_absent.pdf")
    assert not scoring.retrieval_hit([HITS[1], HITS[2], HITS[1], HITS[0]], "sales_metrics.pdf")  # нужный документ четвёртым не считается
    assert scoring.section_hit(HITS, "sales_metrics.pdf", "4. Средний чек") and not scoring.section_hit(HITS, "sales_metrics.pdf", "5. Доля отмен")


def predicted(**overrides):
    base = {"metric": AOV, "group_by": ["Channel"], "filters": [], "chart_type": "bar", "columns": ["Revenue_KZT", "Orders", "Channel"]}
    return {**base, **overrides}


def test_plan_accuracy_needs_metric_columns_and_grouping():
    case = BY_ID["mi-01"]
    assert scoring.score_plan(case, predicted()).accurate
    for wrong, part in (
        ({"metric": "sum:Revenue_KZT", "columns": ["Revenue_KZT", "Channel"]}, "metric_ok"),
        ({"columns": ["Revenue_KZT", "Orders", "Region"], "group_by": ["Channel"]}, "columns_ok"),
        ({"group_by": ["Region"], "columns": ["Revenue_KZT", "Orders", "Channel"]}, "grouping_ok"),
    ):
        verdict = scoring.score_plan(case, predicted(**wrong))
        assert not verdict.accurate and not getattr(verdict, part) and verdict.reasons


def test_chart_type_does_not_affect_plan_accuracy_but_is_reported():
    verdict = scoring.score_plan(BY_ID["mi-01"], predicted(chart_type="pie"))
    assert verdict.accurate and not verdict.chart_ok


def test_a_missing_plan_step_is_never_accurate():
    assert not scoring.score_plan(BY_ID["mi-01"], None).accurate
    assert not scoring.score_plan(BY_ID["mi-01"], predicted(metric=None)).accurate


def test_filters_compare_by_meaning_not_by_spelling():
    web_mobile = [ExpectedFilter(column="Channel", op="in", value=["Web", "Mobile"])]
    assert scoring.filters_match(web_mobile, [{"column": "Channel", "op": "in", "value": ["Mobile", "Web"]}])
    assert scoring.filters_match([ExpectedFilter(column="Region", op="eq", value="Almaty")], [{"column": "Region", "op": "in", "value": ["Almaty"]}])
    assert not scoring.filters_match(web_mobile, [{"column": "Channel", "op": "in", "value": ["Web"]}])
    assert not scoring.filters_match(web_mobile, [])


def numeric_case() -> ExpectedNumeric:
    return ExpectedNumeric(metric=REVENUE, group_by=["Region"], rows=[ExpectedRow(group={"Region": "A"}, value=100.0), ExpectedRow(group={"Region": "B"}, value=0.1 + 0.2)], overall=100.3)


def test_numeric_accuracy_tolerates_float_noise_only():
    expected = numeric_case()
    rows = [{"group": {"Region": "A"}, "value": 100.0}, {"group": {"Region": "B"}, "value": 0.30000000000000004}]
    assert scoring.score_numeric(expected, rows, 100.3).ok
    off = [{"group": {"Region": "A"}, "value": 100.001}, rows[1]]
    verdict = scoring.score_numeric(expected, off, 100.3)
    assert not verdict.ok and "получено 100.001" in verdict.detail


def test_numeric_accuracy_rejects_missing_extra_or_wrong_groups_and_totals():
    expected = numeric_case()
    only_a = [{"group": {"Region": "A"}, "value": 100.0}]
    assert "нет" in scoring.score_numeric(expected, only_a, 100.3).detail
    extra = [{"group": {"Region": "A"}, "value": 100.0}, {"group": {"Region": "B"}, "value": 0.3}, {"group": {"Region": "C"}, "value": 1.0}]
    assert not scoring.score_numeric(expected, extra, 100.3).ok
    ok_rows = [{"group": {"Region": "A"}, "value": 100.0}, {"group": {"Region": "B"}, "value": 0.3}]
    assert not scoring.score_numeric(expected, ok_rows, 999.0).ok and not scoring.score_numeric(expected, ok_rows, None).ok
    assert not scoring.score_numeric(expected, [{"group": {"Region": "A"}, "value": None}, ok_rows[1]], 100.3).ok


def test_percentiles_and_stats():
    assert scoring.percentile([10, 20, 30, 40], 50) == 25.0 and scoring.percentile([1, 2, 3, 4, 5], 95) == 4.8 and scoring.percentile([], 50) is None
    assert scoring.stats([100.0, 200.0, 300.0]) == {"mean": 200.0, "median": 200.0, "p95": 290.0, "max": 300.0}
    assert scoring.stats([]) == {"mean": None, "median": None, "p95": None, "max": None} and scoring.rate(1, 3) == 0.3333 and scoring.rate(0, 0) is None


# ================================================================== запуск случаев (LLM подменён сценарием, всё остальное настоящее)
def scripted_planner(step: StepDraft | None, unsupported=()) -> ScriptedLLM:
    def plan(system, user, n):
        return PlanDraft(goal="x", steps=[step] if step else [], unsupported=list(unsupported))

    return ScriptedLLM(PlanDraft=plan)


def step(metric, group_by, chart="bar", filters=()):
    return StepDraft(title="Шаг", metric=metric, group_by=group_by, filters=list(filters), chart_type=chart, rationale="r")


def make_env(client, dataset_id, llm, kb=None) -> Environment:
    kb = kb or KnowledgeBase()
    for name in sales.DOCS:
        kb.index_document((sales.DEMO_DIR / "sales_docs" / name).read_bytes(), name)
    usage = UsageRecorder()
    runner = AnalysisRunner(WorkflowDeps.of(client, llm, kb))
    return Environment(runner, client, kb, usage, dataset_id, "scripted" if llm else None, "" if llm else "не задан OPENAI_API_KEY",
                       SimpleNamespace(langsmith_tracing=False, langsmith_api_key=""), sales.OUT, [])


def test_an_answered_case_is_scored_on_retrieval_plan_and_numbers(client, sales_dataset):
    env = make_env(client, sales_dataset, scripted_planner(step(AOV, ["Channel"])))
    record = run_case(env, BY_ID["mi-01"])
    assert record["error"] is None and record["plan"]["accurate"] and record["plan"]["chart_ok"]
    assert record["numeric_golden_plan"]["ok"] and record["numeric_system_plan"]["ok"]
    assert record["retrieval"]["hit_at_3"] is True and len(record["retrieval"]["hits"]) == 3
    assert record["plan_latency_ms"] > 0 and record["tokens"]["llm_calls"] == 0  # сценарная LLM токенов не тратит


def test_a_wrong_plan_fails_plan_and_system_numbers_but_not_the_golden_plan_numbers(client, sales_dataset):
    env = make_env(client, sales_dataset, scripted_planner(step(REVENUE, ["Region"])))
    record = run_case(env, BY_ID["mi-01"])
    assert not record["plan"]["accurate"] and not record["plan"]["metric_ok"] and record["plan"]["reasons"]
    assert record["numeric_golden_plan"]["ok"] and not record["numeric_system_plan"]["ok"]  # MCP считает верно, а план — не тот


def test_filters_of_the_plan_are_scored(client, sales_dataset):
    good = step(REVENUE, ["Channel"], filters=[FilterDraft(column="Channel", op="in", values=["Web", "Mobile"])])
    record = run_case(make_env(client, sales_dataset, scripted_planner(good)), BY_ID["fl-01"])
    assert record["plan"]["accurate"] and record["plan"]["filters_ok"] and record["numeric_system_plan"]["ok"]
    record = run_case(make_env(client, sales_dataset, scripted_planner(step(REVENUE, ["Channel"]))), BY_ID["fl-01"])
    # по определению метрики (показатель, колонки, группировка) план «точен», но без фильтра значения фильтров и числа не совпадают
    assert record["plan"]["accurate"] and not record["plan"]["filters_ok"] and not record["plan"]["full_accurate"]
    assert record["numeric_golden_plan"]["ok"] and not record["numeric_system_plan"]["ok"]


def test_an_invalid_request_is_a_rejection_when_no_step_is_built(client, sales_dataset):
    env = make_env(client, sales_dataset, scripted_planner(None, unsupported=["EBITDA"]))
    record = run_case(env, BY_ID["iv-01"])
    assert record["plan"]["behavior_ok"] and record["plan"]["predicted"]["behavior"] == "reject" and record["plan"]["predicted"]["unsupported"] == ["EBITDA"]
    invented = run_case(make_env(client, sales_dataset, scripted_planner(step(REVENUE, ["Month"]))), BY_ID["iv-01"])
    assert not invented["plan"]["behavior_ok"] and invented["plan"]["predicted"]["behavior"] == "answer"  # выдуманный ответ засчитывается как провал


def test_an_ambiguous_request_is_a_clarification(client, sales_dataset):
    def plan(system, user, n):
        return PlanDraft(goal="x", steps=[step(REVENUE, [])], ambiguous=True)

    record = run_case(make_env(client, sales_dataset, ScriptedLLM(PlanDraft=plan)), BY_ID["iv-04"])
    assert record["plan"]["predicted"]["behavior"] == "clarify" and record["plan"]["behavior_ok"]


def test_without_an_llm_plan_metrics_are_not_computed_and_not_invented(client, sales_dataset):
    record = run_case(make_env(client, sales_dataset, None), BY_ID["mi-01"])
    assert record["plan"]["computed"] is False and "OPENAI_API_KEY" in record["plan"]["reason"]
    assert "numeric_system_plan" not in record and record["numeric_golden_plan"]["ok"] and "hit_at_3" in record["retrieval"]
    summary = summarize([record])["overall"]
    assert summary["plan_accuracy"] == {"hits": 0, "total": 0, "rate": None} and summary["numeric_accuracy"]["total"] == 1
    assert summary["numeric_accuracy_system_plan"]["total"] == 0 and summary["invalid_rejection_rate"]["rate"] is None


def test_a_failure_inside_a_case_is_recorded_and_counted_as_a_miss(client, sales_dataset):
    class Broken:
        def start(self, *a, **k):
            raise RuntimeError("сбой")

    env = make_env(client, sales_dataset, scripted_planner(None))
    env.runner = Broken()
    record = run_case(env, BY_ID["mi-01"])
    assert record["error"] == "RuntimeError: сбой"
    summary = summarize([record])["overall"]
    assert summary["errors"] == 1 and summary["plan_accuracy"] == {"hits": 0, "total": 1, "rate": 0.0}
    assert summary["numeric_accuracy_system_plan"]["total"] == 1 and summary["numeric_accuracy_system_plan"]["hits"] == 0


def test_summary_aggregates_by_group_with_latency_and_tokens(client, sales_dataset):
    env = make_env(client, sales_dataset, scripted_planner(step(REVENUE, ["Month"], "line")))
    records = [run_case(env, BY_ID[i]) for i in ("ts-01", "cc-01", "iv-01")]
    records[0]["tokens"] = {"llm_calls": 2, "input": 100, "output": 20, "total": 120}
    summary = summarize(records)
    assert summary["overall"]["cases"] == 3 and set(summary["by_group"]) == {"time_series", "category_comparison", "invalid_ambiguous"}
    assert summary["by_group"]["time_series"]["plan_accuracy"] == {"hits": 1, "total": 1, "rate": 1.0}
    assert summary["by_group"]["category_comparison"]["plan_accuracy"]["hits"] == 0  # групп ожидалась Region
    assert summary["overall"]["invalid_rejection_rate"] == {"hits": 0, "total": 1, "rate": 0.0}  # выдуманный ответ на EBITDA
    assert summary["overall"]["llm_tokens"]["total"] == 120 and summary["overall"]["plan_latency_ms"]["mean"] > 0


def test_predict_picks_the_first_confident_step(client, sales_dataset):
    runner = AnalysisRunner(WorkflowDeps.of(client, scripted_planner(step(AOV, ["Channel"])), KnowledgeBase()))
    prediction = predict(runner.start(sales_dataset, "Какой средний чек по каналам?"))
    assert (prediction["behavior"], prediction["metric"], prediction["group_by"], prediction["chart_type"]) == ("answer", AOV, ["Channel"], "bar")
    assert set(prediction["columns"]) == {"Revenue_KZT", "Orders", "Channel"}


# ================================================================== токены
def test_usage_handler_sums_the_tokens_reported_by_the_model():
    recorder = UsageRecorder()
    handler = UsageHandler(recorder)
    message = SimpleNamespace(usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150})
    handler.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=message)]]))
    handler.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=None))]]))
    usage = recorder.snapshot()
    assert (usage.calls, usage.input_tokens, usage.output_tokens, usage.total_tokens) == (2, 120, 30, 150)
    assert recorder.reset().total_tokens == 150 and recorder.snapshot().calls == 0


def test_embedding_tokens_are_counted_from_the_openai_response():
    class Client:
        class embeddings:  # noqa: N801
            @staticmethod
            def create(model, input):
                data = [SimpleNamespace(index=i, embedding=[0.1]) for i, _ in enumerate(input)]
                return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=7 * len(input)))

    embedder = OpenAIEmbeddings("key", client=Client())
    embedder.embed(["a", "b"])
    embedder.embed(["c"])
    assert embedder.tokens_used == 21


# ================================================================== отчёт и запись результатов
@pytest.fixture
def results(client, sales_dataset):
    env = make_env(client, sales_dataset, scripted_planner(step(REVENUE, ["Month"], "line")))
    records = [run_case(env, BY_ID[i]) for i in ("ts-01", "cc-01", "iv-01")]
    return {
        "schema_version": 1, "generated_at": "2026-09-26T12:00:00+00:00", "duration_s": 12.3,
        "scope": {"complete": False, "cases": 3, "total_cases": 30},
        "environment": {
            "llm_model": "scripted", "llm_available": True, "llm_reason": "", "embedding_provider": "offline-hash", "embedding_is_semantic": False,
            "dataset": "data/demo/sales_2026.xlsx", "dataset_sha256": "a" * 64, "documents": {"sales_metrics.pdf": "b" * 64}, "golden_sha256": "c" * 64,
            "top_k": 3, "float_tolerance": {"rel": 1e-9, "abs": 1e-6}, "git_commit": "d" * 40, "git_dirty": False, "python": "3.11",
        },
        "summary": summarize(records), "cases": records,
    }


def test_the_report_is_rendered_only_from_the_results(results):
    text = report.render_report(results, "f" * 64)
    assert text.startswith("# EVALS") and "f" * 64 in text and "частичный прогон" in text
    overall = results["summary"]["overall"]
    for key in ("plan_accuracy", "numeric_accuracy", "invalid_rejection_rate", "retrieval_hit_at_3"):
        block = overall[key]
        assert f"{block['rate'] * 100:.1f}% ({block['hits']}/{block['total']})" in text
    assert "не семантика" in text and all(case["id"] in text for case in results["cases"])
    percents = set(re.findall(r"(\d+\.\d)% \((\d+)/(\d+)\)", text))  # ни одного процента, которого нет в результатах
    known = {(f"{b['rate'] * 100:.1f}", str(b["hits"]), str(b["total"])) for blk in [overall, *results["summary"]["by_group"].values()] for b in blk.values() if isinstance(b, dict) and "rate" in b and b["total"]}
    assert percents <= known


def test_uncomputed_metrics_are_shown_as_dashes_with_the_reason(results):
    results["environment"].update(llm_model=None, llm_available=False, llm_reason="не задан OPENAI_API_KEY")
    for record in results["cases"]:
        record["llm_used"] = False
        record["plan"] = {"computed": False, "reason": "x"}
        record.pop("numeric_system_plan", None)
    results["summary"] = summarize(results["cases"])
    text = report.render_report(results, "0" * 64)
    assert "| Analysis Plan Accuracy | — (не вычислялась: не задан OPENAI_API_KEY" in text
    assert "Запросы к LLM не выполнялись" in text and "не использовалась (не задан OPENAI_API_KEY)" in text


def test_failures_are_listed_with_reasons(results):
    text = report.render_report(results, "0" * 64)
    assert "## Что не совпало" in text and "**cc-01**" in text and "группировка ['Month'], ожидалась ['Region']" in text
    assert "**iv-01**" in text and "поведение: система — answer, ожидалось — reject" in text


def test_outputs_are_written_together_and_the_report_records_the_file_hash(results, tmp_path):
    json_path, md_path = report.write_outputs(results, tmp_path / "results" / "latest.json", tmp_path / "EVALS.md", tmp_path / "history")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"] == json.loads(json.dumps(results["summary"]))
    assert file_sha256(json_path) in md_path.read_text(encoding="utf-8")


def test_the_report_is_not_created_without_real_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="сначала выполните реальный запуск"):
        report.write_report_from_file(tmp_path / "missing.json", tmp_path / "EVALS.md")
    assert not (tmp_path / "EVALS.md").exists()


def test_cli_report_only_requires_existing_results(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "write_report_from_file", lambda: report.write_report_from_file(tmp_path / "missing.json", tmp_path / "EVALS.md"))
    assert cli.main(["--report-only"]) == 2 and "сначала выполните реальный запуск" in capsys.readouterr().err


def test_a_partial_run_does_not_touch_latest_json_or_the_report(monkeypatch, tmp_path, results, capsys):
    monkeypatch.setattr(cli, "PARTIAL_PATH", tmp_path / "partial.json")
    monkeypatch.setattr(cli, "run_evals", lambda **kwargs: results)
    monkeypatch.setattr(cli, "write_outputs", lambda *a, **k: pytest.fail("частичный прогон не должен писать latest.json и EVALS.md"))
    assert cli.main(["--groups", "filters"]) == 0
    assert (tmp_path / "partial.json").exists() and "Частичный прогон" in capsys.readouterr().out


def test_a_complete_run_writes_latest_json_and_the_report(monkeypatch, tmp_path, results, capsys):
    results["scope"] = {"complete": True, "cases": 30, "total_cases": 30}
    written = {}
    monkeypatch.setattr(cli, "run_evals", lambda **kwargs: results)
    monkeypatch.setattr(cli, "write_outputs", lambda r: written.setdefault("r", r) and (tmp_path / "latest.json", tmp_path / "EVALS.md"))
    assert cli.main([]) == 0 and written["r"] is results and "Отчёт:" in capsys.readouterr().out


def test_an_unknown_selection_is_an_error(monkeypatch, capsys):
    assert cli.main(["--cases", "no-such-case"]) == 2 and "Не выбрано ни одного golden case" in capsys.readouterr().err


# ================================================================== опубликованные результаты
def test_committed_results_and_report_are_consistent():
    """Если в репозитории есть результаты реального запуска, EVALS.md собран из них и golden cases не менялись после запуска."""
    if not RESULTS_PATH.exists():
        pytest.skip("реальный запуск ещё не выполнялся: latest.json нет")
    stored = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assert stored["scope"]["complete"] and stored["scope"]["cases"] == 30 == len(stored["cases"])
    assert stored["environment"]["golden_sha256"] == file_sha256(GOLDEN_PATH), "golden cases изменились после запуска: выполните python -m datastory.evaluation"
    assert file_sha256(RESULTS_PATH) in report.REPORT_PATH.read_text(encoding="utf-8"), "EVALS.md собран не из текущего latest.json"
    assert {c["id"] for c in stored["cases"]} == set(BY_ID)


def test_every_complete_run_is_kept_in_history_and_listed_in_the_report(results, tmp_path):
    results["scope"] = {"complete": True, "cases": 30, "total_cases": 30}
    history = tmp_path / "history"
    older = json.loads(json.dumps(results))
    older["generated_at"] = "2026-09-25T10:00:00+00:00"
    history.mkdir()
    (history / "20260925T100000Z.json").write_text(json.dumps(older), encoding="utf-8")
    _, md = report.write_outputs(results, tmp_path / "latest.json", tmp_path / "EVALS.md", history)
    assert (history / "20260926T120000Z.json").exists()  # копия текущего прогона
    text = md.read_text(encoding="utf-8")
    section = text.split("## История прогонов")[1].split("## Как читать")[0]
    assert "2026-09-25T10:00:00+00:00" in section and "2026-09-26T12:00:00+00:00" in section
    assert "не является независимой отложенной выборкой" in text  # ограничение оценки названо явно
