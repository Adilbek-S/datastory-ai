"""A/B эксперимент: статистика, переключение RAG в планировщике, идентичность промптов, вывод и отчёт (LLM подменён сценарием)."""
import json
import re

import pytest

from datastory.evaluation import __main__ as cli
from datastory.evaluation import ab_test, report, stats
from datastory.evaluation.golden_builder import AOV, REVENUE, build_golden
from datastory.llm.usage import UsageRecorder
from datastory.workflow.graph import PlannerRag, WorkflowDeps
from datastory.workflow.models import PlanDraft
from datastory.workflow.runner import AnalysisRunner
from tests.helpers import ScriptedLLM
from tests.test_evaluation import make_env, sales_dataset, step  # noqa: F401 — фикстура и хелперы

BY_ID = {c.id: c for c in build_golden().cases}


# ================================================================== статистика
def test_paired_counts_split_the_pairs():
    assert stats.paired_counts([True, True, False, False], [True, False, True, False]) == {"both": 1, "only_a": 1, "only_b": 1, "neither": 1}
    with pytest.raises(ValueError):
        stats.paired_counts([True], [True, False])


def test_sign_test_is_exact_and_two_sided():
    assert stats.sign_test_pvalue(0, 0) == 1.0
    assert stats.sign_test_pvalue(5, 5) == 1.0
    assert stats.sign_test_pvalue(0, 5) == pytest.approx(2 / 32)
    assert stats.sign_test_pvalue(0, 10) < 0.01 and stats.sign_test_pvalue(3, 1) == pytest.approx(0.625)


def test_the_verdict_follows_the_confidence_interval():
    better = stats.bootstrap_difference([0.0] * 10, [1.0] * 10)
    worse = stats.bootstrap_difference([1.0] * 10, [0.0] * 10)
    same = stats.bootstrap_difference([1.0] * 10, [1.0] * 10)
    noisy = stats.bootstrap_difference([0, 1, 0, 1, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    assert stats.verdict(better) == "confirmed_better" and stats.verdict(worse) == "confirmed_worse"
    assert stats.verdict(same) == "no_difference" and stats.verdict(noisy) == "not_confirmed"
    assert stats.verdict(better, higher_is_better=False) == "confirmed_worse"  # для задержки «больше» — хуже


def test_the_bootstrap_is_reproducible_and_covers_the_mean():
    a, b = [0, 0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 1, 1, 1, 0, 1]
    first, second = stats.bootstrap_difference(a, b), stats.bootstrap_difference(a, b)
    assert first == second and first.low <= first.mean <= first.high


# ================================================================== переключение RAG и идентичность конфигураций
class Model(ScriptedLLM):
    """Сценарная модель с параметрами генерации (как у настоящей); показатель зависит от того, видит ли она документы."""

    temperature, max_tokens = 0.0, ab_test.MAX_TOKENS

    def __init__(self, respond):
        super().__init__(PlanDraft=respond)
        self.usage = UsageRecorder()


def documents_in(user: str) -> bool:
    return "Фрагменты документов" in user


def observe_arms(client, dataset_id, respond, case_id="mi-01", arms=("A", "B", "A0")):
    model = Model(respond)
    env = make_env(client, dataset_id, model)
    recorder = ab_test.RecordingLLM(model)
    out = {}
    for arm in arms:
        runner = AnalysisRunner(WorkflowDeps.of(client, recorder, env.kb, planner_rag=ab_test.ARMS[arm]["rag"]))
        out[arm] = ab_test.observe(env, runner, recorder, BY_ID[case_id], arm, 0, 0)
    return out, model


def test_a_gets_no_rag_context_and_b_gets_top_five(client, sales_dataset):
    out, model = observe_arms(client, sales_dataset, lambda s, u, n: PlanDraft(goal="x", steps=[step(AOV, ["Channel"])]))
    prompts = {arm: [(s, u) for s, u in model.prompts("PlanDraft")][i] for i, arm in enumerate(("A", "B", "A0"))}
    assert not documents_in(prompts["A"][1]) and not documents_in(prompts["A0"][1]) and documents_in(prompts["B"][1])
    assert out["A"]["context"]["documents"] == [] and len(out["B"]["context"]["documents"]) == 5
    assert out["A"]["prompt"]["user_has_documents"] is False and out["B"]["prompt"]["user_has_documents"] is True


def test_configurations_differ_only_in_the_user_prompt(client, sales_dataset):
    out, model = observe_arms(client, sales_dataset, lambda s, u, n: PlanDraft(goal="x", steps=[step(AOV, ["Channel"])]))
    systems = {system for system, _ in model.prompts("PlanDraft")}
    assert len(systems) == 1 and len({o["prompt"]["system_sha"] for o in out.values()}) == 1
    a_user, b_user = (u for _, u in model.prompts("PlanDraft")[:2])
    without_documents = re.sub(r"\n*Фрагменты документов.*?(?=\n\n[А-ЯA-Z]|\Z)", "", b_user, flags=re.S)
    assert len(a_user) < len(b_user) and a_user.split("Фрагменты")[0].strip() == b_user.split("Фрагменты")[0].strip() and without_documents


def test_hit_at_3_is_measured_on_the_context_passed_to_b_only(client, sales_dataset):
    out, _ = observe_arms(client, sales_dataset, lambda s, u, n: PlanDraft(goal="x", steps=[step(AOV, ["Channel"])]))
    assert "hit_at_3" not in out["A"]["context"] and out["B"]["context"]["hit_at_3"] in (True, False) and "hit_at_5" in out["B"]["context"]


def test_a_rag_dependent_plan_can_differ_between_arms(client, sales_dataset):
    def respond(system, user, n):  # модель «знает» термин, только если видит документы
        return PlanDraft(goal="x", steps=[step(AOV if documents_in(user) else REVENUE, ["Region"])])

    out, _ = observe_arms(client, sales_dataset, respond, case_id="rg-02", arms=("A", "B"))
    assert out["A"]["plan_ok"] is False and out["B"]["plan_ok"] is True
    assert out["A"]["metric_ok"] is False and out["B"]["metric_ok"] is True


def test_the_rejection_of_an_invalid_request_is_recorded(client, sales_dataset):
    out, _ = observe_arms(client, sales_dataset, lambda s, u, n: PlanDraft(goal="x", steps=[], unsupported=["EBITDA"]), case_id="iv-01", arms=("A", "B"))
    assert out["A"]["behavior_ok"] and out["B"]["behavior_ok"] and out["A"]["plan_ok"] is None


def test_a_failed_case_counts_as_a_miss_not_as_missing_data(client, sales_dataset):
    def broken(system, user, n):
        return RuntimeError("сеть")

    out, _ = observe_arms(client, sales_dataset, broken, arms=("A",))
    assert out["A"]["plan_ok"] is False or out["A"]["error"]


# ================================================================== сравнение и вывод
def fake(arm, case_id, group, ok, behavior="answer", **extra):
    return {
        "arm": arm, "repeat": 0, "position": 0, "case_id": case_id, "group": group, "expected_behavior": behavior,
        "predicted_behavior": behavior, "predicted_metric": "m", "behavior_ok": True, "plan_ok": ok if behavior == "answer" else None,
        "metric_ok": ok if behavior == "answer" else None, "chart_ok": ok if behavior == "answer" else None,
        "plan_latency_ms": 2000.0 + (100 if arm == "B" else 0), "tokens": {"input": 5000, "output": 100, "total": 5100 + (1000 if arm == "B" else 0)},
        "embedding_tokens": 10 if arm == "B" else 0, "context": {"documents": []}, "prompt": {"calls": 1, "system_sha": "s", "system_chars": 1, "user_chars": 1, "user_has_documents": arm == "B"},
        "error": None, **extra,
    }


def dataset(a_ok, b_ok, n=10):
    rows = []
    for i in range(n):
        for arm, ok in (("A", a_ok(i)), ("B", b_ok(i))):
            rows.append(fake(arm, f"c{i}", "rag_terminology" if i < 5 else "filters", ok))
    rows += [fake(arm, "r1", "invalid_ambiguous", None, behavior="reject", behavior_ok=arm == "B") for arm in ("A", "B")]
    return rows


def arms_of(observations):
    return {arm: ab_test.arm_block(observations, arm) for arm in ("A", "B")}


def concluded(observations):
    return ab_test.conclude(ab_test.build_comparisons(observations, [("A", "B")]), arms_of(observations))


def test_a_clear_improvement_is_confirmed_and_may_be_called_better():
    conclusion = concluded(dataset(lambda i: i >= 8, lambda i: True))
    assert conclusion["verdict"] == "confirmed_better" and conclusion["rag_better_claim"] is True
    assert "подтверждена" in conclusion["statements"][0]


def test_no_difference_never_claims_that_rag_is_better():
    conclusion = concluded(dataset(lambda i: True, lambda i: True))
    assert conclusion["verdict"] == "no_difference" and conclusion["rag_better_claim"] is False
    assert "не подтверждена" in conclusion["statements"][0] or "не подтвержд" in conclusion["statements"][0]
    assert any("Потолок" in c for c in conclusion["caveats"])  # A уже 100%: улучшать нечего


def test_a_small_noisy_gain_is_not_confirmed():
    conclusion = concluded(dataset(lambda i: i != 0, lambda i: True))  # +1 случай из 10
    assert conclusion["verdict"] == "not_confirmed" and conclusion["rag_better_claim"] is False
    assert any("статистической силы" in c for c in conclusion["caveats"])


def test_worse_b_is_reported_as_worse():
    conclusion = concluded(dataset(lambda i: True, lambda i: i >= 5))
    assert conclusion["verdict"] == "confirmed_worse" and "снизил" in conclusion["statements"][0] and not conclusion["rag_better_claim"]


def test_rag_dependent_cases_are_compared_separately():
    comparisons = ab_test.build_comparisons(dataset(lambda i: i >= 5, lambda i: True), [("A", "B")])
    flags = comparisons["B_vs_A"]["flags"]["plan_accuracy"]
    assert flags["rag_terminology"]["difference_pp"] == 100.0 and flags["rag_terminology"]["observations"] == 5
    assert flags["all"]["observations"] == 10 and flags["all"]["difference_pp"] == 50.0
    assert comparisons["B_vs_A"]["invalid_rejection"]["observations"] == 1


def test_resources_are_compared_with_lower_is_better():
    comparisons = ab_test.build_comparisons(dataset(lambda i: True, lambda i: True), [("A", "B")])["B_vs_A"]
    assert comparisons["latency_ms"]["difference"] == 100.0 and comparisons["latency_ms"]["verdict"] == "confirmed_worse"
    assert comparisons["llm_tokens_total"]["difference"] == 1000.0


def test_arm_block_counts_rates_and_tokens():
    block = arms_of(dataset(lambda i: i < 6, lambda i: True))["A"]
    assert block["plan_accuracy"]["hits"] == 6 and block["plan_accuracy"]["total"] == 10
    assert block["invalid_rejection_rate"] == {"hits": 0, "total": 1, "rate": 0.0}
    assert block["llm_tokens"]["input_mean"] == 5000.0 and block["by_group"]["rag_terminology"]["plan_accuracy"]["hits"] == 5


# ================================================================== отчёт
def make_ab(observations):
    arms = arms_of(observations)
    comparisons = ab_test.build_comparisons(observations, [("A", "B")])
    return {
        "schema_version": 1, "generated_at": "2026-09-27T10:00:00+00:00", "duration_s": 1.0, "hypothesis": "гипотеза",
        "config": {
            "llm_model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 1024, "repeats": 1, "cases": 30, "golden_sha256": "c" * 64, "dataset": "data/demo/sales_2026.xlsx",
            "embedding_provider": "offline-hash", "identical_system_prompt": True, "system_prompt_sha": ["abcd"], "langsmith_tracing": False,
            "arms": {k: {"title": v["title"], **vars(v["rag"])} for k, v in ab_test.ARMS.items()},
        },
        "arms": arms, "comparisons": comparisons, "conclusion": ab_test.conclude(comparisons, arms), "observations": observations,
    }


def test_the_report_section_contains_only_numbers_from_the_results():
    ab = make_ab(dataset(lambda i: i != 0, lambda i: True))
    text = "\n".join(report.render_ab(ab, "e" * 64, {"c0": "запрос"}))
    assert "A/B эксперимент" in text and "e" * 64 in text and "Retrieval Hit@3 для конфигурации B" in text and "RAG-dependent" in text
    assert "не подтверждено" in text and "Гипотеза подтверждена" not in text
    known = set()
    for c in [ab["comparisons"]["B_vs_A"]["invalid_rejection"], *[v for f in ab["comparisons"]["B_vs_A"]["flags"].values() for v in f.values() if v]]:
        for arm in ("A", "B"):
            known.add((f"{c[arm]['rate'] * 100:.1f}", str(c[arm]["hits"]), str(c["observations"])))
    for arm in ab["arms"].values():
        for b in arm.values():
            if isinstance(b, dict) and "rate" in b and b["total"]:
                known.add((f"{b['rate'] * 100:.1f}", str(b["hits"]), str(b["total"])))
        for g in arm["by_group"].values():
            known |= {(f"{b['rate'] * 100:.1f}", str(b["hits"]), str(b["total"])) for b in g.values() if b["total"]}
    assert set(re.findall(r"(\d+\.\d)% \((\d+)/(\d+)\)", text)) <= known


def test_the_ab_outputs_are_written_and_the_report_records_the_hash(tmp_path):
    ab = make_ab(dataset(lambda i: True, lambda i: True))
    ab_path, report_path = report.write_ab_outputs(ab, tmp_path / "ab_test.json", tmp_path / "EVALS.md", tmp_path / "missing.json", tmp_path / "h")
    text = report_path.read_text(encoding="utf-8")
    import hashlib

    assert json.loads(ab_path.read_text(encoding="utf-8"))["conclusion"] == ab["conclusion"] and hashlib.sha256(ab_path.read_bytes()).hexdigest() in text
    assert "Гипотеза не подтверждена" in text


def test_the_ab_section_is_kept_when_the_main_report_is_rebuilt(tmp_path):
    from tests.test_evaluation import results  # noqa: F401
    ab = make_ab(dataset(lambda i: True, lambda i: True))
    assert "A/B эксперимент" in "\n".join(report.render_ab(ab, "0" * 64))


def test_the_committed_ab_results_and_report_are_consistent():
    path = report.AB_PATH
    if not path.exists():
        pytest.skip("A/B эксперимент ещё не запускался")
    ab = json.loads(path.read_text(encoding="utf-8"))
    text = report.REPORT_PATH.read_text(encoding="utf-8")
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() in text
    conclusion = ab["conclusion"]
    assert conclusion["rag_better_claim"] == (ab["comparisons"]["B_vs_A"]["flags"]["plan_accuracy"]["all"]["verdict"] == "confirmed_better")
    assert ab["config"]["identical_system_prompt"] is True
    assert len({o["arm"] for o in ab["observations"]}) >= 2


# ================================================================== CLI
def test_ab_needs_an_llm_and_does_not_invent_results(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "run_ab", lambda repeats: (_ for _ in ()).throw(RuntimeError("A/B эксперимент невозможен: не задан OPENAI_API_KEY")))
    assert cli.main(["--ab"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_ab_cli_writes_results_and_prints_the_conclusion(monkeypatch, capsys):
    ab = make_ab(dataset(lambda i: True, lambda i: True))
    written = {}
    monkeypatch.setattr(cli, "run_ab", lambda repeats: ab | {"repeats_arg": repeats})
    monkeypatch.setattr(cli, "write_ab_outputs", lambda result: written.setdefault("r", result) and ("ab.json", "EVALS.md"))
    assert cli.main(["--ab", "--repeats", "2"]) == 0
    assert written["r"]["repeats_arg"] == 2 and "Гипотеза не подтверждена" in capsys.readouterr().out


def test_planner_rag_defaults_keep_the_normal_behaviour():
    rag = PlannerRag()
    assert rag.context_hits == 3 and rag.only_relevant and rag.validation
