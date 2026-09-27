"""Эксперимент по выбору temperature: метрики стабильности, правило выбора, отчёт и настройки (без обращений к OpenAI)."""
import hashlib
import json
import re

import pytest

from datastory.config import Settings
from datastory.evaluation import __main__ as cli
from datastory.evaluation import hyperparams as hp
from datastory.evaluation import report
from datastory.evaluation.golden_builder import build_golden
from datastory.llm.client import OpenAIStructuredLLM, get_llm
from tests.test_evaluation import results, sales_dataset  # noqa: F401 — фикстуры

CASES = {c.id: c for c in build_golden().cases}


def sig(metric="sum:Revenue_KZT", group="Region", behavior="answer"):
    return json.dumps([behavior, metric, [group], [], "bar", 1], ensure_ascii=False)


def obs(arm, case_id, repeat, *, signature=None, ok=True, error=None, latency=2000.0, total=5000, output=100, behavior="answer"):
    return {
        "arm": arm, "repeat": repeat, "case_id": case_id, "group": "g", "expected_behavior": behavior, "predicted_behavior": None if error else behavior,
        "predicted_metric": None if error else "sum:Revenue_KZT", "behavior_ok": None if error else True,
        "plan_ok": (False if error else ok) if behavior == "answer" else None, "metric_ok": ok, "chart_ok": ok,
        "signature": None if error else (signature or sig()), "plan_latency_ms": latency, "tokens": {"llm_calls": 1, "input": total - output, "output": output, "total": total},
        "embedding_tokens": 0, "error": error, "context": {"documents": []}, "prompt": {"system_sha": "s"},
    }


def experiment(t0_ok, t4_ok, t0_sig=lambda c, r: None, t4_sig=lambda c, r: None, cases=8, repeats=5, t4_latency=2000.0, t4_total=5000):
    rows = []
    for c in range(cases):
        for r in range(repeats):
            rows.append(obs("T0", f"c{c}", r, ok=t0_ok(c, r), signature=t0_sig(c, r)))
            rows.append(obs("T04", f"c{c}", r, ok=t4_ok(c, r), signature=t4_sig(c, r), latency=t4_latency, total=t4_total))
    return rows


def summarize(rows):
    arms = {a: hp.arm_summary(rows, a, 5) for a in hp.CONFIGS}
    comparison = hp.compare(rows, "T0", "T04")
    return arms, comparison, hp.decide(arms, comparison)


# ================================================================== метрики стабильности
def test_agreement_is_the_share_of_the_most_common_value():
    assert hp.agreement(["a", "a", "a", "a", "a"]) == 1.0 and hp.agreement(["a", "a", "b", "c"]) == 0.5 and hp.agreement([]) == 0.0


def test_structure_stability_counts_different_structures_and_errors():
    rows = [obs("T0", "c0", r) for r in range(5)] + [obs("T04", "c0", r, signature=sig(group=f"G{r % 2}")) for r in range(4)] + [obs("T04", "c0", 4, error="LLMError")]
    a0, a4 = hp.arm_summary(rows, "T0", 5), hp.arm_summary(rows, "T04", 5)
    assert a0["structure_agreement"] == 1.0 and a0["identical_cases"] == {"cases": 1, "total": 1}
    assert a4["structure_agreement"] == 0.4 and a4["distinct_structures_per_case"] == {"c0": 3}  # G0, G1 и «error»
    assert a4["errors"] == 1 and a4["valid_output_rate"] == 0.8


def test_the_signature_ignores_the_order_of_columns_and_filters():
    one = {"behavior": "answer", "metric": "m", "group_by": ["B", "A"], "filters": [{"column": "X", "op": "in", "value": ["1"]}], "chart_type": "bar", "steps": 1}
    two = {**one, "group_by": ["A", "B"]}
    from datastory.evaluation.ab_test import structure_signature

    assert structure_signature(one) == structure_signature(two) and structure_signature(one) != structure_signature({**one, "chart_type": "line"})


def test_token_cap_choice_reports_the_observed_maximum():
    rows = experiment(lambda c, r: True, lambda c, r: True)
    rows[3]["tokens"]["output"] = 300
    choice = hp.output_token_choice({a: hp.arm_summary(rows, a, 5) for a in hp.CONFIGS}, 1024)
    assert choice["observed_max_output_tokens"] == 300 and choice["share_of_cap"] == pytest.approx(0.293, abs=1e-3) and "300" in choice["reason"]


# ================================================================== правило выбора
def test_confirmed_higher_accuracy_decides_first():
    _, comparison, decision = summarize(experiment(lambda c, r: c >= 4, lambda c, r: True))
    assert comparison["plan_accuracy"]["verdict"] == "confirmed_better"
    assert decision["selected"] == "T04" and decision["temperature"] == 0.4 and decision["rule_step"] == 1


def test_confirmed_lower_accuracy_keeps_temperature_zero():
    _, _, decision = summarize(experiment(lambda c, r: True, lambda c, r: c >= 4))
    assert decision["selected"] == "T0" and decision["rule_step"] == 1


def test_without_an_accuracy_difference_the_more_stable_structure_wins():
    rows = experiment(lambda c, r: True, lambda c, r: True, t4_sig=lambda c, r: sig(group=f"G{r % 3}"))
    _, comparison, decision = summarize(rows)
    assert comparison["plan_accuracy"]["verdict"] == "no_difference" and decision["selected"] == "T0" and decision["rule_step"] == 2
    rows = experiment(lambda c, r: True, lambda c, r: True, t0_sig=lambda c, r: sig(group=f"G{r % 3}"))
    assert summarize(rows)[2]["selected"] == "T04"  # выбор не привязан к temperature 0: побеждает более стабильная


def test_equal_stability_falls_back_to_cost_then_to_temperature_zero():
    _, _, decision = summarize(experiment(lambda c, r: True, lambda c, r: True, t4_latency=1000.0))
    assert decision["selected"] == "T04" and decision["rule_step"] == 3
    _, _, decision = summarize(experiment(lambda c, r: True, lambda c, r: True))
    assert decision["selected"] == "T0" and decision["rule_step"] == 3 and "воспроизводим" in decision["reasons"][-1] or "воспроизводим" in " ".join(decision["reasons"])


def test_the_reasons_quote_the_measured_numbers():
    _, _, decision = summarize(experiment(lambda c, r: True, lambda c, r: True, t4_latency=2500.0))
    text = " ".join(decision["reasons"])
    assert "100.0% (40/40)" in text and "Задержка" in text and "Токены LLM" in text


# ================================================================== отчёт
def make_hp(rows, selected_cases=None):
    arms = {a: hp.arm_summary(rows, a, 5) for a in hp.CONFIGS}
    comparison = hp.compare(rows, "T0", "T04")
    decision = hp.decide(arms, comparison)
    return {
        "schema_version": 1, "generated_at": "2026-09-27T12:00:00+00:00", "duration_s": 1.0,
        "config": {
            "llm_model": "gpt-4o-mini", "max_output_tokens": 1024, "repeats": 5, "case_ids": [f"c{i}" for i in range(8)], "golden_sha256": "c" * 64, "dataset": "data/demo/sales_2026.xlsx",
            "embedding_provider": "offline-hash", "identical_system_prompt": True, "system_prompt_sha": ["abcd"], "configs": {a: {"title": hp.TITLES[a], "temperature": t} for a, t in hp.CONFIGS.items()},
            "not_studied": "top_p и другие комбинации",
        },
        "arms": arms, "comparison": comparison, "decision": decision, "max_output_tokens_choice": hp.output_token_choice(arms, 1024),
        "final": {"llm_model": "gpt-4o-mini", "temperature": decision["temperature"], "max_output_tokens": 1024}, "observations": rows,
    }


def test_the_report_section_states_model_temperature_cap_and_reasons():
    result = make_hp(experiment(lambda c, r: True, lambda c, r: True, t4_sig=lambda c, r: sig(group=f"G{r % 3}")))
    text = "\n".join(report.render_hp(result, "e" * 64))
    assert "Выбор гиперпараметров LLM" in text and "e" * 64 in text
    assert "`gpt-4o-mini`" in text and "**0.0**" in text and "**1024**" in text and "Причины" in text and "top_p" in text
    assert "max output tokens:" in text and "Стабильность структуры" in text and "Analysis Plan Accuracy" in text


def test_the_final_row_follows_the_decision_not_the_default():
    result = make_hp(experiment(lambda c, r: c >= 4, lambda c, r: True))
    assert "**0.4**" in "\n".join(report.render_hp(result, "0" * 64))


def test_outputs_are_written_and_the_report_records_the_file_hash(tmp_path):
    result = make_hp(experiment(lambda c, r: True, lambda c, r: True))
    hp_path, report_path = report.write_hp_outputs(result, tmp_path / "hp.json", tmp_path / "EVALS.md", tmp_path / "none.json", tmp_path / "h", tmp_path / "no_ab.json")
    assert hashlib.sha256(hp_path.read_bytes()).hexdigest() in report_path.read_text(encoding="utf-8")


def test_the_section_survives_a_rebuild_of_the_main_report(results, tmp_path):
    result = make_hp(experiment(lambda c, r: True, lambda c, r: True))
    hp_path = tmp_path / "hp.json"
    hp_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    results_path = tmp_path / "latest.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    report_path = report.write_report_from_file(results_path, tmp_path / "EVALS.md", tmp_path / "h", tmp_path / "no_ab.json", hp_path)
    text = report_path.read_text(encoding="utf-8")
    assert "Выбор гиперпараметров LLM" in text and hashlib.sha256(hp_path.read_bytes()).hexdigest() in text and "## Итоговые метрики" in text


def test_percentages_in_the_section_come_from_the_results():
    result = make_hp(experiment(lambda c, r: c != 0, lambda c, r: True))
    text = "\n".join(report.render_hp(result, "e" * 64))
    cmp_ = result["comparison"]
    known = set()
    for key in ("plan_accuracy", "behavior_accuracy"):
        c = cmp_[key]
        for arm in ("T0", "T04"):
            known.add((f"{c[arm]['rate'] * 100:.1f}", str(c[arm]["hits"]), str(c["observations"])))
    for arm in result["arms"].values():
        known.add((f"{arm['valid_output_rate'] * 100:.1f}", str(arm["observations"] - arm["errors"]), str(arm["observations"])))
    assert set(re.findall(r"(\d+\.\d)% \((\d+)/(\d+)\)", text)) <= known


def test_the_committed_result_matches_the_published_defaults():
    path = report.HP_PATH
    if not path.exists():
        pytest.skip("эксперимент ещё не запускался")
    result = json.loads(path.read_text(encoding="utf-8"))
    settings = Settings(_env_file=None)
    assert result["final"]["temperature"] == settings.llm_temperature == result["decision"]["temperature"]
    assert result["final"]["max_output_tokens"] == settings.llm_max_output_tokens and result["final"]["llm_model"] == settings.openai_model
    assert result["config"]["identical_system_prompt"] and len(result["config"]["case_ids"]) == 10
    assert hashlib.sha256(path.read_bytes()).hexdigest() in report.REPORT_PATH.read_text(encoding="utf-8")
    assert all(c in CASES for c in result["config"]["case_ids"])


# ================================================================== настройки и CLI
def test_settings_define_the_generation_parameters():
    settings = Settings(_env_file=None, openai_api_key="k", llm_temperature=0.4, llm_max_output_tokens=500)
    llm = get_llm(settings)
    assert (llm.temperature, llm.max_tokens) == (0.4, 500) and llm._chat.temperature == 0.4 and llm._chat.max_tokens == 500
    overridden = get_llm(settings, max_tokens=64, temperature=0.0)
    assert (overridden.temperature, overridden.max_tokens) == (0.0, 64)


def test_the_client_default_stays_deterministic():
    assert OpenAIStructuredLLM("k", "m").temperature == 0.0


def test_a_missing_key_makes_the_experiment_impossible_instead_of_inventing_results(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_hyperparams", lambda repeats: (_ for _ in ()).throw(RuntimeError("Эксперимент невозможен: не задан OPENAI_API_KEY")))
    assert cli.main(["--hyperparams"]) == 2 and "OPENAI_API_KEY" in capsys.readouterr().err


def test_cli_writes_the_result_and_prints_the_selection(monkeypatch, capsys):
    result = make_hp(experiment(lambda c, r: True, lambda c, r: True))
    seen = {}
    monkeypatch.setattr(cli, "run_hyperparams", lambda repeats: seen.setdefault("repeats", repeats) and result)
    monkeypatch.setattr(cli, "write_hp_outputs", lambda hp_result: ("hp.json", "EVALS.md"))
    assert cli.main(["--hyperparams"]) == 0 and seen["repeats"] == hp.DEFAULT_REPEATS
    assert "Выбрано: gpt-4o-mini, temperature 0.0, max output tokens 1024" in capsys.readouterr().out


def test_the_ten_cases_are_fixed_in_advance_and_cover_every_group():
    assert len(hp.CASE_IDS) == 10 and len(set(hp.CASE_IDS)) == 10 and all(c in CASES for c in hp.CASE_IDS)
    assert {CASES[c].group for c in hp.CASE_IDS} == {c.group for c in CASES.values()}


# ================================================================== обрезание по лимиту токенов, оговорки и архив прошлых запусков
def test_a_response_cut_by_the_token_limit_is_an_llm_error_not_a_crash():
    import openai
    from pydantic import BaseModel

    from datastory.errors import LLMError
    from datastory.llm.client import LENGTH_LIMIT_MESSAGE

    class Answer(BaseModel):
        x: int

    class Chat:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages, config=None):
            raise openai.LengthFinishReasonError.__new__(openai.LengthFinishReasonError)

    llm = OpenAIStructuredLLM("k", "m", chat=Chat())
    with pytest.raises(LLMError, match="max output tokens") as info:
        llm.generate(Answer, system="s", user="u")
    assert info.value.user_message == LENGTH_LIMIT_MESSAGE


def test_a_fallback_to_rules_counts_as_an_unusable_model_answer_and_a_truncation_is_counted():
    rows = experiment(lambda c, r: True, lambda c, r: True)
    rows[0]["llm_fallback"], rows[0]["truncated"] = True, True  # T0 c0 r0
    arms = {a: hp.arm_summary(rows, a, 5) for a in hp.CONFIGS}
    assert arms["T0"]["errors"] == 1 and arms["T0"]["truncated"] == 1 and arms["T04"]["errors"] == 0
    assert arms["T0"]["distinct_structures_per_case"]["c0"] == 2  # «error» — отдельная структура
    choice = hp.output_token_choice(arms, 1024)
    assert choice["truncated"] == 1 and "обрезанных лимитом, — 1 из 80" in choice["reason"] and "вырожденные" in choice["reason"]


def test_a_choice_based_on_an_unconfirmed_stability_gap_carries_a_caveat():
    rows = experiment(lambda c, r: True, lambda c, r: True, t0_sig=lambda c, r: sig(group="G") if (c, r) != (0, 0) else sig(group="H"))
    _, comparison, decision = summarize(rows)
    assert decision["rule_step"] == 2 and comparison["structure_agreement"]["verdict"] != "confirmed_better"
    assert any("включает ноль" in c for c in decision["caveats"])
    assert summarize(experiment(lambda c, r: c >= 4, lambda c, r: True))[2]["caveats"] == []


def test_a_confirmed_cost_of_the_chosen_configuration_is_disclosed():
    _, _, decision = summarize(experiment(lambda c, r: c >= 4, lambda c, r: True, t4_total=6000, t4_latency=3000.0))
    assert decision["selected"] == "T04" and any("число токенов" in c for c in decision["caveats"])


def test_the_previous_result_is_archived_and_listed_not_overwritten(tmp_path):
    first = make_hp(experiment(lambda c, r: True, lambda c, r: True))
    second = {**make_hp(experiment(lambda c, r: True, lambda c, r: True)), "generated_at": "2026-09-28T09:00:00+00:00"}
    hp_path, history = tmp_path / "hp.json", tmp_path / "h"
    kwargs = dict(report_path=tmp_path / "EVALS.md", results_path=tmp_path / "none.json", history_dir=history, ab_path=tmp_path / "no_ab.json")
    report.write_hp_outputs(first, hp_path, **kwargs)
    assert not list(history.glob("hp-*.json"))
    report.write_hp_outputs(second, hp_path, **kwargs)
    assert [p.name for p in history.glob("hp-*.json")] == ["hp-20260927T120000Z.json"]
    assert json.loads(hp_path.read_text(encoding="utf-8"))["generated_at"] == "2026-09-28T09:00:00+00:00"


def test_previous_runs_are_listed_in_the_rebuilt_report(results, tmp_path):
    current = {**make_hp(experiment(lambda c, r: True, lambda c, r: True)), "generated_at": "2026-09-28T09:00:00+00:00"}
    old = make_hp(experiment(lambda c, r: True, lambda c, r: True))
    history = tmp_path / "h"
    history.mkdir()
    (history / "hp-20260927T120000Z.json").write_text(json.dumps(old), encoding="utf-8")
    hp_path, results_path = tmp_path / "hp.json", tmp_path / "latest.json"
    hp_path.write_text(json.dumps(current), encoding="utf-8")
    results_path.write_text(json.dumps(results), encoding="utf-8")
    text = report.write_report_from_file(results_path, tmp_path / "EVALS.md", history, tmp_path / "no_ab.json", hp_path).read_text(encoding="utf-8")
    assert "Предыдущие запуски этого эксперимента" in text and "2026-09-27T12:00:00+00:00" in text


def test_recompute_rebuilds_the_derived_values_from_the_stored_observations():
    from datetime import datetime

    result = make_hp(experiment(lambda c, r: True, lambda c, r: True))
    result["observations"][0]["truncated"] = True  # правило подсчёта учитывает обрезание: пересчёт это отражает
    again = hp.recompute(result)
    assert again["observations"] == result["observations"] and again["generated_at"] == result["generated_at"]
    assert again["arms"]["T0"]["errors"] == 1 and result["arms"]["T0"]["errors"] == 0
    assert datetime.fromisoformat(again["generated_at"])


def test_a_confirmed_saving_of_the_other_configuration_is_disclosed_when_zero_is_chosen():
    rows = experiment(lambda c, r: True, lambda c, r: True, t4_total=4000, t4_latency=1500.0, t4_sig=lambda c, r: sig(group=f"G{r % 2}"))
    _, _, decision = summarize(rows)
    assert decision["selected"] == "T0" and any("T0.4 экономнее" in c for c in decision["caveats"])
