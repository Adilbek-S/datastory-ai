"""A/B эксперимент: улучшает ли контекст RAG правильность интерпретации доменных показателей и AnalysisPlan.

Гипотеза: планировщик с контекстом RAG в промпте (B) строит более точные планы, чем без него (A).
Конфигурации отличаются ТОЛЬКО пользовательской частью промпта LLM:
    A — DatasetProfile (сводка набора, показатели, измерения) и user_query; контекста RAG нет вовсе;
    B — то же и Top-5 фрагментов базы знаний по запросу (без порога релевантности).
Одинаково: LLM, системный промпт, temperature, max_tokens, golden cases, набор данных, MCP, база знаний.
Проверка плана после ответа LLM (определения показателей, защита от подмены термина) в A и B одинаковая и обращается к базе
знаний, поэтому A — не «система без RAG». Дополнительная конфигурация A0 отключает RAG полностью, в том числе при проверке плана.

Все случаи выполняются в каждой конфигурации несколько раз (repeats), порядок конфигураций чередуется по случаям и повторам.
Вывод формируется по правилу из stats.verdict: «RAG лучше» пишется только при доверительном интервале выше нуля.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datastory.evaluation import scoring, stats
from datastory.evaluation.golden import GOLDEN_PATH, GoldenCase, file_sha256, load_golden
from datastory.evaluation.report import AB_PATH
from datastory.llm.client import LENGTH_LIMIT_MESSAGE
from datastory.evaluation.runner import Environment, build_environment, run_case
from datastory.workflow.graph import PlannerRag, WorkflowDeps
from datastory.workflow.runner import AnalysisRunner

SCHEMA_VERSION = 1
MAX_TOKENS = 1024
DEFAULT_REPEATS = 3
ARMS: dict[str, dict[str, Any]] = {
    "A": {"title": "A: без RAG в промпте", "rag": PlannerRag(context_hits=0, validation=True)},
    "B": {"title": "B: Top-5 RAG в промпте", "rag": PlannerRag(context_hits=5, only_relevant=False, validation=True)},
    "A0": {"title": "A0 (дополнительно): RAG выключен полностью", "rag": PlannerRag(context_hits=0, validation=False)},
}
SUBSETS = {"all": None, "rag_terminology": "rag_terminology", "metric_interpretation": "metric_interpretation"}
SUBSET_TITLES = {"all": "Все случаи", "rag_terminology": "RAG-dependent (группа 5: терминология из базы знаний)", "metric_interpretation": "Интерпретация показателя (группа 1)"}


class RecordingLLM:
    """Обёртка над LLM: запоминает промпты каждого вызова (для проверки, что системный промпт одинаков в конфигурациях)."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []
        self.name, self.usage = inner.name, inner.usage
        self.temperature, self.max_tokens = inner.temperature, inner.max_tokens

    def generate(self, schema, *, system: str, user: str):
        self.calls.append({"schema": schema.__name__, "system": system, "user": user})
        return self.inner.generate(schema, system=system, user=user)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- одно наблюдение
def structure_signature(predicted: dict[str, Any]) -> str:
    """Структура ответа планировщика (поведение, показатель, группировка, фильтры, тип графика, число шагов) одной строкой: для сравнения повторов."""
    filters = sorted((f["column"], f["op"], json.dumps(f["value"], ensure_ascii=False, sort_keys=True, default=str)) for f in predicted.get("filters", []))
    return json.dumps(
        [predicted.get("behavior"), predicted.get("metric"), sorted(predicted.get("group_by", [])), filters, predicted.get("chart_type"), predicted.get("steps")],
        ensure_ascii=False, default=str,
    )


def observe(env: Environment, runner: AnalysisRunner, llm: RecordingLLM, case: GoldenCase, arm: str, repeat: int, position: int) -> dict[str, Any]:
    llm.calls.clear()
    record = run_case(env, case, runner=runner, retrieval=False, numeric=False)
    plan = record.get("plan") or {}
    predicted = plan.get("predicted") or {}
    answer_case = case.expected_behavior == "answer"
    prompts = [c for c in llm.calls if c["schema"] == "PlanDraft"]
    context = []
    if record.get("thread_id") and not record["error"]:
        state = runner.graph.get_state({"configurable": {"thread_id": record["thread_id"]}}).values.get("context")
        context = [{"document": h.source.filename, "section": h.source.section, "score": h.score} for h in (state.query_hits if state else [])]
    observation: dict[str, Any] = {
        "arm": arm, "repeat": repeat, "position": position, "case_id": case.id, "group": case.group, "expected_behavior": case.expected_behavior,
        "predicted_behavior": predicted.get("behavior"), "predicted_metric": predicted.get("metric"),
        "behavior_ok": None if record["error"] else bool(plan.get("behavior_ok")),
        "plan_ok": (False if record["error"] else bool(plan.get("accurate"))) if answer_case else None,
        "metric_ok": (False if record["error"] else bool(plan.get("metric_ok"))) if answer_case else None,
        "chart_ok": (False if record["error"] else bool(plan.get("chart_ok"))) if answer_case else None,
        "signature": structure_signature(predicted) if predicted else None, "planner": predicted.get("planner"),
        "llm_fallback": bool(predicted) and predicted.get("planner") == "rules",  # модель не дала пригодного ответа: план по правилам
        "truncated": any(LENGTH_LIMIT_MESSAGE in w for w in predicted.get("warnings", [])),  # ответ достиг лимита max output tokens
        "plan_latency_ms": record.get("plan_latency_ms"), "tokens": record.get("tokens", {}), "embedding_tokens": record.get("embedding_tokens", 0),
        "context": {"documents": context},
        "prompt": {
            "calls": len(prompts), "system_sha": _sha(prompts[0]["system"]) if prompts else None,
            "system_chars": len(prompts[0]["system"]) if prompts else None, "user_chars": len(prompts[0]["user"]) if prompts else None,
            "user_has_documents": ("Фрагменты документов" in prompts[0]["user"]) if prompts else None,
        },
        "error": record["error"],
    }
    if case.expected_rag_document and context:
        observation["context"].update(
            hit_at_3=scoring.retrieval_hit(context, case.expected_rag_document, 3), hit_at_5=scoring.retrieval_hit(context, case.expected_rag_document, 5),
            section_hit_at_3=bool(case.expected_rag_section) and scoring.section_hit(context, case.expected_rag_document, case.expected_rag_section, 3),
        )
    return observation


# --------------------------------------------------------------------------- сравнение
def _flags(observations: list[dict], arm: str, key: str, subset: str | None) -> dict[tuple, bool]:
    return {
        (o["case_id"], o["repeat"]): bool(o[key]) for o in observations
        if o["arm"] == arm and o[key] is not None and (subset is None or o["group"] == subset)
    }


def _per_case(flags: dict[tuple, bool]) -> dict[str, float]:
    by_case: dict[str, list[bool]] = {}
    for (case_id, _), value in flags.items():
        by_case.setdefault(case_id, []).append(value)
    return {case: sum(v) / len(v) for case, v in by_case.items()}


def compare_flag(observations: list[dict], base: str, other: str, key: str, subset: str | None) -> dict[str, Any] | None:
    a, b = _flags(observations, base, key, subset), _flags(observations, other, key, subset)
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    a_list, b_list = [a[k] for k in keys], [b[k] for k in keys]
    per_a, per_b = _per_case(a), _per_case(b)
    cases = sorted(set(per_a) & set(per_b))
    diff = stats.bootstrap_difference([per_a[c] for c in cases], [per_b[c] for c in cases])
    counts = stats.paired_counts(a_list, b_list)
    return {
        "observations": len(keys), "cases": len(cases),
        base: {"hits": sum(a_list), "rate": round(sum(a_list) / len(a_list), 4)},
        other: {"hits": sum(b_list), "rate": round(sum(b_list) / len(b_list), 4)},
        "difference_pp": round(diff.mean * 100, 2), "ci95_pp": [round(diff.low * 100, 2), round(diff.high * 100, 2)],
        "discordant": counts, "sign_test_p": round(stats.sign_test_pvalue(counts["only_a"], counts["only_b"]), 4),
        "verdict": stats.verdict(diff),
    }


def compare_numeric(observations: list[dict], base: str, other: str, extract) -> dict[str, Any] | None:
    """Разница средних (латентность, токены) B − A: чем меньше, тем лучше."""
    def per_case(arm: str) -> dict[str, float]:
        by_case: dict[str, list[float]] = {}
        for o in observations:
            value = extract(o)
            if o["arm"] == arm and value is not None:
                by_case.setdefault(o["case_id"], []).append(value)
        return {c: sum(v) / len(v) for c, v in by_case.items()}

    a, b = per_case(base), per_case(other)
    cases = sorted(set(a) & set(b))
    if not cases:
        return None
    diff = stats.bootstrap_difference([a[c] for c in cases], [b[c] for c in cases])
    mean_a, mean_b = sum(a.values()) / len(a), sum(b.values()) / len(b)
    return {
        "cases": len(cases), base: round(mean_a, 1), other: round(mean_b, 1), "difference": round(diff.mean, 1),
        "ci95": [round(diff.low, 1), round(diff.high, 1)], "verdict": stats.verdict(diff, higher_is_better=False),
    }


def arm_block(observations: list[dict], arm: str) -> dict[str, Any]:
    mine = [o for o in observations if o["arm"] == arm]
    latency = [o["plan_latency_ms"] for o in mine if o["plan_latency_ms"] is not None]
    tokens = [o["tokens"] for o in mine if o["tokens"]]

    def rate(key: str, group: str | None = None, behavior: str | None = None) -> dict[str, Any]:
        values = [o[key] for o in mine if o[key] is not None and (group is None or o["group"] == group) and (behavior is None or o["expected_behavior"] == behavior)]
        return {"hits": sum(values), "total": len(values), "rate": scoring.rate(sum(values), len(values))}

    docs = [o["context"] for o in mine if "hit_at_3" in o["context"]]
    return {
        "observations": len(mine), "errors": sum(1 for o in mine if o["error"]),
        "plan_accuracy": rate("plan_ok"), "plan_metric_accuracy": rate("metric_ok"), "chart_type_accuracy": rate("chart_ok"),
        "invalid_rejection_rate": rate("behavior_ok", behavior="reject"), "clarification_rate": rate("behavior_ok", behavior="clarify"),
        "behavior_accuracy": rate("behavior_ok"),
        "by_group": {g: {"plan_accuracy": rate("plan_ok", g), "plan_metric_accuracy": rate("metric_ok", g), "behavior_accuracy": rate("behavior_ok", g)} for g in sorted({o["group"] for o in mine})},
        "plan_latency_ms": scoring.stats(latency),
        "llm_tokens": {
            "input_mean": round(sum(t["input"] for t in tokens) / len(tokens), 1) if tokens else None,
            "output_mean": round(sum(t["output"] for t in tokens) / len(tokens), 1) if tokens else None,
            "total_mean": round(sum(t["total"] for t in tokens) / len(tokens), 1) if tokens else None,
            "total_sum": sum(t["total"] for t in tokens),
        },
        "embedding_tokens_mean": round(sum(o["embedding_tokens"] for o in mine) / len(mine), 1) if mine else None,
        "context_retrieval": {
            "hit_at_3": {"hits": sum(d["hit_at_3"] for d in docs), "total": len(docs), "rate": scoring.rate(sum(d["hit_at_3"] for d in docs), len(docs))},
            "hit_at_5": {"hits": sum(d["hit_at_5"] for d in docs), "total": len(docs), "rate": scoring.rate(sum(d["hit_at_5"] for d in docs), len(docs))},
            "section_hit_at_3": {"hits": sum(d["section_hit_at_3"] for d in docs), "total": len(docs), "rate": scoring.rate(sum(d["section_hit_at_3"] for d in docs), len(docs))},
        } if docs else None,
    }


def build_comparisons(observations: list[dict], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for base, other in pairs:
        name = f"{other}_vs_{base}"
        out[name] = {
            "flags": {
                key: {subset: compare_flag(observations, base, other, flag, group) for subset, group in SUBSETS.items()}
                for key, flag in (("plan_accuracy", "plan_ok"), ("plan_metric_accuracy", "metric_ok"), ("chart_type_accuracy", "chart_ok"), ("behavior_accuracy", "behavior_ok"))
            },
            "invalid_rejection": compare_flag([o for o in observations if o["expected_behavior"] == "reject"], base, other, "behavior_ok", None),
            "latency_ms": compare_numeric(observations, base, other, lambda o: o["plan_latency_ms"]),
            "llm_tokens_total": compare_numeric(observations, base, other, lambda o: (o["tokens"] or {}).get("total")),
            "embedding_tokens": compare_numeric(observations, base, other, lambda o: o["embedding_tokens"]),
        }
    return out


# --------------------------------------------------------------------------- вывод (формируется по правилу, не вручную)
def _pp(value: float) -> str:
    return f"{value:+.1f}".replace("+0.0", "0.0").replace("-0.0", "0.0") + " п.п."


def _sentence(title: str, c: dict[str, Any] | None, a: str = "A", b: str = "B") -> str:
    if c is None:
        return f"{title}: нет данных для сравнения."
    low, high = c["ci95_pp"]
    core = (
        f"{title}: {a} {c[a]['rate'] * 100:.1f}% ({c[a]['hits']}/{c['observations']}), {b} {c[b]['rate'] * 100:.1f}% ({c[b]['hits']}/{c['observations']}); "
        f"разница {b}−{a} {_pp(c['difference_pp'])}, 95% ДИ [{low:+.1f}; {high:+.1f}] п.п., расходящихся пар: только {a} — {c['discordant']['only_a']}, только {b} — {c['discordant']['only_b']}, "
        f"тест знаков p={c['sign_test_p']:.3f}."
    )
    tail = {
        "confirmed_better": f" Улучшение {b} подтверждено: интервал целиком выше нуля.",
        "confirmed_worse": f" {b} хуже {a}: интервал целиком ниже нуля.",
        "not_confirmed": " Разница не подтверждена: интервал включает ноль.",
        "no_difference": " Различий нет.",
    }[c["verdict"]]
    return core + tail


def conclude(comparisons: dict[str, Any], arms: dict[str, Any]) -> dict[str, Any]:
    """Итоговый вывод. «RAG лучше» допускается только при подтверждённом улучшении Plan Accuracy на всех случаях."""
    main = comparisons["B_vs_A"]
    primary = main["flags"]["plan_accuracy"]["all"]
    rag_dependent = main["flags"]["plan_accuracy"]["rag_terminology"]
    rejection = main["invalid_rejection"]
    headroom = round((1 - arms["A"]["plan_accuracy"]["rate"]) * 100, 1) if arms["A"]["plan_accuracy"]["rate"] is not None else None
    code = primary["verdict"] if primary else "not_confirmed"
    headline = {
        "confirmed_better": "Гипотеза подтверждена: контекст RAG в промпте повысил Analysis Plan Accuracy.",
        "confirmed_worse": "Гипотеза опровергнута: контекст RAG в промпте снизил Analysis Plan Accuracy.",
        "not_confirmed": "Гипотеза «RAG улучшает построение AnalysisPlan» данными этого эксперимента не подтверждается.",
        "no_difference": "Гипотеза не подтверждена: Analysis Plan Accuracy в конфигурациях A и B совпала.",
    }[code]
    lines = [
        headline,
        _sentence("Analysis Plan Accuracy (все ответные случаи)", primary),
        _sentence("RAG-dependent случаи", rag_dependent),
        _sentence("Invalid request rejection rate", rejection),
    ]
    caveats = []
    if headroom is not None and headroom <= 5.0 and code != "confirmed_better":
        caveats.append(f"Потолок: конфигурация A уже достигает {100 - headroom:.1f}%, запас для улучшения не более {headroom:.1f} п.п., поэтому эксперимент почти не способен показать выигрыш RAG на этом наборе.")
    if code != "confirmed_better":
        caveats.append("Отсутствие подтверждённой разницы не доказывает, что RAG бесполезен: набор из 30 синтетических случаев и малое число расходящихся пар дают мало статистической силы.")
    return {"verdict": code, "headroom_pp": headroom, "rag_better_claim": code == "confirmed_better", "statements": lines, "caveats": caveats}


# --------------------------------------------------------------------------- прогон
def run_ab(repeats: int = DEFAULT_REPEATS, arms: tuple[str, ...] = ("A", "B", "A0"), log=print) -> dict[str, Any]:
    from datastory.config import get_settings

    golden = load_golden()
    started_at, started = datetime.now(timezone.utc), time.perf_counter()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = build_environment(golden, Path(tmp), max_tokens=MAX_TOKENS)
        try:
            if env.llm is None:
                raise RuntimeError(f"A/B эксперимент невозможен: {env.llm_reason}. Задайте OPENAI_API_KEY.")
            recorder = RecordingLLM(env.llm)
            runners = {
                arm: AnalysisRunner(WorkflowDeps.of(env.client, recorder, env.kb, planner_rag=ARMS[arm]["rag"]), max_threads=len(golden.cases) * len(arms) * repeats + 5)
                for arm in arms
            }
            log(f"A/B: {len(golden.cases)} случаев × {repeats} повторов × {len(arms)} конфигураций; LLM {env.llm_model}, temperature 0, max_tokens {MAX_TOKENS}")
            observations: list[dict[str, Any]] = []
            for repeat in range(repeats):
                for index, case in enumerate(golden.cases):
                    order = [arms[(index + repeat + shift) % len(arms)] for shift in range(len(arms))]  # порядок чередуется: смещение прогрева и кэша
                    for position, arm in enumerate(order):
                        observations.append(observe(env, runners[arm], recorder, case, arm, repeat, position))
                log(f"  повтор {repeat + 1}/{repeats} завершён")
            system_hashes = {o["prompt"]["system_sha"] for o in observations if o["prompt"]["system_sha"]}
            settings = get_settings()
            arm_data = {arm: arm_block(observations, arm) for arm in arms}
            pairs = [("A", "B")] + ([("A", "A0")] if "A0" in arms else [])
            comparisons = build_comparisons(observations, pairs)
            result = {
                "schema_version": SCHEMA_VERSION, "generated_at": started_at.isoformat(timespec="seconds"), "duration_s": round(time.perf_counter() - started, 1),
                "hypothesis": "Использование RAG улучшает правильность интерпретации доменных показателей и построения AnalysisPlan.",
                "config": {
                    "llm_model": env.llm_model, "temperature": recorder.temperature, "max_tokens": recorder.max_tokens, "repeats": repeats,
                    "cases": len(golden.cases), "golden_sha256": file_sha256(GOLDEN_PATH), "dataset": golden.dataset,
                    "embedding_provider": env.kb.embedder.name, "identical_system_prompt": len(system_hashes) == 1, "system_prompt_sha": sorted(system_hashes),
                    "arms": {arm: {"title": ARMS[arm]["title"], **vars(ARMS[arm]["rag"])} for arm in arms},
                    "langsmith_tracing": settings.langsmith_tracing and bool(settings.langsmith_api_key),
                },
                "arms": arm_data, "comparisons": comparisons, "conclusion": conclude(comparisons, arm_data), "observations": observations,
            }
        finally:
            env.close()
    return result


def write_ab(result: dict[str, Any], path: Path = AB_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path
