"""Ограниченный эксперимент по выбору гиперпараметров LLM планировщика: temperature 0 против 0.4.

Не исследование: один параметр (temperature), две конфигурации, 10 golden cases, несколько повторов каждого случая. top_p, число токенов
и другие комбинации не перебираются. Одинаково в обеих конфигурациях: модель, системный и пользовательский промпты, RAG, набор данных,
max output tokens, golden cases; меняется только temperature.

Правило выбора задано ДО запуска и применяется кодом (decide), а не вручную:
1. если Analysis Plan Accuracy у одной конфигурации подтверждённо выше (95% ДИ разницы по случаям целиком по одну сторону от нуля) — выбирается она;
2. иначе выбирается конфигурация с более стабильной структурой вывода (доля повторов, совпавших с типичной структурой случая);
3. при равенстве — с меньшими задержкой и числом токенов; при равенстве и здесь — temperature 0 (воспроизводимость).
"""
from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datastory.evaluation import ab_test, scoring, stats
from datastory.evaluation.golden import GOLDEN_PATH, file_sha256, load_golden
from datastory.evaluation.report import HP_PATH
from datastory.evaluation.runner import build_environment
from datastory.llm.client import get_llm
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.runner import AnalysisRunner

SCHEMA_VERSION = 1
DEFAULT_REPEATS = 5
MAX_OUTPUT_TOKENS = 1024
# 10 случаев выбраны заранее по группам (не по результатам): 2 интерпретация, 2 динамика, 1 сравнение, 1 фильтры, 2 терминология, 2 недопустимые/неоднозначные
CASE_IDS = ("mi-01", "mi-03", "ts-01", "ts-04", "cc-01", "fl-01", "rg-02", "rg-04", "iv-01", "iv-04")
CONFIGS = {"T0": 0.0, "T04": 0.4}
TITLES = {"T0": "Конфигурация 1: temperature = 0", "T04": "Конфигурация 2: temperature = 0.4"}


# --------------------------------------------------------------------------- метрики стабильности структуры
def _by_case(observations: list[dict], arm: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for o in observations:
        if o["arm"] == arm:
            grouped.setdefault(o["case_id"], []).append(o)
    return grouped


def agreement(values: list[Any]) -> float:
    """Доля значений, совпавших с самым частым (1.0 — все повторы одинаковы)."""
    return Counter(values).most_common(1)[0][1] / len(values) if values else 0.0


def stability_per_case(observations: list[dict], arm: str, key) -> dict[str, float]:
    """key(observation) -> значение структуры; ошибка выполнения (нет структуры) считается отдельным значением «error»."""
    return {case: agreement([key(o) for o in items]) for case, items in _by_case(observations, arm).items()}


def failed(o: dict) -> bool:
    """Модель не дала пригодного структурного ответа: ошибка выполнения или переход на правила после сбоя модели (в том числе обрезание по лимиту)."""
    return bool(o["error"] or o.get("llm_fallback") or o.get("truncated"))


def signature(o: dict) -> str:
    return "error" if failed(o) or o["signature"] is None else o["signature"]


def arm_summary(observations: list[dict], arm: str, repeats: int) -> dict[str, Any]:
    mine = [o for o in observations if o["arm"] == arm]
    full = stability_per_case(observations, arm, signature)
    behavior = stability_per_case(observations, arm, lambda o: o["predicted_behavior"] or "error")
    metric = stability_per_case(observations, arm, lambda o: o["predicted_metric"] or "none")
    answer_cases = {o["case_id"] for o in mine if o["expected_behavior"] == "answer"}
    plan = [o["plan_ok"] for o in mine if o["plan_ok"] is not None]
    behavior_ok = [o["behavior_ok"] for o in mine if o["behavior_ok"] is not None]
    valid = [o for o in mine if not failed(o)]
    latency = [o["plan_latency_ms"] for o in mine if o["plan_latency_ms"] is not None]
    tokens = [o["tokens"] for o in mine if o["tokens"]]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "observations": len(mine), "errors": len(mine) - len(valid), "truncated": sum(1 for o in mine if o.get("truncated")), "valid_output_rate": scoring.rate(len(valid), len(mine)),
        "plan_accuracy": {"hits": sum(plan), "total": len(plan), "rate": scoring.rate(sum(plan), len(plan))},
        "behavior_accuracy": {"hits": sum(behavior_ok), "total": len(behavior_ok), "rate": scoring.rate(sum(behavior_ok), len(behavior_ok))},
        "structure_agreement": mean(list(full.values())), "behavior_agreement": mean(list(behavior.values())), "metric_agreement": mean(list(metric.values())),
        "identical_cases": {"cases": sum(1 for v in full.values() if v == 1.0), "total": len(full)},
        "answer_cases_identical": {"cases": sum(1 for c in answer_cases if full[c] == 1.0), "total": len(answer_cases)},
        "distinct_structures_per_case": {case: len({signature(o) for o in items}) for case, items in sorted(_by_case(observations, arm).items())},
        "plan_latency_ms": scoring.stats(latency),
        "llm_tokens": {
            "input_mean": round(sum(t["input"] for t in tokens) / len(tokens), 1) if tokens else None,
            "output_mean": round(sum(t["output"] for t in tokens) / len(tokens), 1) if tokens else None,
            "output_max": max((t["output"] for t in tokens), default=None), "total_mean": round(sum(t["total"] for t in tokens) / len(tokens), 1) if tokens else None,
            "llm_calls_mean": round(sum(t["llm_calls"] for t in tokens) / len(tokens), 2) if tokens and all("llm_calls" in t for t in tokens) else None,
        },
        "repeats": repeats,
    }


def compare(observations: list[dict], base: str, other: str) -> dict[str, Any]:
    """other − base по метрикам; для стабильности единица перевыборки — случай."""
    base_stab, other_stab = stability_per_case(observations, base, signature), stability_per_case(observations, other, signature)
    cases = sorted(set(base_stab) & set(other_stab))
    stability = stats.bootstrap_difference([base_stab[c] for c in cases], [other_stab[c] for c in cases])
    return {
        "plan_accuracy": ab_test.compare_flag(observations, base, other, "plan_ok", None),
        "behavior_accuracy": ab_test.compare_flag(observations, base, other, "behavior_ok", None),
        "structure_agreement": {
            base: round(sum(base_stab[c] for c in cases) / len(cases), 4), other: round(sum(other_stab[c] for c in cases) / len(cases), 4),
            "difference_pp": round(stability.mean * 100, 2), "ci95_pp": [round(stability.low * 100, 2), round(stability.high * 100, 2)], "verdict": stats.verdict(stability),
        },
        "latency_ms": ab_test.compare_numeric(observations, base, other, lambda o: o["plan_latency_ms"]),
        "llm_tokens_total": ab_test.compare_numeric(observations, base, other, lambda o: (o["tokens"] or {}).get("total")),
        "llm_tokens_output": ab_test.compare_numeric(observations, base, other, lambda o: (o["tokens"] or {}).get("output")),
    }


# --------------------------------------------------------------------------- выбор
def decide(arms: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    """Применяет правило выбора из докстринга модуля. Возвращает выбор и причины (тексты строятся из чисел результата)."""
    base, other = "T0", "T04"
    accuracy = comparison["plan_accuracy"]
    stability = comparison["structure_agreement"]
    reasons: list[str] = []
    selected, rule = None, None
    if accuracy and accuracy["verdict"] in ("confirmed_better", "confirmed_worse"):
        selected, rule = (other if accuracy["verdict"] == "confirmed_better" else base), 1
        reasons.append(
            f"Analysis Plan Accuracy: T0 {accuracy[base]['rate'] * 100:.1f}% ({accuracy[base]['hits']}/{accuracy['observations']}), T0.4 {accuracy[other]['rate'] * 100:.1f}% "
            f"({accuracy[other]['hits']}/{accuracy['observations']}); разница подтверждена (95% ДИ [{accuracy['ci95_pp'][0]:+.1f}; {accuracy['ci95_pp'][1]:+.1f}] п.п.)."
        )
    else:
        if accuracy:
            reasons.append(
                f"Analysis Plan Accuracy не различает конфигурации: T0 {accuracy[base]['rate'] * 100:.1f}% ({accuracy[base]['hits']}/{accuracy['observations']}), "
                f"T0.4 {accuracy[other]['rate'] * 100:.1f}% ({accuracy[other]['hits']}/{accuracy['observations']}), разница {accuracy['difference_pp']:+.1f} п.п., "
                f"95% ДИ [{accuracy['ci95_pp'][0]:+.1f}; {accuracy['ci95_pp'][1]:+.1f}] п.п. включает ноль."
            )
        s0, s4 = stability[base], stability[other]
        if abs(s0 - s4) > 1e-9:
            selected, rule = (base if s0 > s4 else other), 2
            reasons.append(
                f"Стабильность структуры вывода (совпадение повторов с типичной структурой случая): T0 {s0 * 100:.1f}%, T0.4 {s4 * 100:.1f}% "
                f"(разница T0.4 − T0 {stability['difference_pp']:+.1f} п.п., 95% ДИ [{stability['ci95_pp'][0]:+.1f}; {stability['ci95_pp'][1]:+.1f}]); выбрана более стабильная."
            )
        else:
            latency, tokens = comparison["latency_ms"], comparison["llm_tokens_total"]
            cost = [(base, latency[base] if latency else 0, tokens[base] if tokens else 0), (other, latency[other] if latency else 0, tokens[other] if tokens else 0)]
            if cost[0][1:] != cost[1][1:]:
                selected, rule = min(cost, key=lambda c: (c[1], c[2]))[0], 3
                reasons.append("Стабильность структуры одинакова; выбрана конфигурация с меньшей задержкой и числом токенов.")
            else:
                selected, rule = base, 3
                reasons.append("Метрики совпали; выбрана temperature 0 как более воспроизводимая.")
    if arms[base]["errors"] or arms[other]["errors"]:
        reasons.append(
            f"Запросов без пригодного структурного ответа модели: T0 — {arms[base]['errors']}, T0.4 — {arms[other]['errors']} из {arms[base]['observations']} в каждой "
            f"(обрезано лимитом токенов: {arms[base].get('truncated', 0)} и {arms[other].get('truncated', 0)})."
        )
    for key, title in (("latency_ms", "Задержка"), ("llm_tokens_total", "Токены LLM")):
        c = comparison[key]
        if c:
            reasons.append(f"{title}: T0 {c[base]:,.1f}, T0.4 {c[other]:,.1f} (разница T0.4 − T0 {c['difference']:+,.1f}; 95% ДИ [{c['ci95'][0]:+,.1f}; {c['ci95'][1]:+,.1f}]) — {ab_test_verdict(c['verdict'])}.".replace(",", " "))
    caveats: list[str] = []
    if rule == 2 and stability["verdict"] not in ("confirmed_better", "confirmed_worse"):
        caveats.append(
            "Выбор сделан по разнице стабильности, чей 95% ДИ включает ноль: конфигурации на этих данных практически равноценны, а выбор опирается на точечную оценку "
            "(число запросов без пригодного ответа и различающиеся структуры). Его стоит перепроверить на большем числе повторов (`--hyperparams --repeats 10`)."
        )
    for key, title in (("latency_ms", "задержка"), ("llm_tokens_total", "число токенов")):
        c = comparison[key]
        if not c:
            continue
        if c["verdict"] == "confirmed_worse" and selected == other:
            caveats.append(f"Выбранная конфигурация (T0.4) подтверждённо хуже по показателю «{title}»: {c['difference']:+,.1f} на запрос относительно T0.".replace(",", " "))
        if c["verdict"] == "confirmed_better" and selected == base:
            caveats.append(f"Выбранная конфигурация (T0) подтверждённо хуже по показателю «{title}»: T0.4 экономнее на {-c['difference']:,.1f} на запрос.".replace(",", " "))
    return {"selected": selected, "temperature": CONFIGS[selected], "rule_step": rule, "reasons": reasons, "caveats": caveats}


def ab_test_verdict(code: str) -> str:
    return {"confirmed_better": "T0.4 подтверждённо меньше", "confirmed_worse": "T0.4 подтверждённо больше", "not_confirmed": "различие не подтверждено", "no_difference": "различий нет"}[code]


def output_token_choice(arms: dict[str, Any], cap: int) -> dict[str, Any]:
    observed = max(a["llm_tokens"]["output_max"] or 0 for a in arms.values())
    errors = sum(a["errors"] for a in arms.values())
    truncated = sum(a["truncated"] for a in arms.values())
    requests = sum(a["observations"] for a in arms.values())
    base = (
        f"Максимум, который модель потратила на ответ планировщика в запросах, где usage удалось получить, — {observed} токенов при лимите {cap} ({observed / cap * 100:.0f}% лимита); "
        f"ответов, обрезанных лимитом, — {truncated} из {requests}, запросов без пригодного ответа модели — {errors}."
    )
    tail = (
        " Нормальные ответы укладываются в лимит с запасом; лимит ограничивает только вырожденные ответы (они по-прежнему приводят к плану по правилам с предупреждением, а не к сбою)."
        if truncated else " Лимит с запасом покрывает нормальные ответы и защищает от неконтролируемо длинных."
    )
    return {"max_output_tokens": cap, "observed_max_output_tokens": observed, "share_of_cap": round(observed / cap, 3), "errors": errors, "truncated": truncated, "reason": base + tail}


# --------------------------------------------------------------------------- прогон
def run_hyperparams(repeats: int = DEFAULT_REPEATS, log=print) -> dict[str, Any]:
    golden = load_golden()
    cases = {c.id: c for c in golden.cases}
    missing = [i for i in CASE_IDS if i not in cases]
    if missing:
        raise RuntimeError(f"В golden cases нет выбранных случаев: {', '.join(missing)}")
    started_at, started = datetime.now(timezone.utc), time.perf_counter()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = build_environment(golden, Path(tmp), max_tokens=MAX_OUTPUT_TOKENS)
        try:
            if env.llm is None:
                raise RuntimeError(f"Эксперимент невозможен: {env.llm_reason}. Задайте OPENAI_API_KEY.")
            recorders, runners = {}, {}
            for arm, temperature in CONFIGS.items():
                llm = get_llm(env.settings, usage=env.usage, max_tokens=MAX_OUTPUT_TOKENS, temperature=temperature)
                recorders[arm] = ab_test.RecordingLLM(llm)
                runners[arm] = AnalysisRunner(WorkflowDeps.of(env.client, recorders[arm], env.kb), max_threads=len(CASE_IDS) * len(CONFIGS) * repeats + 5)
            log(f"Гиперпараметры: {len(CASE_IDS)} случаев × {repeats} повторов × {len(CONFIGS)} конфигурации; LLM {env.llm_model}, max_tokens {MAX_OUTPUT_TOKENS}")
            observations: list[dict[str, Any]] = []
            arms_order = list(CONFIGS)
            for repeat in range(repeats):
                for index, case_id in enumerate(CASE_IDS):
                    order = arms_order if (index + repeat) % 2 == 0 else arms_order[::-1]  # порядок чередуется: смещение прогрева и кэша
                    for position, arm in enumerate(order):
                        observations.append(ab_test.observe(env, runners[arm], recorders[arm], cases[case_id], arm, repeat, position))
                log(f"  повтор {repeat + 1}/{repeats} завершён")
            return assemble(observations, repeats, env.llm_model, golden.dataset, env.kb.embedder.name, started_at, time.perf_counter() - started)
        finally:
            env.close()


def assemble(
    observations: list[dict[str, Any]], repeats: int, llm_model: str, dataset: str, embedding_provider: str, started_at: datetime, duration_s: float
) -> dict[str, Any]:
    """Итоговый словарь результата целиком из сырых наблюдений: метрики, сравнение, выбор и причины — производные, их можно пересчитать без обращений к OpenAI."""
    system_hashes = {o["prompt"]["system_sha"] for o in observations if o["prompt"]["system_sha"]}
    arms = {arm: arm_summary(observations, arm, repeats) for arm in CONFIGS}
    comparison = compare(observations, "T0", "T04")
    decision = decide(arms, comparison)
    return {
        "schema_version": SCHEMA_VERSION, "generated_at": started_at.isoformat(timespec="seconds"), "duration_s": round(duration_s, 1),
        "config": {
            "llm_model": llm_model, "max_output_tokens": MAX_OUTPUT_TOKENS, "repeats": repeats, "case_ids": list(CASE_IDS), "golden_sha256": file_sha256(GOLDEN_PATH),
            "dataset": dataset, "embedding_provider": embedding_provider, "identical_system_prompt": len(system_hashes) == 1,
            "system_prompt_sha": sorted(system_hashes), "configs": {arm: {"title": TITLES[arm], "temperature": t} for arm, t in CONFIGS.items()},
            "not_studied": "top_p, frequency/presence penalty и другие комбинации не исследовались: ограниченный эксперимент для MVP",
        },
        "arms": arms, "comparison": comparison, "decision": decision, "max_output_tokens_choice": output_token_choice(arms, MAX_OUTPUT_TOKENS),
        "final": {"llm_model": llm_model, "temperature": decision["temperature"], "max_output_tokens": MAX_OUTPUT_TOKENS},
        "observations": observations,
    }


def recompute(result: dict[str, Any]) -> dict[str, Any]:
    """Пересчитывает производные значения из сохранённых наблюдений (после изменения правил подсчёта); сами наблюдения и дата запуска не меняются."""
    cfg = result["config"]
    return assemble(
        result["observations"], cfg["repeats"], cfg["llm_model"], cfg["dataset"], cfg["embedding_provider"],
        datetime.fromisoformat(result["generated_at"]), result["duration_s"],
    )


def write_hp(result: dict[str, Any], path: Path = HP_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path
