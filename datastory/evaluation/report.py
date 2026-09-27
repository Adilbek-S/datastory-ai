"""EVALS.md из результатов реального запуска (evaluation/results/latest.json).

Отчёт строится ТОЛЬКО из словаря результатов: ни одного числа здесь не задано вручную, шаблон подставляет значения из
результатов. Метрика, которую не удалось вычислить (нет ключа OpenAI, нет подходящих случаев), показывается как «—»
с причиной, а не заменяется правдоподобным значением.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from datastory.evaluation.golden import EVAL_DIR, RESULTS_PATH

REPORT_PATH = EVAL_DIR.parent / "EVALS.md"
AB_PATH = EVAL_DIR / "results" / "ab_test.json"
HP_PATH = EVAL_DIR / "results" / "hyperparameters.json"
HISTORY_DIR = EVAL_DIR / "results" / "history"
SHA_MARK = "sha256 результатов:"

METRIC_ROWS = (
    ("retrieval_hit_at_3", "Retrieval Hit@3", "нужный документ базы знаний среди трёх найденных фрагментов"),
    ("retrieval_section_hit_at_3", "Retrieval Hit@3 (раздел)", "нужный раздел нужного документа среди трёх фрагментов (строже)"),
    ("plan_accuracy", "Analysis Plan Accuracy", "совпали показатель, колонки и группировка с golden case"),
    ("plan_full_accuracy", "Plan Accuracy + фильтры", "то же и совпали значения фильтров (строже)"),
    ("plan_metric_accuracy", "  · показатель", "совпал показатель"),
    ("plan_columns_accuracy", "  · колонки", "совпал набор колонок (показатель + группировка + фильтры)"),
    ("plan_grouping_accuracy", "  · группировка", "совпала группировка"),
    ("plan_filters_accuracy", "  · фильтры", "совпали фильтры (только случаи с фильтрами)"),
    ("chart_type_accuracy", "Chart type accuracy", "совпал тип графика (отдельно от Plan Accuracy)"),
    ("numeric_accuracy", "Numeric Accuracy", "числа MCP при эталонном плане совпали с расчётом на чистом Python (допуск float)"),
    ("numeric_accuracy_system_plan", "Numeric Accuracy (план системы)", "числа MCP по плану, который построила система, совпали с эталоном"),
    ("invalid_rejection_rate", "Invalid request rejection rate", "недопустимый запрос отклонён: показатель не выдуман"),
    ("clarification_rate", "Clarification rate", "неоднозначный запрос закончился вопросом пользователю"),
    ("false_rejection_rate", "False rejection rate", "допустимый запрос ошибочно отклонён (чем меньше, тем лучше)"),
)
GROUP_TITLES = {
    "metric_interpretation": "1. Интерпретация показателя", "time_series": "2. Динамика во времени", "category_comparison": "3. Сравнение категорий",
    "filters": "4. Фильтры", "rag_terminology": "5. Терминология из базы знаний", "invalid_ambiguous": "6. Недопустимые и неоднозначные запросы",
}


def _pct(block: dict[str, Any], reason: str = "не вычислялась") -> str:
    if not block or block.get("total", 0) == 0:
        return "—" if reason == "—" else f"— ({reason})"
    return f"{block['rate'] * 100:.1f}% ({block['hits']}/{block['total']})"


def _num(value: Any, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:,.1f}".replace(",", " ")
        text = text[:-2] if text.endswith(".0") else text
    else:
        text = f"{value:,}".replace(",", " ")
    return text + suffix


def _case_status(case: dict[str, Any]) -> str:
    if case["error"]:
        return "ошибка"
    plan = case.get("plan") or {}
    if not plan.get("computed"):
        return "план не запускался"
    return "ок" if plan.get("accurate", plan.get("behavior_ok")) and plan.get("behavior_ok", True) else "не совпало"


def _history_rows(history: list[dict[str, Any]]) -> list[str]:
    lines = ["| Дата запуска (UTC) | Коммит | LLM | Plan Accuracy | Numeric Accuracy | Rejection rate | Hit@3 | План, мс (среднее) | Токены LLM |", "|---|---|---|---|---|---|---|---|---|"]
    for past in history:
        s, e = past["summary"]["overall"], past["environment"]
        lines.append(
            f"| {past['generated_at']} | `{(e['git_commit'] or '?')[:7]}` | {e['llm_model'] or '—'} | {_pct(s['plan_accuracy'], '—')} | {_pct(s['numeric_accuracy'], '—')} | "
            f"{_pct(s['invalid_rejection_rate'], '—')} | {_pct(s['retrieval_hit_at_3'], '—')} | {_num(s['plan_latency_ms']['mean'])} | {_num(s['llm_tokens']['total'])} |"
        )
    return lines


VERDICT_TITLES = {
    "confirmed_better": "улучшение подтверждено", "confirmed_worse": "ухудшение подтверждено",
    "not_confirmed": "не подтверждено (ДИ включает 0)", "no_difference": "различий нет",
}
AB_SUBSETS = (("all", "все ответные случаи"), ("rag_terminology", "RAG-dependent (группа 5)"), ("metric_interpretation", "интерпретация показателя (группа 1)"))


def _signed(value: float) -> str:
    return f"{value + 0.0:+.1f}" if round(value, 1) else "0.0"  # без «-0.0»


def _pp_ci(c: dict[str, Any]) -> str:
    return f"{_signed(c['difference_pp'])} п.п. [{_signed(c['ci95_pp'][0])}; {_signed(c['ci95_pp'][1])}]"


def _flag_row(title: str, c: dict[str, Any] | None, a: str = "A", b: str = "B") -> str:
    if c is None:
        return f"| {title} | — | — | — | — | — |"

    def fmt(arm: str) -> str:
        return f"{c[arm]['rate'] * 100:.1f}% ({c[arm]['hits']}/{c['observations']})"

    return f"| {title} | {fmt(a)} | {fmt(b)} | {_pp_ci(c)} | {c['sign_test_p']:.3f} | {VERDICT_TITLES[c['verdict']]} |"


def _numeric_row(title: str, c: dict[str, Any] | None, a: str = "A", b: str = "B") -> str:
    if c is None:
        return f"| {title} | — | — | — | — |"
    delta = f"{c['difference']:+,.1f} [{c['ci95'][0]:+,.1f}; {c['ci95'][1]:+,.1f}]".replace(",", " ")
    return f"| {title} | {_num(c[a])} | {_num(c[b])} | {delta} | {VERDICT_TITLES[c['verdict']]} |"


def _case_rates(ab: dict[str, Any], arm: str, group: str) -> dict[str, tuple[int, int]]:
    rates: dict[str, list[bool]] = {}
    for o in ab["observations"]:
        if o["arm"] == arm and o["group"] == group:
            key = "plan_ok" if o["plan_ok"] is not None else "behavior_ok"
            rates.setdefault(o["case_id"], []).append(bool(o[key]))
    return {case: (sum(v), len(v)) for case, v in rates.items()}


def render_ab(ab: dict[str, Any], ab_sha256: str, queries: dict[str, str] | None = None) -> list[str]:
    """Секция A/B эксперимента для EVALS.md: только значения из ab_test.json, вывод — из conclusion (вычислен по правилу ДИ)."""
    cfg, arms, cmp_ = ab["config"], ab["arms"], ab["comparisons"]
    main, queries = cmp_["B_vs_A"], queries or {}
    prompt_note = ("идентичен во всех вызовах (sha256 `" + cfg["system_prompt_sha"][0] + "`)") if cfg["identical_system_prompt"] else ("НЕ идентичен: " + ", ".join(cfg["system_prompt_sha"]))
    lines = [
        "", "## A/B эксперимент: влияет ли RAG на построение AnalysisPlan", "",
        f"> Сформировано автоматически из `evaluation/results/ab_test.json` (sha256 `{ab_sha256}`); числа не редактировались вручную.", "",
        f"**Гипотеза.** {ab['hypothesis']}", "",
        "| Параметр | Значение |", "|---|---|",
        f"| Дата запуска (UTC) | {ab['generated_at']}, длительность {ab['duration_s']} с |",
        f"| Конфигурация A | {cfg['arms']['A']['title']}: LLM получает DatasetProfile и user_query, контекст RAG не передаётся |",
        f"| Конфигурация B | {cfg['arms']['B']['title']}: DatasetProfile, user_query и Top-{cfg['arms']['B']['context_hits']} фрагментов базы знаний |",
        f"| Одинаково в A и B | LLM `{cfg['llm_model']}`, temperature {cfg['temperature']}, max_tokens {cfg['max_tokens']}, набор данных `{cfg['dataset']}`, MCP, база знаний, golden cases (sha256 `{cfg['golden_sha256'][:12]}…`) |",
        f"| Системный промпт | {prompt_note} |",
        f"| Объём | {cfg['cases']} случаев × {cfg['repeats']} повторов × {len(arms)} конфигураций; порядок конфигураций чередуется по случаям и повторам |",
        f"| Эмбеддинги | {cfg['embedding_provider']} |",
        "",
        "Проверка плана после ответа LLM (определения показателей, защита от подмены термина) одинакова в A и B и обращается к базе знаний, поэтому A — это «LLM без RAG в промпте», "
        "а не «система без RAG»; конфигурация A0 ниже отключает RAG полностью.", "",
        "### Сравнение A и B", "",
        "Разница B − A с 95% бутстрап-интервалом по случаям (повторы усредняются внутри случая); p — точный тест знаков по расходящимся парам.", "",
        "| Метрика | A | B | B − A [95% ДИ] | p | Вывод |", "|---|---|---|---|---|---|",
    ]
    for key, title in (("plan_accuracy", "Analysis Plan Accuracy"), ("plan_metric_accuracy", "Правильный показатель"), ("chart_type_accuracy", "Тип графика")):
        for subset, label in AB_SUBSETS:
            if key != "plan_accuracy" and subset == "metric_interpretation":
                continue
            lines.append(_flag_row(f"{title}, {label}", main["flags"][key][subset]))
    lines.append(_flag_row("Invalid request rejection rate (4 случая)", main["invalid_rejection"]))
    lines.append(_flag_row("Верное поведение (ответ / отказ / уточнение)", main["flags"]["behavior_accuracy"]["all"]))
    lines += [
        "", "| Ресурсы (среднее на запрос, чем меньше — тем лучше) | A | B | B − A [95% ДИ] | Вывод |", "|---|---|---|---|---|",
        _numeric_row("Задержка построения плана, мс", main["latency_ms"]),
        _numeric_row("Токены LLM (промпт + ответ)", main["llm_tokens_total"]),
        _numeric_row("Токены эмбеддингов", main["embedding_tokens"]),
    ]
    a_tok, b_tok = arms["A"]["llm_tokens"], arms["B"]["llm_tokens"]
    lines += [
        "", f"Токены LLM в среднем на запрос: A — промпт {_num(a_tok['input_mean'])}, ответ {_num(a_tok['output_mean'])}; B — промпт {_num(b_tok['input_mean'])}, ответ {_num(b_tok['output_mean'])}. "
        f"Ошибок выполнения: A — {arms['A']['errors']}, B — {arms['B']['errors']}.",
        "", "### Retrieval Hit@3 для конфигурации B", "",
    ]
    retrieval = arms["B"]["context_retrieval"]
    if retrieval:
        lines += [
            "| Метрика | Значение | Что измеряет |", "|---|---|---|",
            f"| Retrieval Hit@3 | {_pct(retrieval['hit_at_3'])} | нужный документ среди первых 3 фрагментов, переданных LLM |",
            f"| Retrieval Hit@5 | {_pct(retrieval['hit_at_5'])} | нужный документ среди всех 5 переданных фрагментов |",
            f"| Hit@3 (раздел) | {_pct(retrieval['section_hit_at_3'])} | нужный раздел нужного документа среди первых 3 |",
            "", "Значения посчитаны по случаям с ожидаемым документом и по всем повторам.",
        ]
    else:
        lines.append("Не вычислялся: в конфигурации B не было ни одного случая с найденным контекстом.")

    lines += ["", "### RAG-dependent случаи (группа 5: терминология из базы знаний)", "", "| Случай | Запрос | A: план верен | B: план верен |", "|---|---|---|---|"]
    a_rates, b_rates = _case_rates(ab, "A", "rag_terminology"), _case_rates(ab, "B", "rag_terminology")

    def show(rate: tuple[int, int] | None) -> str:
        return "—" if rate is None else f"{rate[0]}/{rate[1]}"

    for case_id in sorted(set(a_rates) | set(b_rates)):
        lines.append(f"| {case_id} | {queries.get(case_id, '')} | {show(a_rates.get(case_id))} | {show(b_rates.get(case_id))} |")
    lines += ["", "### По группам golden cases", "", "| Группа | A: Plan Accuracy | B: Plan Accuracy | A: верное поведение | B: верное поведение |", "|---|---|---|---|---|"]
    for group in sorted(arms["A"]["by_group"], key=lambda g: list(GROUP_TITLES).index(g) if g in GROUP_TITLES else len(GROUP_TITLES)):
        ga, gb = arms["A"]["by_group"][group], arms["B"]["by_group"].get(group, {})
        lines.append(
            f"| {GROUP_TITLES.get(group, group)} | {_pct(ga['plan_accuracy'], '—')} | {_pct(gb.get('plan_accuracy'), '—')} | "
            f"{_pct(ga['behavior_accuracy'], '—')} | {_pct(gb.get('behavior_accuracy'), '—')} |"
        )

    failures: dict[tuple[str, str, str], int] = {}
    for o in ab["observations"]:
        wrong = o["plan_ok"] is False or (o["plan_ok"] is None and o["behavior_ok"] is False)
        if wrong and o["arm"] in ("A", "B"):
            key = (o["case_id"], o["arm"], f"{o['predicted_behavior']}" + (f", {o['predicted_metric']}" if o["predicted_metric"] else ""))
            failures[key] = failures.get(key, 0) + 1
    lines += ["", "### Где конфигурации ошиблись", ""]
    if failures:
        lines += [f"Повторов всего на случай: {cfg['repeats']}. Что построила система вместо ожидаемого:", "", "| Случай | Запрос | Конфигурация | Что построено | Повторов с ошибкой |", "|---|---|---|---|---|"]
        lines += [f"| {c} | {queries.get(c, '')} | {arm} | {what} | {n} |" for (c, arm, what), n in sorted(failures.items())]
    else:
        lines.append("Ошибок в конфигурациях A и B нет.")

    extra = cmp_.get("A0_vs_A")
    if extra:
        lines += [
            "", "### Дополнительно: A0 — RAG выключен полностью (в промпте и при проверке плана)", "",
            "| Метрика | A | A0 | A0 − A [95% ДИ] | p | Вывод |", "|---|---|---|---|---|---|",
            _flag_row("Analysis Plan Accuracy, все ответные случаи", extra["flags"]["plan_accuracy"]["all"], "A", "A0"),
            _flag_row("Analysis Plan Accuracy, RAG-dependent", extra["flags"]["plan_accuracy"]["rag_terminology"], "A", "A0"),
            _flag_row("Invalid request rejection rate", extra["invalid_rejection"], "A", "A0"),
            "", "A0 показывает вклад RAG-проверки плана (определения показателей), а не вклад контекста в промпте.",
        ]

    conclusion = ab["conclusion"]
    lines += ["", "### Вывод", "", f"**{conclusion['statements'][0]}**", ""]
    lines += [f"- {statement}" for statement in conclusion["statements"][1:]]
    if conclusion["caveats"]:
        lines += ["", "Оговорки:", ""] + [f"- {caveat}" for caveat in conclusion["caveats"]]
    lines += [
        "", "Как читать: вывод «RAG лучше» допускается только если 95% доверительный интервал разницы Plan Accuracy целиком выше нуля; "
        "иначе результат называется неподтверждённым. Golden cases синтетические и не независимая выборка, ответы LLM недетерминированы даже при temperature 0 (поэтому каждый случай повторён), "
        "задержка включает сеть OpenAI.",
    ]
    return lines


def _agree_row(title: str, c: dict[str, Any]) -> str:
    return f"| {title} | {c['T0'] * 100:.1f}% | {c['T04'] * 100:.1f}% | {_pp_ci(c)} | {VERDICT_TITLES[c['verdict']]} |"


def _previous_hp(previous: list[dict[str, Any]]) -> list[str]:
    if not previous:
        return []
    lines = ["", "Предыдущие запуски этого эксперимента (сохранены в `evaluation/results/history/`, не отброшены):", "",
             "| Дата запуска (UTC) | Повторы | Plan Accuracy T0 / T0.4 | Стабильность структуры T0 / T0.4 | Запросов без пригодного ответа T0 / T0.4 | Выбор по правилу |", "|---|---|---|---|---|---|"]
    for old in previous:
        a = old["arms"]
        lines.append(
            f"| {old['generated_at']} | {old['config']['repeats']} | {a['T0']['plan_accuracy']['rate'] * 100:.1f}% / {a['T04']['plan_accuracy']['rate'] * 100:.1f}% | "
            f"{a['T0']['structure_agreement'] * 100:.1f}% / {a['T04']['structure_agreement'] * 100:.1f}% | {a['T0']['errors']} / {a['T04']['errors']} | temperature {old['decision']['temperature']} |"
        )
    return lines


def render_hp(hp: dict[str, Any], hp_sha256: str, queries: dict[str, str] | None = None, previous: list[dict[str, Any]] | None = None) -> list[str]:
    """Раздел «Выбор гиперпараметров LLM»: только значения из hyperparameters.json; выбор и причины вычислены кодом по правилу (hyperparams.decide)."""
    cfg, arms, cmp_, decision, final, tokens = hp["config"], hp["arms"], hp["comparison"], hp["decision"], hp["final"], hp["max_output_tokens_choice"]
    queries = queries or {}
    prompt_note = ("идентичен во всех вызовах (sha256 `" + cfg["system_prompt_sha"][0] + "`)") if cfg["identical_system_prompt"] else ("НЕ идентичен: " + ", ".join(cfg["system_prompt_sha"]))
    t0, t4 = arms["T0"], arms["T04"]
    case_list = ", ".join(f"`{c}`" for c in cfg["case_ids"])
    lines = [
        "", "## Выбор гиперпараметров LLM (temperature)", "",
        f"> Сформировано автоматически из `evaluation/results/hyperparameters.json` (sha256 `{hp_sha256}`); числа не редактировались вручную.", "",
        "Небольшой ограниченный эксперимент для MVP: одна переменная (temperature), две конфигурации, 10 golden cases, несколько повторов каждого случая. "
        "Конфигурация для основной версии выбрана по правилу, заданному до запуска, и применена кодом.", "",
        "| Параметр | Значение |", "|---|---|",
        f"| Дата запуска (UTC) | {hp['generated_at']}, длительность {hp['duration_s']} с |",
        f"| Конфигурация 1 | temperature = {cfg['configs']['T0']['temperature']} |",
        f"| Конфигурация 2 | temperature = {cfg['configs']['T04']['temperature']} |",
        f"| Одинаково | LLM `{cfg['llm_model']}`, max output tokens {cfg['max_output_tokens']}, системный и пользовательский промпты, RAG, набор данных `{cfg['dataset']}`, MCP |",
        f"| Системный промпт | {prompt_note} |",
        f"| Случаи ({len(cfg['case_ids'])}) | {case_list}: выбраны заранее по группам, не по результатам; golden cases sha256 `{cfg['golden_sha256'][:12]}…` |",
        f"| Повторы | {cfg['repeats']} на случай и конфигурацию ({t0['observations']} запросов на конфигурацию); порядок конфигураций чередуется |",
        f"| Не исследовалось | {cfg['not_studied']} |",
        "", "### Результаты", "",
        "| Метрика | temperature 0 | temperature 0.4 | 0.4 − 0 [95% ДИ] | Вывод |", "|---|---|---|---|---|",
    ]
    for title, c in (("Analysis Plan Accuracy (ответные случаи)", cmp_["plan_accuracy"]), ("Верное поведение (ответ / отказ / уточнение)", cmp_["behavior_accuracy"])):
        if c:
            lines.append(
                f"| {title} | {c['T0']['rate'] * 100:.1f}% ({c['T0']['hits']}/{c['observations']}) | {c['T04']['rate'] * 100:.1f}% ({c['T04']['hits']}/{c['observations']}) | "
                f"{_pp_ci(c)} | {VERDICT_TITLES[c['verdict']]} |"
            )
        else:
            lines.append(f"| {title} | — | — | — | нет данных |")
    valid0, valid4 = t0["observations"] - t0["errors"], t4["observations"] - t4["errors"]
    lines += [
        f"| Пригодный структурный ответ модели | {t0['valid_output_rate'] * 100:.1f}% ({valid0}/{t0['observations']}) | {t4['valid_output_rate'] * 100:.1f}% ({valid4}/{t4['observations']}) | — | ошибок: {t0['errors']} и {t4['errors']}, из них обрезано лимитом: {t0.get('truncated', 0)} и {t4.get('truncated', 0)} |",
        _agree_row("Стабильность структуры: совпадение повторов с типичной структурой случая", cmp_["structure_agreement"]),
        f"| Случаи, где все повторы дали одну структуру | {t0['identical_cases']['cases']}/{t0['identical_cases']['total']} | {t4['identical_cases']['cases']}/{t4['identical_cases']['total']} | — | — |",
        f"| Стабильность выбранного показателя | {t0['metric_agreement'] * 100:.1f}% | {t4['metric_agreement'] * 100:.1f}% | — | — |",
        f"| Стабильность поведения (ответ / отказ / уточнение) | {t0['behavior_agreement'] * 100:.1f}% | {t4['behavior_agreement'] * 100:.1f}% | — | — |",
        "",
        "| Ресурсы (среднее на запрос плана) | temperature 0 | temperature 0.4 | 0.4 − 0 [95% ДИ] | Вывод |", "|---|---|---|---|---|",
        _numeric_row("Задержка построения плана, мс", cmp_["latency_ms"], "T0", "T04"),
        _numeric_row("Токены LLM (промпт + ответ)", cmp_["llm_tokens_total"], "T0", "T04"),
        _numeric_row("Токены ответа LLM", cmp_["llm_tokens_output"], "T0", "T04"),
        "",
        "«Структура» — поведение, показатель, группировка, фильтры, тип графика и число шагов плана; ошибка выполнения считается отдельным значением. "
        "Стабильность по случаю — доля повторов, совпавших с самым частым вариантом (100% — все повторы одинаковы). Разница — по случаям, бутстрап-интервал 95%.", "",
        "### Структуры по случаям", "", "| Случай | Запрос | Разных структур за повторы, T0 | Разных структур, T0.4 |", "|---|---|---|---|",
    ]
    for case_id in cfg["case_ids"]:
        lines.append(f"| {case_id} | {queries.get(case_id, '')} | {t0['distinct_structures_per_case'].get(case_id, '—')} | {t4['distinct_structures_per_case'].get(case_id, '—')} |")
    lines += [
        "", "### Выбранная конфигурация для основной версии", "",
        "| Параметр | Значение |", "|---|---|",
        f"| Модель | `{final['llm_model']}` |", f"| temperature | **{final['temperature']}** |", f"| max output tokens | **{final['max_output_tokens']}** |",
        "| top_p | не менялся (значение по умолчанию API) |",
        "", f"Правило выбора, шаг {decision['rule_step']} из 3 (1 — точность, 2 — стабильность структуры, 3 — задержка и токены, затем temperature 0). Причины:", "",
        *[f"- {reason}" for reason in decision["reasons"]],
        f"- max output tokens: {tokens['reason']}",
        *(["", "Оговорки к выбору:", "", *[f"- {caveat}" for caveat in decision.get("caveats", [])]] if decision.get("caveats") else []),
        *_previous_hp(previous or []),
        "", "Ограничения: 10 случаев и несколько повторов дают мало статистической силы, поэтому «не подтверждено» означает «на этих данных не различимо», а не «одинаково»; "
        "выбор относится к планировщику `gpt-4o-mini` на синтетическом наборе и не переносится на другие стадии (выводы, чат) без отдельной проверки. "
        "Задержка включает сеть OpenAI.",
    ]
    return lines


def render_report(
    results: dict[str, Any], results_sha256: str, history: list[dict[str, Any]] | None = None, ab: dict[str, Any] | None = None, ab_sha256: str = "",
    hp: dict[str, Any] | None = None, hp_sha256: str = "", previous_hp: list[dict[str, Any]] | None = None,
) -> str:
    env, scope, overall = results["environment"], results["scope"], results["summary"]["overall"]
    llm_note = "" if env["llm_available"] else f"не вычислялась: {env.get('llm_reason', 'LLM недоступна')}, LLM-планирование не запускалось"
    lines = [
        "# EVALS — результаты evaluation DataStory AI",
        "",
        "> Файл сформирован командой `python -m datastory.evaluation` **только** по результатам реального запуска "
        f"(`evaluation/results/latest.json`). Числа не редактировались вручную. {SHA_MARK} `{results_sha256}`.",
        "",
        "## Запуск",
        "",
        "| Параметр | Значение |", "|---|---|",
        f"| Дата запуска (UTC) | {results['generated_at']} |",
        f"| Длительность | {results['duration_s']} с |",
        f"| Случаев | {scope['cases']} из {scope['total_cases']} ({'полный прогон' if scope['complete'] else 'частичный прогон'}) |",
        f"| LLM | {env['llm_model'] or 'не использовалась (' + env.get('llm_reason', 'недоступна') + ')'} |",
        f"| Эмбеддинги | {env['embedding_provider']} ({'семантические' if env['embedding_is_semantic'] else 'офлайн, лексические: не семантика'}) |",
        f"| Набор данных | `{env['dataset']}` (sha256 `{env['dataset_sha256'][:12]}…`) |",
        f"| Документы базы знаний | {', '.join(f'`{n}`' for n in env['documents'])} |",
        f"| Golden cases | sha256 `{env['golden_sha256'][:12]}…` |",
        f"| Коммит | `{(env['git_commit'] or 'неизвестен')[:10]}`{' (есть несохранённые изменения)' if env['git_dirty'] else ''} |",
        f"| Допуск float | rel {env['float_tolerance']['rel']}, abs {env['float_tolerance']['abs']} |",
        "",
        "## Итоговые метрики",
        "",
        "| Метрика | Значение | Что измеряет |", "|---|---|---|",
    ]
    for key, title, meaning in METRIC_ROWS:
        reason = llm_note if key.startswith(("plan_", "chart", "invalid", "clarification", "false", "numeric_accuracy_system")) else "нет подходящих случаев"
        lines.append(f"| {title} | {_pct(overall[key], reason or 'не вычислялась')} | {meaning} |")

    latency, tokens = overall["plan_latency_ms"], overall["llm_tokens"]
    lines += [
        "", "## Задержка и токены", "",
        "| Показатель | Среднее | Медиана | p95 | Максимум |", "|---|---|---|---|---|",
        f"| Построение плана (RAG + LLM + проверка), мс | {_num(latency['mean'])} | {_num(latency['median'])} | {_num(latency['p95'])} | {_num(latency['max'])} |",
        f"| Поиск по базе знаний (Top-3), мс | {_num(overall['retrieval_latency_ms']['mean'])} | {_num(overall['retrieval_latency_ms']['median'])} | {_num(overall['retrieval_latency_ms']['p95'])} | {_num(overall['retrieval_latency_ms']['max'])} |",
        f"| MCP calculate_metrics, мс | {_num(overall['mcp_latency_ms']['mean'])} | {_num(overall['mcp_latency_ms']['median'])} | {_num(overall['mcp_latency_ms']['p95'])} | {_num(overall['mcp_latency_ms']['max'])} |",
        "",
        f"Токены LLM: вызовов {tokens['calls']}, промпт {tokens['input']}, ответ {tokens['output']}, всего {tokens['total']}"
        + (f" (в среднем {tokens['mean_total_per_case']} на случай)." if tokens["mean_total_per_case"] is not None else ".")
        + f" Токены эмбеддингов (запросы поиска и проверки): {overall['embedding_tokens']}." + (" Запросы к LLM не выполнялись." if not env["llm_available"] else ""),
        "",
        "## По группам", "",
        "| Группа | Случаев | Hit@3 | Plan Accuracy | Numeric Accuracy | Отказ / уточнение |", "|---|---|---|---|---|---|",
    ]
    for group, block in results["summary"]["by_group"].items():
        behavior = block["invalid_rejection_rate"] if block["invalid_rejection_rate"]["total"] else block["clarification_rate"]
        lines.append(
            f"| {GROUP_TITLES.get(group, group)} | {block['cases']} | {_pct(block['retrieval_hit_at_3'], '—')} | {_pct(block['plan_accuracy'], '—')} | "
            f"{_pct(block['numeric_accuracy'], '—')} | {_pct(behavior, '—')} |"
        )

    failures = [c for c in results["cases"] if _case_status(c) != "ок"]
    lines += ["", "## Что не совпало", ""]
    if not failures:
        lines.append("Расхождений с golden cases нет.")
    for case in failures:
        plan = case.get("plan") or {}
        lines.append(f"- **{case['id']}** «{case['user_query']}» — {_case_status(case)}")
        if case["error"]:
            lines.append(f"  - ошибка: {case['error']}")
        for reason in plan.get("reasons", []):
            lines.append(f"  - {reason}")
        predicted = plan.get("predicted") or {}
        if plan.get("computed") and not plan.get("behavior_ok", True):
            lines.append(f"  - поведение: система — {predicted.get('behavior')}, ожидалось — {case['expected_behavior']}"
                         + (f"; не выполнено: {'; '.join(predicted.get('unsupported', []))}" if predicted.get("unsupported") else ""))
        for key, label in (("numeric_golden_plan", "числа MCP (эталонный план)"), ("numeric_system_plan", "числа MCP (план системы)")):
            if key in case and not case[key].get("ok"):
                lines.append(f"  - {label}: {case[key].get('detail')}")

    lines += ["", "## Все случаи", "", "| ID | Запрос | Ожидание | Итог | Hit@3 | Числа | План, мс | Токены |", "|---|---|---|---|---|---|---|---|"]
    for case in results["cases"]:
        hit = (case.get("retrieval") or {}).get("hit_at_3")
        numeric = (case.get("numeric_golden_plan") or {}).get("ok")
        lines.append(
            f"| {case['id']} | {case['user_query']} | {case['expected_behavior']} | {_case_status(case)} | "
            f"{'—' if hit is None else ('да' if hit else 'нет')} | {'—' if numeric is None else ('да' if numeric else 'нет')} | "
            f"{_num(case.get('plan_latency_ms'))} | {_num((case.get('tokens') or {}).get('total'))} |"
        )

    lines += [
        "", "## История прогонов", "",
        *(_history_rows(history) if history else ["Других сохранённых прогонов нет."]),
        "", "Каждый прогон сохраняется в `evaluation/results/history/`; таблица строится из этих файлов.",
        "", "## Как читать и что не проверяется", "",
        "- **Набор не является независимой отложенной выборкой.** Промпт и правила планировщика дорабатывались во время разработки на запросах, "
        "близких к этим (в том числе примеры из постановки задачи и провалы первого прогона), поэтому высокие значения завышают качество на новых запросах.",
        "- Golden cases и данные полностью синтетические (вымышленная сеть SalesDemo KZ); эталонные числа посчитаны обычным Python-кодом по строкам генератора, а не MCP и не LLM.",
        "- Numeric Accuracy проверяет расчёты MCP при эталонном плане (независимо от LLM); «план системы» — сквозная проверка: числа по тому плану, который построила модель.",
        "- Ответы LLM недетерминированы даже при temperature=0: повторный запуск может дать другие значения. Одиночный запуск на 30 случаях — не статистическая оценка.",
        "- Каждый случай выполняется один раз; задержка включает сеть OpenAI. Hit@3 зависит от эмбеддингов: с офлайн-режимом (лексический поиск) значения ниже и не отражают семантический поиск.",
        "- Оценивается построение плана и расчёты; качество формулировок выводов и чата проверяют автотесты (`pytest`), а не эти метрики.",
        "",
    ]
    if ab:
        lines += render_ab(ab, ab_sha256, {c["id"]: c["user_query"] for c in results["cases"]}) + [""]
    if hp:
        lines += render_hp(hp, hp_sha256, {c["id"]: c["user_query"] for c in results["cases"]}, previous_hp) + [""]
    return "\n".join(lines)


def write_outputs(
    results: dict[str, Any], results_path: Path = RESULTS_PATH, report_path: Path = REPORT_PATH, history_dir: Path = HISTORY_DIR
) -> tuple[Path, Path]:
    """Сохраняет latest.json (и копию в history/) и формирует EVALS.md из него (по хешу записанного файла)."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, ensure_ascii=False, indent=1) + "\n"
    results_path.write_text(text, encoding="utf-8")
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = results["generated_at"].replace("+00:00", "Z").replace("-", "").replace(":", "")
    (history_dir / f"{stamp}.json").write_text(text, encoding="utf-8")
    return results_path, write_report_from_file(results_path, report_path, history_dir)


def write_report_from_file(
    results_path: Path = RESULTS_PATH, report_path: Path = REPORT_PATH, history_dir: Path = HISTORY_DIR, ab_path: Path = AB_PATH, hp_path: Path = HP_PATH
) -> Path:
    """EVALS.md из существующего latest.json. Без файла результатов отчёт не создаётся."""
    if not results_path.exists():
        raise FileNotFoundError(f"Нет файла результатов {results_path}: сначала выполните реальный запуск (python -m datastory.evaluation).")
    raw = results_path.read_bytes()
    history = load_history(history_dir)
    ab_raw = ab_path.read_bytes() if ab_path.exists() else None
    ab = json.loads(ab_raw) if ab_raw else None
    hp_raw = hp_path.read_bytes() if hp_path.exists() else None
    hp = json.loads(hp_raw) if hp_raw else None
    report_path.write_text(
        render_report(
            json.loads(raw), hashlib.sha256(raw).hexdigest(), history, ab, hashlib.sha256(ab_raw).hexdigest() if ab_raw else "",
            hp, hashlib.sha256(hp_raw).hexdigest() if hp_raw else "", load_previous_hp(history_dir, hp),
        ),
        encoding="utf-8",
    )
    return report_path


def write_ab_outputs(
    ab: dict[str, Any], ab_path: Path = AB_PATH, report_path: Path = REPORT_PATH, results_path: Path = RESULTS_PATH, history_dir: Path = HISTORY_DIR
) -> tuple[Path, Path]:
    """ab_test.json и EVALS.md с секцией A/B. Без latest.json (нет полного evaluation) создаётся EVALS.md только с A/B секцией."""
    ab_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(ab, ensure_ascii=False, indent=1) + "\n"
    ab_path.write_text(text, encoding="utf-8")
    if results_path.exists():
        return ab_path, write_report_from_file(results_path, report_path, history_dir, ab_path)
    sha = hashlib.sha256(ab_path.read_bytes()).hexdigest()  # по байтам записанного файла (в Windows перевод строки меняется при записи)
    report_path.write_text("\n".join(["# EVALS — результаты evaluation DataStory AI", *render_ab(ab, sha), ""]), encoding="utf-8")
    return ab_path, report_path


def write_hp_outputs(
    hp: dict[str, Any], hp_path: Path = HP_PATH, report_path: Path = REPORT_PATH, results_path: Path = RESULTS_PATH, history_dir: Path = HISTORY_DIR, ab_path: Path = AB_PATH
) -> tuple[Path, Path]:
    """hyperparameters.json и EVALS.md с разделом о гиперпараметрах. Без latest.json создаётся EVALS.md только с разделами экспериментов."""
    hp_path.parent.mkdir(parents=True, exist_ok=True)
    archive_previous_hp(hp_path, hp, history_dir)
    hp_path.write_text(json.dumps(hp, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if results_path.exists():
        return hp_path, write_report_from_file(results_path, report_path, history_dir, ab_path, hp_path)
    lines = ["# EVALS — результаты evaluation DataStory AI"]
    if ab_path.exists():
        lines += render_ab(json.loads(ab_path.read_bytes()), hashlib.sha256(ab_path.read_bytes()).hexdigest())
    lines += render_hp(hp, hashlib.sha256(hp_path.read_bytes()).hexdigest()) + [""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return hp_path, report_path


def _hp_stamp(generated_at: str) -> str:
    return generated_at.replace("+00:00", "Z").replace("-", "").replace(":", "")


def archive_previous_hp(hp_path: Path, new: dict[str, Any], history_dir: Path = HISTORY_DIR) -> None:
    """Прошлый результат эксперимента не затирается: перед записью нового он копируется в history/hp-<дата>.json."""
    if not hp_path.exists():
        return
    old = json.loads(hp_path.read_text(encoding="utf-8"))
    if old.get("generated_at") == new.get("generated_at"):
        return
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"hp-{_hp_stamp(old['generated_at'])}.json").write_bytes(hp_path.read_bytes())


def load_previous_hp(history_dir: Path, current: dict[str, Any] | None) -> list[dict[str, Any]]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(history_dir.glob("hp-*.json"))] if history_dir.exists() else []
    return [r for r in runs if not current or r["generated_at"] != current["generated_at"]]


def load_history(history_dir: Path = HISTORY_DIR) -> list[dict[str, Any]]:
    """Сохранённые прогоны по возрастанию даты (файлы с полным прогоном)."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(history_dir.glob("*.json"))] if history_dir.exists() else []
    return [r for r in runs if r.get("scope", {}).get("complete")]
