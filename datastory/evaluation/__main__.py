"""Команда запуска всех evals.

    python -m datastory.evaluation                    # полный прогон 30 golden cases → latest.json и EVALS.md
    python -m datastory.evaluation --groups filters   # часть случаев (результат в partial.json, EVALS.md не меняется)
    python -m datastory.evaluation --cases mi-01,iv-01
    python -m datastory.evaluation --no-llm           # без LLM (метрики плана помечаются «не вычислялась»)
    python -m datastory.evaluation --report-only      # заново собрать EVALS.md из существующего latest.json
    python -m datastory.evaluation --ab               # A/B эксперимент (без RAG / с Top-5 RAG) → ab_test.json и секция в EVALS.md
    python -m datastory.evaluation --ab --repeats 5   # число повторов каждого случая в каждой конфигурации (по умолчанию 3)
    python -m datastory.evaluation --hyperparams      # temperature 0 против 0.4 на 10 случаях (5 повторов) → hyperparameters.json и раздел в EVALS.md

Нужны OPENAI_API_KEY (LLM и эмбеддинги) и MCP-сервер (запускается автоматически). Без ключа шаги LLM пропускаются и
честно помечаются в результатах: значения не подставляются.
"""
from __future__ import annotations

import argparse
import json
import sys

from datastory.evaluation.golden import GROUPS, RESULTS_PATH
from datastory.evaluation.ab_test import DEFAULT_REPEATS, run_ab
from datastory.evaluation.hyperparams import DEFAULT_REPEATS as HP_REPEATS
from datastory.evaluation.hyperparams import run_hyperparams
from datastory.evaluation.report import REPORT_PATH, write_ab_outputs, write_hp_outputs, write_outputs, write_report_from_file
from datastory.evaluation.runner import run_evals

PARTIAL_PATH = RESULTS_PATH.with_name("partial.json")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # консоль Windows (cp1251) не выводит «→», «−», «п.п.»
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m datastory.evaluation", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", help=f"группы через запятую: {', '.join(GROUPS)}")
    parser.add_argument("--cases", help="идентификаторы случаев через запятую")
    parser.add_argument("--limit", type=int, help="первые N случаев")
    parser.add_argument("--no-llm", action="store_true", help="не использовать LLM")
    parser.add_argument("--ab", action="store_true", help="A/B эксперимент: RAG в промпте планировщика против без RAG (нужен OPENAI_API_KEY)")
    parser.add_argument("--hyperparams", action="store_true", help="выбор temperature: 0 против 0.4 на 10 случаях (нужен OPENAI_API_KEY)")
    parser.add_argument("--repeats", type=int, help="повторов каждого случая в каждой конфигурации (A/B — 3, hyperparams — 5 по умолчанию)")
    parser.add_argument("--report-only", action="store_true", help="собрать EVALS.md из существующего latest.json")
    args = parser.parse_args(argv)

    if args.report_only:
        try:
            print(f"Отчёт: {write_report_from_file()}")
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2
        return 0

    if args.hyperparams:
        try:
            hp = run_hyperparams(repeats=max(1, args.repeats or HP_REPEATS))
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 2
        hp_path, report_path = write_hp_outputs(hp)
        print(f"\nРезультаты: {hp_path}\nОтчёт: {report_path}\n")
        final = hp["final"]
        print(f"Выбрано: {final['llm_model']}, temperature {final['temperature']}, max output tokens {final['max_output_tokens']}")
        print("\n".join(f"- {r}" for r in hp["decision"]["reasons"]))
        return 0

    if args.ab:
        try:
            ab = run_ab(repeats=max(1, args.repeats or DEFAULT_REPEATS))
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 2
        ab_path, report_path = write_ab_outputs(ab)
        print(f"\nРезультаты: {ab_path}\nОтчёт: {report_path}\n")
        print("\n".join(ab["conclusion"]["statements"] + ab["conclusion"]["caveats"]))
        return 0

    try:
        results = run_evals(
            case_ids=args.cases.split(",") if args.cases else None, groups=args.groups.split(",") if args.groups else None,
            limit=args.limit, use_llm=not args.no_llm,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    summary = results["summary"]["overall"]
    if results["scope"]["complete"]:
        json_path, report_path = write_outputs(results)
        print(f"\nРезультаты: {json_path}\nОтчёт: {report_path}")
    else:
        PARTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PARTIAL_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\nЧастичный прогон: {PARTIAL_PATH}. latest.json и {REPORT_PATH.name} не изменены (обновляются только полным прогоном).")
    for key in ("retrieval_hit_at_3", "plan_accuracy", "numeric_accuracy", "invalid_rejection_rate"):
        block = summary[key]
        print(f"  {key}: " + (f"{block['rate'] * 100:.1f}% ({block['hits']}/{block['total']})" if block["total"] else "не вычислялась"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
