"""Запуск evaluation: настоящие RAG, LLM и MCP на демо-наборе sales_2026; ничего не подменяется.

Для каждого golden case:
1. Retrieval — поиск по базе знаний (Top-3) и проверка нужного документа;
2. План — workflow до подтверждения пользователем (LLM + RAG + проверка по данным): показатель, колонки, группировка;
3. Числа — MCP calculate_metrics по эталонному плану (правильность расчёта) и по плану системы (сквозная правильность),
   сравнение с эталоном, рассчитанным обычным Python-кодом;
4. Поведение — ответ, отказ или уточнение (invalid rejection rate).
Если нет ключа OpenAI, шаги LLM не выполняются и в результатах помечены «не вычислялось»: значения не подставляются.
"""
from __future__ import annotations

import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datastory.config import PROJECT_ROOT, Settings, get_settings
from datastory.errors import LLMUnavailableError
from datastory.evaluation import scoring
from datastory.evaluation.golden import GOLDEN_PATH, GROUPS, GoldenCase, GoldenSet, file_sha256, load_golden
from datastory.file_processing.loader import read_table
from datastory.llm.client import get_llm
from datastory.llm.usage import UsageRecorder
from datastory.mcp_client.analytics_client import AnalyticsMcpClient
from datastory.profiler.profiler import build_profile
from datastory.rag.embeddings import get_embedding_provider
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.storage.store import DatasetStore
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.runner import AnalysisRunner

SCHEMA_VERSION = 1
TOP_K = 3
SALES_SHEET = "sales"


@dataclass
class Environment:
    runner: AnalysisRunner
    client: AnalyticsMcpClient
    kb: KnowledgeBase
    usage: UsageRecorder
    dataset_id: str
    llm_model: str | None  # None — шаги LLM не выполняются
    llm_reason: str  # почему LLM не используется (пусто, если используется)
    settings: Settings
    dataset_path: Path
    documents: list[Path]
    llm: Any = None  # StructuredLLM или None

    def close(self) -> None:
        self.client.close()


def build_environment(golden: GoldenSet, workdir: Path, settings: Settings | None = None, use_llm: bool = True, max_tokens: int | None = None) -> Environment:
    """Сохраняет sales_2026 в рабочее хранилище, индексирует документы, запускает MCP-сервер и создаёт runner."""
    settings = settings or get_settings()
    dataset_path = PROJECT_ROOT / golden.dataset
    raw = read_table(dataset_path.read_bytes(), dataset_path.name, SALES_SHEET)
    profile, typed = build_profile(raw, dataset_path.name, SALES_SHEET)
    DatasetStore(workdir / "workspace").save(typed, profile)

    kb = KnowledgeBase(get_embedding_provider(settings), workdir / "chroma")
    documents = [PROJECT_ROOT / name for name in golden.documents]
    for path in documents:
        kb.index_document(path.read_bytes(), path.name)

    usage = UsageRecorder()
    llm, model, reason = None, None, "LLM отключена флагом --no-llm"
    if use_llm:
        try:
            llm, model, reason = get_llm(settings, usage=usage, max_tokens=max_tokens), settings.openai_model, ""
        except LLMUnavailableError:
            reason = "не задан OPENAI_API_KEY"
    client = AnalyticsMcpClient.from_settings(Settings(workspace_dir=workdir / "workspace", _env_file=None))
    client.profile_dataset(profile.dataset_id)  # запуск MCP-сервера до замеров: задержка не должна включать холодный старт
    runner = AnalysisRunner(WorkflowDeps.of(client, llm, kb), max_threads=max(50, len(golden.cases) + 5))
    return Environment(runner, client, kb, usage, profile.dataset_id, model, reason, settings, dataset_path, documents, llm)


# --------------------------------------------------------------------------- один случай
def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def retrieve(env: Environment, case: GoldenCase) -> dict[str, Any]:
    started = time.perf_counter()
    result = env.kb.search_business_context(case.user_query, TOP_K)
    hits = [{"document": h.source.filename, "section": h.section, "score": h.score, "relevant": h.relevant} for h in result.hits]
    out: dict[str, Any] = {"hits": hits, "latency_ms": _ms(started), "provider": result.provider}
    if case.expected_rag_document:
        out["hit_at_3"] = scoring.retrieval_hit(hits, case.expected_rag_document, TOP_K)
        if case.expected_rag_section:
            out["section_hit_at_3"] = scoring.section_hit(hits, case.expected_rag_document, case.expected_rag_section, TOP_K)
    return out


def predict(snapshot) -> dict[str, Any]:
    """Что предложила система: поведение (ответ, уточнение, отказ) и первый шаг плана."""
    plan = snapshot.plan or (snapshot.request.plan if snapshot.request else None)
    steps = plan.charts if plan else []
    confident = [s for s in steps if s.mapping.status != "ambiguous"]
    ambiguous = [s for s in steps if s.mapping.status == "ambiguous"]
    behavior = "answer" if confident else "clarify" if ambiguous else "reject"
    first = (confident or ambiguous or [None])[0]
    prediction: dict[str, Any] = {
        "behavior": behavior, "phase": snapshot.phase, "steps": len(steps), "planner": plan.planner if plan else None,
        "warnings": list(snapshot.warnings), "unsupported": list(plan.unsupported) if plan else [], "excluded": [f"{e.label}: {e.reason}" for e in (plan.excluded if plan else [])],
        "notes": list(plan.notes) if plan else [], "metric": None, "group_by": [], "filters": [], "chart_type": None, "columns": [],
    }
    if first is not None:
        filters = [{"column": f.column, "op": f.op.value, "value": f.value} for f in first.filters]
        prediction.update(
            metric=first.metric, group_by=list(first.grouping), filters=filters, chart_type=first.chart_type,
            columns=scoring.plan_columns(first.metric, list(first.grouping), filters),
        )
    return prediction


def execute_numeric(env: Environment, case: GoldenCase, metric: str, group_by: list[str], filters: list[dict[str, Any]]) -> dict[str, Any]:
    """MCP calculate_metrics по заданному шагу и сравнение с эталоном."""
    started = time.perf_counter()
    try:
        result = env.client.calculate_metrics(env.dataset_id, metric, group_by, filters or None)
    except Exception as exc:  # noqa: BLE001 — отказ MCP или ошибка запроса фиксируются в результате, а не роняют прогон
        return {"ok": False, "detail": f"MCP: {getattr(exc, 'message', None) or exc}", "latency_ms": _ms(started)}
    latency = _ms(started)
    verdict = scoring.score_numeric(
        case.expected_numeric, [{"group": r.group, "value": r.value} for r in result.rows], result.overall.value
    )
    return {"ok": verdict.ok, "max_abs_error": verdict.max_abs_error, "max_rel_error": verdict.max_rel_error, "detail": verdict.detail, "latency_ms": latency}


def run_case(
    env: Environment, case: GoldenCase, *, runner: AnalysisRunner | None = None, retrieval: bool = True, numeric: bool = True
) -> dict[str, Any]:
    """runner — подменяет env.runner (A/B: разные конфигурации RAG); retrieval/numeric=False пропускают эти проверки."""
    runner = runner or env.runner
    record: dict[str, Any] = {
        "id": case.id, "group": case.group, "user_query": case.user_query, "expected_behavior": case.expected_behavior,
        "has_numeric": case.expected_numeric is not None, "llm_used": env.llm_model is not None,
        "expected": {
            "metric": case.expected_metric, "columns": case.expected_columns, "group_by": case.expected_group_by,
            "filters": [f.model_dump() for f in case.expected_filters], "chart_type": case.expected_chart_type,
            "rag_document": case.expected_rag_document, "rag_section": case.expected_rag_section,
        },
        "error": None,
    }
    tokens_before = getattr(env.kb.embedder, "tokens_used", 0)
    env.usage.reset()
    try:
        if retrieval:
            record["retrieval"] = retrieve(env, case)
        if numeric and case.expected_numeric:  # правильность расчёта MCP при эталонном плане не зависит от LLM
            spec = case.expected_numeric
            record["numeric_golden_plan"] = execute_numeric(env, case, spec.metric, spec.group_by, [f.model_dump() for f in spec.filters])
        if env.llm_model is None:
            record["plan"] = {"computed": False, "reason": f"{env.llm_reason}: LLM-планирование не запускалось"}
        else:
            env.usage.reset()
            started = time.perf_counter()
            snapshot = runner.start(env.dataset_id, case.user_query)
            record["plan_latency_ms"] = _ms(started)
            record["thread_id"] = snapshot.thread_id
            usage = env.usage.reset()
            record["tokens"] = {"llm_calls": usage.calls, "input": usage.input_tokens, "output": usage.output_tokens, "total": usage.total_tokens}
            if snapshot.phase == "failed":
                raise RuntimeError(f"workflow: {snapshot.failure.message}")
            predicted = predict(snapshot)
            verdict = scoring.score_plan(case, predicted) if case.expected_behavior == "answer" else None
            record["plan"] = {
                "computed": True, "predicted": predicted, "behavior_ok": predicted["behavior"] == case.expected_behavior,
                **({"accurate": verdict.accurate, "full_accurate": verdict.accurate and verdict.filters_ok, "metric_ok": verdict.metric_ok, "columns_ok": verdict.columns_ok, "grouping_ok": verdict.grouping_ok,
                    "filters_ok": verdict.filters_ok, "chart_ok": verdict.chart_ok, "reasons": verdict.reasons} if verdict else {}),
            }
            if not numeric:
                pass
            elif case.expected_numeric and predicted["metric"] and predicted["behavior"] == "answer":
                record["numeric_system_plan"] = execute_numeric(env, case, predicted["metric"], predicted["group_by"], predicted["filters"])
            elif case.expected_numeric:
                record["numeric_system_plan"] = {"ok": False, "detail": "система не построила шаг плана", "latency_ms": None}
    except Exception as exc:  # noqa: BLE001 — сбой одного случая не останавливает прогон: он фиксируется и учитывается как неудача
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["embedding_tokens"] = getattr(env.kb.embedder, "tokens_used", 0) - tokens_before
    return record


# --------------------------------------------------------------------------- сводка
def _ratio(records: list[dict], pick) -> dict[str, Any]:
    """{hits, total, rate} по случаям, где pick(record) не None."""
    values = [pick(r) for r in records]
    values = [v for v in values if v is not None]
    return {"hits": sum(1 for v in values if v), "total": len(values), "rate": scoring.rate(sum(1 for v in values if v), len(values))}


def _plan_flag(name: str):
    def pick(record: dict) -> bool | None:
        if record["expected_behavior"] != "answer" or not record["llm_used"]:
            return None
        if record["error"]:
            return False  # сбой случая — промах, а не пропуск
        return bool((record.get("plan") or {}).get(name))

    return pick


def _numeric_flag(key: str):
    def pick(record: dict) -> bool | None:
        if key in record:
            return bool(record[key].get("ok"))
        # сбой случая — промах; без сбоя шаг не выполнялся (нет LLM), и значение не подставляется
        return False if (record["error"] and record["has_numeric"] and (key == "numeric_golden_plan" or record["llm_used"])) else None

    return pick


def _behavior_rate(behavior: str):
    def pick(record: dict) -> bool | None:
        if record["expected_behavior"] != behavior or not record["llm_used"]:
            return None
        return False if record["error"] else bool((record.get("plan") or {}).get("behavior_ok"))

    return pick


def summarize(records: list[dict]) -> dict[str, Any]:
    def block(items: list[dict]) -> dict[str, Any]:
        plan_latency = [r["plan_latency_ms"] for r in items if "plan_latency_ms" in r]
        tokens = [r["tokens"] for r in items if "tokens" in r]
        return {
            "cases": len(items),
            "retrieval_hit_at_3": _ratio(items, lambda r: (r.get("retrieval") or {}).get("hit_at_3")),
            "retrieval_section_hit_at_3": _ratio(items, lambda r: (r.get("retrieval") or {}).get("section_hit_at_3")),
            "plan_accuracy": _ratio(items, _plan_flag("accurate")),
            "plan_full_accuracy": _ratio(items, _plan_flag("full_accurate")),
            "plan_metric_accuracy": _ratio(items, _plan_flag("metric_ok")),
            "plan_columns_accuracy": _ratio(items, _plan_flag("columns_ok")),
            "plan_grouping_accuracy": _ratio(items, _plan_flag("grouping_ok")),
            "plan_filters_accuracy": _ratio(items, lambda r: _plan_flag("filters_ok")(r) if r["expected"]["filters"] else None),
            "chart_type_accuracy": _ratio(items, _plan_flag("chart_ok")),
            "numeric_accuracy": _ratio(items, _numeric_flag("numeric_golden_plan")),
            "numeric_accuracy_system_plan": _ratio(items, _numeric_flag("numeric_system_plan")),
            "invalid_rejection_rate": _ratio(items, _behavior_rate("reject")),
            "clarification_rate": _ratio(items, _behavior_rate("clarify")),
            "false_rejection_rate": _ratio(
                items, lambda r: None if r["expected_behavior"] != "answer" or not (r.get("plan") or {}).get("computed") or r["error"]
                else r["plan"]["predicted"]["behavior"] == "reject"
            ),
            "plan_latency_ms": scoring.stats(plan_latency),
            "retrieval_latency_ms": scoring.stats([r["retrieval"]["latency_ms"] for r in items if "retrieval" in r]),
            "mcp_latency_ms": scoring.stats([
                r[k]["latency_ms"] for r in items for k in ("numeric_golden_plan", "numeric_system_plan") if k in r and r[k].get("latency_ms") is not None
            ]),
            "llm_tokens": {
                "calls": sum(t["llm_calls"] for t in tokens), "input": sum(t["input"] for t in tokens),
                "output": sum(t["output"] for t in tokens), "total": sum(t["total"] for t in tokens),
                "mean_total_per_case": round(sum(t["total"] for t in tokens) / len(tokens), 1) if tokens else None,
            },
            "embedding_tokens": sum(r.get("embedding_tokens", 0) for r in items),
            "errors": sum(1 for r in items if r["error"]),
        }

    return {"overall": block(records), "by_group": {g: block([r for r in records if r["group"] == g]) for g in GROUPS if any(r["group"] == g for r in records)}}


# --------------------------------------------------------------------------- окружение и прогон
def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — не git-репозиторий или git недоступен
        return None


def describe_environment(env: Environment, golden: GoldenSet) -> dict[str, Any]:
    settings = env.settings
    return {
        "llm_model": env.llm_model, "llm_available": env.llm_model is not None, "llm_reason": env.llm_reason,
        "embedding_provider": env.kb.embedder.name, "embedding_is_semantic": env.kb.embedder.is_semantic,
        "dataset": golden.dataset, "dataset_sha256": file_sha256(env.dataset_path),
        "documents": {p.name: file_sha256(p) for p in env.documents},
        "golden_sha256": file_sha256(GOLDEN_PATH), "top_k": TOP_K, "float_tolerance": {"rel": scoring.REL_TOL, "abs": scoring.ABS_TOL},
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(), "langsmith_tracing": settings.langsmith_tracing and bool(settings.langsmith_api_key),
    }


def run_evals(case_ids: list[str] | None = None, groups: list[str] | None = None, limit: int | None = None, use_llm: bool = True, log=print) -> dict[str, Any]:
    golden = load_golden()
    selected = [c for c in golden.cases if (not case_ids or c.id in case_ids) and (not groups or c.group in groups)]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Не выбрано ни одного golden case: проверьте --cases и --groups.")
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows: Chroma держит файлы до выхода из процесса
        env = build_environment(golden, Path(tmp), use_llm=use_llm)
        try:
            environment = describe_environment(env, golden)
            log(f"Evals: {len(selected)} из {len(golden.cases)} случаев; LLM: {env.llm_model or env.llm_reason + ' — шаги LLM пропущены'}; эмбеддинги: {env.kb.embedder.name}")
            records = []
            for index, case in enumerate(selected, 1):
                record = run_case(env, case)
                records.append(record)
                plan = record.get("plan") or {}
                if record["error"]:
                    status = "ошибка"
                elif not plan.get("computed"):
                    status = "без плана"
                else:
                    status = "ok" if plan.get("accurate", plan.get("behavior_ok")) else "не совпало"
                log(f"  [{index:2d}/{len(selected)}] {case.id} {status}: {case.user_query}")
        finally:
            env.close()
    return {
        "schema_version": SCHEMA_VERSION, "generated_at": started_at.isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - started, 1),
        "scope": {"complete": len(selected) == len(golden.cases), "cases": len(selected), "total_cases": len(golden.cases)},
        "environment": environment, "summary": summarize(records), "cases": records,
    }
