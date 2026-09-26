"""Промпты этапов воркфлоу. Таблица целиком в LLM не передаётся: только сводка датасета (DatasetSummary) без персональных данных."""
from __future__ import annotations

from datastory.analytics.models import DatasetSummary
from datastory.skills import Skill
from datastory.workflow.catalog import ANALYSES
from datastory.workflow.context import BusinessContext
from datastory.workflow.methodology import methodology_prompt
from datastory.workflow.models import MAX_CHARTS, AnalysisCandidate

INTENT_SYSTEM = (
    "Ты определяешь, какие из поддерживаемых анализов запросил пользователь. Поддерживаются только:\n"
    + "\n".join(f"- {a.id}: {a.label}" for a in ANALYSES)
    + "\nВ analyses верни идентификаторы запрошенных анализов из этого списка. Всё, что запрошено сверх списка "
    "(прогноз, причины, другие показатели), перечисли в unsupported. Если запрос общий («проанализируй данные»), "
    "верни все анализы."
)

PLAN_TASK = (
    f"Ты составляешь план анализа данных: не более {MAX_CHARTS} графиков, по одному на каждый выбранный анализ. "
    "Для каждого графика укажи: анализ (analysis), название на русском, тип графика, колонку оси X и одно предложение обоснования. "
    "Колонку оси X выбирай только из перечисленных кандидатов; анализы вне списка выбранных не добавляй."
)

INSIGHT_TASK = (
    "Ты пишешь аналитический вывод по одному графику и возвращаешь структурный ответ: название (title), краткий вывод (summary, "
    "1–3 предложения), важность (severity), id использованных числовых доказательств (evidence_ids), ограничение интерпретации "
    "(limitation). Если в выводе использован фрагмент документа, укажи его chunk_id и дословную цитату."
)


def plan_system(skill: Skill) -> str:
    """Системный промпт этапа build_analysis_plan: формат ответа + разделы Skill для этого этапа."""
    return "\n\n".join([PLAN_TASK, methodology_prompt(skill, "plan")])


def insight_system(skill: Skill) -> str:
    """Системный промпт этапа generate_insights: формат ответа + разделы Skill для этого этапа."""
    return "\n\n".join([INSIGHT_TASK, methodology_prompt(skill, "insights")])


def summary_text(summary: DatasetSummary) -> str:
    lines = [f"Датасет: {summary.row_count} строк, {summary.column_count} колонок."]
    for column in summary.columns:
        extra = ""
        if column.statistics and not column.is_sensitive:
            extra = f"; {column.statistics}"
        lines.append(f"- {column.name}: {column.kind}, уникальных значений {column.unique_count}{extra}")
    return "\n".join(lines)


def intent_prompt(request: str, summary: DatasetSummary) -> str:
    return f"Запрос пользователя: «{request}»\n\n{summary_text(summary)}"


def plan_prompt(
    summary: DatasetSummary, candidates: list[AnalysisCandidate], selected: list[str], context: BusinessContext, request: str
) -> str:
    rows = []
    for c in candidates:
        if c.id not in selected:
            continue
        if c.available:
            rows.append(f"- {c.id} ({c.label}): доступен; кандидаты для оси X: {', '.join(c.dimension_candidates)}")
        else:
            rows.append(f"- {c.id} ({c.label}): НЕДОСТУПЕН — {'; '.join(c.reasons)} (в план не включать)")
    documented = [m for m in context.definitions] or ["нет"]
    parts = [
        summary_text(summary),
        "Выбранные анализы:\n" + "\n".join(rows),
        f"Показатели с бизнес-определением в документации: {', '.join(documented)}.",
    ]
    if request.strip():
        parts.insert(0, f"Запрос пользователя: «{request.strip()}»")
    return "\n\n".join(parts)
