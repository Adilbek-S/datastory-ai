"""Связь Skill datastory-analysis с этапами воркфлоу.

Текст методики хранится только в .claude/skills/datastory-analysis/SKILL.md. Здесь задано, какие разделы Skill
получает каждый LLM-этап, и где в коде проверяется каждое правило: набор идентификаторов в этой таблице обязан
совпадать с набором в SKILL.md (это проверяет тест), поэтому методика и код не расходятся.
"""
from __future__ import annotations

from datastory.skills import Skill
from datastory.workflow.models import MethodologyInfo

SKILL_NAME = "datastory-analysis"

# этап -> регулярные выражения для заголовков разделов Skill, которые попадают в системный промпт этапа
STAGE_SECTIONS: dict[str, tuple[str, ...]] = {
    "plan": (r"^Шаг [1-4]\.", r"^Правила выбора визуализации", r"^Правила анализа"),
    "insights": (r"^Шаг [5-8]\.", r"^Правила анализа"),
    "chat": (r"^Шаг [35678]\.", r"^Правила анализа"),  # ответ на вопрос: определения, расчёты, изменения, выводы, проверка
}
STAGE_TITLES = {"plan": "план анализа", "insights": "аналитические выводы", "chat": "ответы в чате"}

HEADER = f"Методика анализа (Skill «{SKILL_NAME}»). Следуй ей строго: нарушения отклоняются проверкой и выводы переписываются."

# правило Skill -> где оно проверяется в коде
ENFORCEMENT: dict[str, str] = {
    "V1": "workflow.catalog.fit_chart_type: для динамики допустим только line",
    "V2": "workflow.catalog.fit_chart_type: для категорий допустимы bar и pie",
    "V3": "workflow.catalog.fit_chart_type: pie только до PIE_MAX_CATEGORIES категорий, иначе bar",
    "V4": "workflow.catalog.fit_chart_type и MCP create_chart_spec: pie для временных рядов и отношений не строится",
    "V5": "структура плана: PlannedChart.metric — один показатель, create_chart_spec принимает один MetricResult",
    "V6": "workflow.models.AnalysisPlan: charts не более MAX_CHARTS",
    "A1": "insights.verifier: numbers-grounded, evidence-required, unknown-evidence",
    "A2": "workflow.planning.apply_decision: без определения в документации нужна подтверждённая пользователем формула",
    "A3": "insights.verifier: no-causal-claim; insights.generator: обязательное ограничение CAUSE_UNKNOWN",
    "A4": "insights.verifier: trend-needs-comparison, trend-direction-mismatch",
    "A5": "insights.facts: разности и доли считает Python, числа считает MCP; verifier отклоняет чужие числа",
    "A6": "insights.verifier: numbers-grounded, period-not-in-data; в LLM уходит только сводка без персональных данных",
    "A7": "models.Insight.evidence_type и rag.evidence.validate_evidence: context-not-in-source",
}


def methodology_prompt(skill: Skill, stage: str) -> str:
    """Разделы Skill для этапа (plan | insights) с заголовком; единственный источник методики в промптах."""
    return f"{HEADER}\n\n{skill.select(*STAGE_SECTIONS[stage])}"


def methodology_info(skill: Skill, stages: list[str]) -> MethodologyInfo:
    return MethodologyInfo(skill=skill.name, description=skill.description, stages=stages, rules=skill.rule_ids)
