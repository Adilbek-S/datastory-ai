"""Отчёт по результатам анализа: ключевые выводы, сводка набора данных и Markdown-файл для скачивания.

Отчёт строится только из уже готовых и проверенных данных (AnalysisResult, DatasetProfile): новых расчётов и обращений
к LLM здесь нет, поэтому числа в отчёте совпадают с числами на дашборде.
"""
from __future__ import annotations

import re
from datetime import datetime

from datastory.analytics.formatting import format_metric_value
from datastory.insights.generator import CAUSE_UNKNOWN
from datastory.models import ChartSpec, DatasetProfile, Insight, Severity
from datastory.workflow.models import AUTO_GOAL, AnalysisResult

DISCLAIMER = (
    "Аналитические выводы сформированы автоматически на основании загруженных данных. "
    "Расчёты выполняются программно; интерпретация формируется AI."
)
CHART_LABELS = {"line": "линейный график", "bar": "столбчатый график", "pie": "круговая диаграмма"}
SEVERITY_ORDER = {"warning": 0, "info": 1, "positive": 2}
MAX_FINDINGS = 5
MAX_TABLE_ROWS = 30
KIND_LABELS = {"numeric": "число", "categorical": "категория", "datetime": "дата/время", "boolean": "да/нет", "text": "текст"}
_ABBREVIATION = "п.п."


# --------------------------------------------------------------------------- сводка набора данных
def _short_date(value: str) -> str:
    """«2026-01-01 00:00:00» → «2026-01», если в данных только месяцы; иначе дата без времени."""
    match = re.match(r"^(\d{4}-\d{2})-01(?: 00:00:00)?$", value)
    return match.group(1) if match else value.split(" ")[0]


def dataset_period(profile: DatasetProfile) -> str | None:
    """Период данных по первой временной колонке: «2026-01 — 2026-06»."""
    for column in profile.columns:
        if column.kind.value == "datetime" and column.date_min and column.date_max:
            low, high = _short_date(column.date_min), _short_date(column.date_max)
            return low if low == high else f"{low} — {high}"
    return None


def quality_summary(profile: DatasetProfile) -> str:
    """Одна строка о качестве данных для карточки и отчёта."""
    problems = [i for i in profile.quality_issues if i.severity is not Severity.INFO]
    if not profile.quality_issues:
        return "Без замечаний"
    facts = [f"замечаний: {len(problems)}"] if problems else ["критичных замечаний нет"]
    if profile.missing_cell_count:
        facts.append(f"пропущено ячеек: {profile.missing_cell_count}")
    if profile.duplicate_row_count:
        facts.append(f"дубликатов строк: {profile.duplicate_row_count}")
    return ", ".join(facts)


# --------------------------------------------------------------------------- ключевые выводы
def _leading_sentences(text: str, count: int) -> str:
    """Первые предложения вывода; сокращение «п.п.» не считается концом предложения."""
    protected = text.replace(_ABBREVIATION, "п․п․")
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return " ".join(parts[:count]).replace("․", ".")


def key_findings(result: AnalysisResult) -> list[str]:
    """Главное из выводов по графикам: сначала предупреждения; плюс честные оговорки о причинах и документации."""
    ranked = sorted(result.insights, key=lambda i: SEVERITY_ORDER.get(i.severity, 1))
    # предупреждение (существенное падение) приводится целиком, остальные выводы — двумя первыми предложениями
    findings = [
        f"**{i.title}.** {i.text if i.severity == 'warning' else _leading_sentences(i.text, 2)}" for i in ranked[:MAX_FINDINGS]
    ]
    if any(CAUSE_UNKNOWN in (i.limitation or "") for i in result.insights):
        findings.append("Причины изменений по данным не установлены: расчёты фиксируют сам факт изменения, но не объясняют его.")
    if result.plan and result.plan.documentation_gaps:
        gaps = ", ".join(g.split(":")[0] for g in result.plan.documentation_gaps)
        findings.append(f"Для показателей ({gaps}) нет определений в документации: использованы формулы, подтверждённые пользователем.")
    return findings


def based_on(insight: Insight) -> str:
    """Подпись «На основе: …» под графиком: откуда взяты числа и контекст."""
    parts = [insight.data_source] if insight.data_source else []
    if insight.context_source:
        parts.append(f"бизнес-контекст: {insight.context_source}")
    return "; ".join(parts) or "результаты расчётов MCP"


# --------------------------------------------------------------------------- Markdown
def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _chart_table(spec: ChartSpec) -> list[str]:
    if not spec.data:
        return []
    keys = list(spec.data[0])
    rows = [
        [format_metric_value(row.get(k), spec.unit or "") if k == spec.y and isinstance(row.get(k), (int, float)) else row.get(k, "") for k in keys]
        for row in spec.data[:MAX_TABLE_ROWS]
    ]
    lines = _table([spec.x_title if k == spec.x else (spec.y_title or k) if k == spec.y else k for k in keys], rows)
    if len(spec.data) > MAX_TABLE_ROWS:
        lines.append(f"\n_Показаны первые {MAX_TABLE_ROWS} из {len(spec.data)} строк._")
    return lines


def _quote(text: str) -> str:
    return "> " + " ".join(text.split())


def build_markdown_report(result: AnalysisResult, profile: DatasetProfile | None = None, generated_at: datetime | None = None) -> str:
    """Markdown-отчёт: описание набора, KPI, анализы, выводы, использованные источники базы знаний."""
    summary = result.summary
    when = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    goal = "автоматический анализ" if result.goal in ("", AUTO_GOAL) else result.goal
    out = [f"# Отчёт DataStory AI: {summary.filename}", "", f"_Сформирован {when}._", "", f"> {DISCLAIMER}", ""]

    out += ["## Описание набора данных", ""]
    out += [f"- **Файл:** {summary.filename}" + (f", лист «{summary.sheet_name}»" if summary.sheet_name else "")]
    out += [f"- **Строк:** {summary.row_count}", f"- **Колонок:** {summary.column_count}"]
    period = dataset_period(profile) if profile else None
    out += [f"- **Период данных:** {period or 'не определён'}"]
    out += [f"- **Цель анализа:** {goal}"]
    if profile is not None:
        out += [f"- **Качество данных:** {quality_summary(profile)}"]
        if profile.description:
            out += [f"- **Описание:** {profile.description}"]
        for issue in profile.quality_issues:
            out += [f"  - {issue.message}"]
    out += [""]
    out += _table(
        ["Колонка", "Тип", "Пропусков"],
        [[c.name, KIND_LABELS.get(c.kind, c.kind), f"{c.missing_count}"] for c in summary.columns],
    )
    out += ["", "## Ключевые показатели (KPI)", ""]
    out += _table(["Показатель", "Значение", "Расчёт"], [[k.label, k.value, k.hint] for k in result.kpis]) if result.kpis else ["Показатели не рассчитаны."]

    out += ["", "## Анализы", ""]
    insights = {i.chart_id: i for i in result.insights}
    steps = {c.chart_id: c for c in (result.plan.charts if result.plan else [])}
    for number, (metric, spec) in enumerate(zip(result.metrics, result.charts), 1):
        step = next((s for s in steps.values() if s.title == spec.title), None)
        insight = insights.get(step.chart_id) if step else None
        out += [f"### {number}. {spec.title}", ""]
        if step and step.rationale:
            out += [f"- **Зачем:** {step.rationale}"]
        out += [f"- **Показатель:** {metric.label} = {metric.formula} ({metric.unit})"]
        out += [f"- **Группировка:** {', '.join(metric.group_by)}", f"- **Визуализация:** {CHART_LABELS.get(spec.kind, spec.kind)}", ""]
        out += _chart_table(spec)
        if insight:
            out += ["", f"**{insight.title}.** {insight.text}", ""]
            if insight.evidence:
                out += ["Числовые доказательства:", ""]
                out += _table(["Число", "Значение", "Как получено"], [[e.label, e.formatted, e.computation] for e in insight.evidence])
            out += ["", f"_На основе: {based_on(insight)}_"]
            if insight.limitation:
                out += ["", f"⚠ **Ограничение интерпретации:** {insight.limitation}"]
        out += [""]

    out += ["## Ключевые выводы", ""]
    findings = key_findings(result)
    out += [f"- {f}" for f in findings] if findings else ["Выводов нет: ни один график не построен."]

    out += ["", "## Использованные источники базы знаний", ""]
    if result.context_sources:
        for source in result.context_sources:
            kind = f"определение показателя «{source.metric_label}»" if source.kind == "definition" else "сопроводительный контекст"
            out += [f"- **{source.citation}** — {kind}", f"  {_quote(source.text)}"]
            if source.used_in:
                out += [f"  Использовано в выводах: {', '.join(source.used_in)}."]
    else:
        out += ["Документы базы знаний не использованы: подходящих определений и контекста не найдено."]

    notes = [w for w in result.warnings]
    if notes:
        out += ["", "## Примечания", ""] + [f"- {w}" for w in notes]
    out += ["", "---", "_Отчёт сформирован DataStory AI. Числа рассчитаны MCP-сервером аналитики; интерпретация подготовлена AI и проверена на соответствие расчётам._", ""]
    return "\n".join(out)
