"""Markdown-отчёт и ключевые выводы: строятся из готовых результатов, числа совпадают с дашбордом."""
import re
from datetime import datetime

import pandas as pd
import pytest

from datastory.analytics.formatting import format_metric_value
from datastory.file_processing.loader import read_table
from datastory.models import DataQualityIssue, Severity
from datastory.profiler.profiler import build_profile
from datastory.rag.knowledge_base import KnowledgeBase
from datastory.report.builder import (
    DISCLAIMER,
    _leading_sentences,
    based_on,
    build_markdown_report,
    dataset_period,
    key_findings,
    quality_summary,
)
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.models import AUTO_GOAL
from datastory.workflow.runner import AnalysisRunner
from scripts import generate_demo_data as gen
from tests.workflow_helpers import approve

ROWS = gen.build_rows()
DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
DEMO_XLSX = (gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes()
NOW = datetime(2026, 9, 26, 12, 0)


def rate(rows) -> float:
    return sum(r["Successful"] for r in rows) / sum(r["Transactions"] for r in rows) * 100


def analyze(client, dataset_id, kb, goal=AUTO_GOAL):
    runner = AnalysisRunner(WorkflowDeps.of(client, None, kb))
    snapshot = runner.start(dataset_id, goal)
    return runner.resume(snapshot.thread_id, approve(snapshot)).result


@pytest.fixture
def profile():
    raw = read_table(DEMO_XLSX, "transactions_2026.xlsx", "transactions")
    return build_profile(raw, "transactions_2026.xlsx", "transactions")[0]


@pytest.fixture
def documented(client, datasets):
    kb = KnowledgeBase()
    kb.index_document(DEMO_PDF, "business_metrics.pdf")
    return analyze(client, datasets["demo"], kb)


@pytest.fixture
def undocumented(client, datasets):
    return analyze(client, datasets["demo"], KnowledgeBase(), goal="Как менялась успешность?")


# ================================================================== сводка набора данных
def test_period_quality_and_description_helpers(profile):
    assert dataset_period(profile) == "2026-01 — 2026-06"
    assert quality_summary(profile) == "Без замечаний"

    frame = pd.DataFrame({"day": pd.to_datetime(["2026-03-05", "2026-03-20"]), "v": [1, 2]})
    assert dataset_period(build_profile(frame, "d.csv")[0]) == "2026-03-05 — 2026-03-20"
    single = pd.DataFrame({"day": pd.to_datetime(["2026-03-05", "2026-03-05"]), "v": [1, 2]})
    assert dataset_period(build_profile(single, "d.csv")[0]) == "2026-03-05"
    assert dataset_period(build_profile(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), "d.csv")[0]) is None


def test_quality_summary_counts_problems(profile):
    dirty = profile.model_copy(
        update={"missing_cell_count": 4, "duplicate_row_count": 2, "quality_issues": [DataQualityIssue(code="x", severity=Severity.WARNING, message="m")]}
    )
    assert quality_summary(dirty) == "замечаний: 1, пропущено ячеек: 4, дубликатов строк: 2"
    only_notes = profile.model_copy(update={"quality_issues": [DataQualityIssue(code="x", severity=Severity.INFO, message="m")]})
    assert quality_summary(only_notes) == "критичных замечаний нет"


def test_abbreviation_does_not_end_a_sentence():
    text = "Рост на 0,21 п.п. Минимум — 91,98%. Максимум — 97,88%."
    assert _leading_sentences(text, 1) == "Рост на 0,21 п.п. Минимум — 91,98%."
    assert _leading_sentences(text, 2) == "Рост на 0,21 п.п. Минимум — 91,98%. Максимум — 97,88%."


# ================================================================== ключевые выводы
def test_key_findings_put_the_warning_first_and_admit_the_unknown_cause(documented):
    findings = key_findings(documented)
    assert findings[0].startswith("**Динамика успешности.**") and "2026-02 → 2026-03" in findings[0]
    assert len(findings) >= 4 and any(f.startswith("Причины изменений по данным не установлены") for f in findings)
    assert not any("вызван" in f for f in findings)


def test_key_findings_mention_missing_documentation(undocumented):
    assert any("нет определений в документации" in f and "Success Rate" in f for f in key_findings(undocumented))


def test_based_on_names_the_data_and_the_context(documented):
    text = based_on(documented.insights[0])
    assert "MCP calculate_metrics" in text and "бизнес-контекст: business_metrics.pdf" in text


def test_based_on_without_documents_names_only_the_data(undocumented):
    text = based_on(undocumented.insights[0])
    assert "MCP calculate_metrics" in text and "бизнес-контекст" not in text


# ================================================================== Markdown-отчёт
def test_report_has_all_required_sections_in_order(documented, profile):
    report = build_markdown_report(documented, profile, NOW)
    headings = re.findall(r"^## (.+)$", report, re.MULTILINE)
    assert headings[:5] == [
        "Описание набора данных", "Ключевые показатели (KPI)", "Анализы", "Ключевые выводы", "Использованные источники базы знаний",
    ]
    assert report.startswith("# Отчёт DataStory AI: transactions_2026.xlsx") and DISCLAIMER in report
    assert "_Сформирован 2026-09-26 12:00._" in report


def test_report_describes_the_dataset(documented, profile):
    report = build_markdown_report(documented, profile, NOW)
    for line in ("**Строк:** 18", "**Колонок:** 6", "**Период данных:** 2026-01 — 2026-06", "**Качество данных:** Без замечаний", "**Цель анализа:** автоматический анализ"):
        assert line in report
    assert "| Amount_KZT | число | 0 |" in report and "| Month | дата/время | 0 |" in report


def test_report_kpis_match_python(documented, profile):
    report = build_markdown_report(documented, profile, NOW)
    transactions = sum(r["Transactions"] for r in ROWS)
    assert f"| Количество транзакций | {format_metric_value(transactions, 'шт.')} | SUM(Transactions) |" in report
    assert f"| Success Rate | {format_metric_value(rate(ROWS), '%')} |" in report


def test_report_lists_every_analysis_with_data_insight_and_basis(documented, profile):
    report = build_markdown_report(documented, profile, NOW)
    for number, title in enumerate(("Динамика количества операций", "Динамика объёма", "Динамика успешности", "Распределение по каналам"), 1):
        assert f"### {number}. {title}" in report
    assert "- **Показатель:** Success Rate = SUM(Successful) / SUM(Transactions) × 100 (%)" in report
    assert "- **Группировка:** Month" in report and "- **Визуализация:** круговая диаграмма" in report
    assert f"| 2026-03 | {format_metric_value(rate([r for r in ROWS if r['Month'] == '2026-03']), '%')} |" in report  # данные графика
    assert report.count("_На основе: MCP calculate_metrics") == 4 and "Числовые доказательства:" in report
    assert "**Ограничение интерпретации:** Причина изменения по данным не установлена" in report


def test_report_lists_the_knowledge_base_sources(documented, profile):
    report = build_markdown_report(documented, profile, NOW)
    sources = report.split("## Использованные источники базы знаний")[1]
    assert "business_metrics.pdf, стр. 1" in sources and "определение показателя «Success Rate»" in sources
    assert "> " in sources and "Использовано в выводах" in sources


def test_report_says_when_no_documents_were_used(undocumented, profile):
    report = build_markdown_report(undocumented, profile, NOW)
    assert "Документы базы знаний не использованы" in report
    assert "**Цель анализа:** Как менялась успешность?" in report
    assert "определение не найдено в документации" in report  # примечания


def test_report_works_without_a_stored_profile(documented):
    report = build_markdown_report(documented, None, NOW)
    assert "**Период данных:** не определён" in report and "**Строк:** 18" in report


def test_markdown_tables_are_well_formed(documented, profile):
    report = build_markdown_report(documented, profile, NOW)
    block: list[str] = []
    for line in [*report.splitlines(), ""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if block:
            widths = {len(re.findall(r"(?<!\\)\|", row)) for row in block}
            assert len(widths) == 1, block  # у всех строк таблицы одинаковое число столбцов
            block = []


def test_report_is_deterministic(documented, profile):
    assert build_markdown_report(documented, profile, NOW) == build_markdown_report(documented, profile, NOW)
