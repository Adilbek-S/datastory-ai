"""Факт из документа, результат вычисления и предположение: модель не может выдать догадку за факт."""
import pytest

from datastory.insights.generator import generate_insights
from datastory.models import EvidenceType
from datastory.profiler.profiler import build_profile
from datastory.rag.embeddings import OfflineHashEmbeddings
from datastory.rag.evidence import (
    EVIDENCE_LABELS,
    GROUNDING_RULES,
    EvidenceItem,
    context_from_result,
    format_context_for_prompt,
    render_evidence,
    validate_evidence,
)
from datastory.rag.knowledge_base import KnowledgeBase
from scripts import generate_demo_data as gen
import pandas as pd

DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
FACT, COMPUTED, ASSUMPTION = EvidenceType.DOCUMENT_FACT, EvidenceType.COMPUTED, EvidenceType.ASSUMPTION


@pytest.fixture
def search(tmp_path):
    kb = KnowledgeBase(OfflineHashEmbeddings(), tmp_path / "kb")
    kb.index_document(DEMO_PDF, "business_metrics.pdf")
    return kb.search_business_context("плановое обновление инфраструктуры в марте")


@pytest.fixture
def context(search):
    return context_from_result(search)


@pytest.fixture
def march_source(search):
    return search.hits[0].source


def fact(source, quote, statement="В марте 2026 года проводилось плановое обновление инфраструктуры."):
    return EvidenceItem(type=FACT, statement=statement, sources=[source], quote=quote)


# ------------------------------------------------------------------ факт из документа
def test_verified_fact_stays_a_fact(march_source, context):
    item = fact(march_source, "проводилось плановое обновление инфраструктуры обработки транзакций")
    (checked,) = validate_evidence([item], context)
    assert checked.type is FACT and checked.note == "" and checked.label == "Факт из документа"


def test_quote_matching_ignores_case_spaces_and_quote_marks(march_source, context):
    item = fact(march_source, "  ПРОВОДИЛОСЬ   плановое\nобновление  инфраструктуры ")
    assert validate_evidence([item], context)[0].type is FACT


def test_fact_without_source_is_downgraded(context):
    item = EvidenceItem(type=FACT, statement="Success Rate считается иначе.", quote="что-то")
    (checked,) = validate_evidence([item], context)
    assert checked.type is ASSUMPTION and "не указан источник" in checked.note


def test_fact_without_quote_is_downgraded(march_source, context):
    item = EvidenceItem(type=FACT, statement="Было обновление.", sources=[march_source])
    (checked,) = validate_evidence([item], context)
    assert checked.type is ASSUMPTION and "нет дословной цитаты" in checked.note


def test_invented_quote_is_downgraded(march_source, context):
    item = fact(march_source, "обновление вызвало падение успешности на 6 процентных пунктов")
    (checked,) = validate_evidence([item], context)
    assert checked.type is ASSUMPTION and "цитата не найдена" in checked.note


def test_source_outside_retrieved_context_is_not_trusted(march_source):
    item = fact(march_source, "плановое обновление инфраструктуры")
    (checked,) = validate_evidence([item], context={})  # фрагмент не был найден поиском
    assert checked.type is ASSUMPTION


def test_causal_claim_is_not_a_document_fact(march_source, context):
    """Документ упоминает обновление, но прямо отрицает доказанную причинность."""
    item = fact(
        march_source,
        "проводилось плановое обновление инфраструктуры обработки транзакций",
        statement="Снижение успешности в марте вызвано обновлением инфраструктуры.",
    )
    (checked,) = validate_evidence([item], context)
    assert checked.type is ASSUMPTION and "причине" in checked.note


def test_document_statement_about_causality_can_be_a_fact(march_source, context):
    item = fact(
        march_source,
        "наличие этого события не доказывает причинную связь",
        statement="Документ указывает, что наличие события не доказывает причинную связь.",
    )
    assert validate_evidence([item], context)[0].type is FACT


# ------------------------------------------------------------------ вычисление и предположение
def test_computed_result_needs_a_method(context):
    with_formula = EvidenceItem(type=COMPUTED, statement="Success Rate марта 91,98%.", computation="Successful / Transactions × 100 по строкам марта")
    without = EvidenceItem(type=COMPUTED, statement="Success Rate марта 91,98%.")
    ok, downgraded = validate_evidence([with_formula, without], context)
    assert ok.type is COMPUTED and downgraded.type is ASSUMPTION
    assert "способ вычисления" in downgraded.note


def test_assumption_is_kept_as_is(context):
    item = EvidenceItem(type=ASSUMPTION, statement="Возможно, часть отказов связана с обновлением.")
    (checked,) = validate_evidence([item], context)
    assert checked == item and checked.label == "Предположение"


def test_validation_returns_new_items_without_mutating_input(march_source, context):
    original = EvidenceItem(type=FACT, statement="x", sources=[march_source], quote="нет такой цитаты")
    validate_evidence([original], context)
    assert original.type is FACT and original.note == ""


def test_mixed_answer_keeps_three_kinds_apart(march_source, context):
    items = validate_evidence(
        [
            fact(march_source, "плановое обновление инфраструктуры обработки транзакций"),
            EvidenceItem(type=COMPUTED, statement="Success Rate в марте 91,98%", computation="Successful / Transactions × 100"),
            EvidenceItem(type=ASSUMPTION, statement="Обновление могло повлиять на успешность."),
            fact(march_source, "выдуманная цитата", statement="Обновление снизило успешность."),
        ],
        context,
    )
    assert [i.type for i in items] == [FACT, COMPUTED, ASSUMPTION, ASSUMPTION]
    rendered = render_evidence(items)
    assert rendered.count("[Факт из документа]") == 1
    assert rendered.count("[Результат вычисления]") == 1
    assert rendered.count("[Предположение]") == 2
    assert "business_metrics.pdf, стр. 2, раздел «9. Контекстное событие»" in rendered
    assert "Successful / Transactions × 100" in rendered


# ------------------------------------------------------------------ подготовка промпта
def test_prompt_context_lists_ids_and_citations(search):
    text = format_context_for_prompt(search.hits[:2])
    for hit in search.hits[:2]:
        assert f"[{hit.source.chunk_id}]" in text and hit.source.citation in text and hit.text in text
    assert format_context_for_prompt([]) == "Фрагменты документов не найдены."


def test_grounding_rules_mention_all_three_types():
    for name in ("document_fact", "computed_result", "model_assumption", "причинную связь"):
        assert name in GROUNDING_RULES
    assert set(EVIDENCE_LABELS) == set(EvidenceType)


def test_context_from_result_maps_chunk_ids_to_text(search):
    mapping = context_from_result(search)
    assert set(mapping) == {h.source.chunk_id for h in search.hits}


# ------------------------------------------------------------------ выводы по данным — это вычисления
def test_rule_based_insights_are_marked_as_computed():
    profile, _ = build_profile(pd.DataFrame({"a": [1, 1, None]}), "t.csv")
    insights = generate_insights(profile)
    assert insights and all(i.evidence_type is COMPUTED for i in insights)
