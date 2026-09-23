"""Живая проверка RAG на настоящих эмбеддингах OpenAI (text-embedding-3-small).

Запуск:  RUN_LIVE_TESTS=1 pytest tests/test_rag_live.py
Ключ берётся из .env. Тесты тратят несколько десятков токенов; в обычном прогоне пропускаются.
"""
import math

import pytest
from dotenv import dotenv_values

from datastory.config import PROJECT_ROOT
from datastory.file_processing.loader import read_table
from datastory.profiler.profiler import build_profile
from datastory.rag.embeddings import OpenAIEmbeddings
from datastory.rag.knowledge_base import KnowledgeBase
from scripts import generate_demo_data as gen

pytestmark = pytest.mark.live

DEMO_PDF = (gen.DEFAULT_OUT / "business_metrics.pdf").read_bytes()
DEMO_XLSX = (gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes()


@pytest.fixture(scope="module")
def live_kb(tmp_path_factory):
    key = (dotenv_values(PROJECT_ROOT / ".env").get("OPENAI_API_KEY") or "").strip()
    if not key:
        pytest.skip("OPENAI_API_KEY не задан в .env")
    kb = KnowledgeBase(OpenAIEmbeddings(key), tmp_path_factory.mktemp("live_chroma"))
    kb.index_document(DEMO_PDF, "business_metrics.pdf")
    profile = build_profile(read_table(DEMO_XLSX, "transactions_2026.xlsx", "transactions"), "transactions_2026.xlsx", "transactions")[0]
    kb.index_dataset_profile(profile)
    return kb


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


def test_live_embeddings_have_expected_shape_and_meaning(live_kb):
    embedder = live_kb.embedder
    assert embedder.name == "openai-text-embedding-3-small"
    query, related, unrelated = embedder.embed(["количество операций", "Transactions", "рецепт борща"])
    assert len(query) == 1536
    assert cosine(query, related) > cosine(query, unrelated) + 0.1


@pytest.mark.parametrize(
    ("question", "section"),
    [
        ("Как считается Success Rate?", "4. Формула Success Rate"),
        ("формула Success Rate", "4. Формула Success Rate"),
        ("Что такое Transaction Volume?", "5. Определение Transaction Volume"),
        ("Как рассчитывается средняя сумма транзакции?", "7. Формула Average Transaction Amount"),
        ("Что происходило в марте с инфраструктурой?", "9. Контекстное событие"),
    ],
)
def test_live_required_facts_are_in_top3(live_kb, question, section):
    result = live_kb.search_business_context(question)
    assert result.found and len(result.hits) == 3
    assert section in [h.section for h in result.hits], [(h.section, h.score) for h in result.hits]


@pytest.mark.parametrize(
    ("question", "section"),
    [
        ("формула Success Rate", "4. Формула Success Rate"),
        ("Что такое Transaction Volume?", "5. Определение Transaction Volume"),
        ("Что происходило в марте с инфраструктурой?", "9. Контекстное событие"),
    ],
)
def test_live_best_hit_is_the_right_section(live_kb, question, section):
    assert live_kb.search_business_context(question).hits[0].section == section


def test_live_irrelevant_question_is_not_found(live_kb):
    result = live_kb.search_business_context("рецепт борща со сметаной")
    assert not result.found and not result.relevant_hits


def test_live_columns_from_the_task_description(live_kb):
    operations = live_kb.resolve_column("Количество операций")
    assert (operations.status, operations.chosen.column_name) == ("confident", "Transactions")
    volume = live_kb.resolve_column("Объём транзакций")
    assert (volume.status, volume.chosen.column_name) == ("confident", "Amount_KZT")


def test_live_bare_word_is_ambiguous_and_nonsense_is_not_found(live_kb):
    assert live_kb.resolve_column("количество").status == "ambiguous"
    assert live_kb.resolve_column("погода в Астане").status == "not_found"


def test_live_repeated_indexing_does_not_call_api_again(live_kb):
    assert live_kb.index_document(DEMO_PDF, "copy.pdf").status == "already_indexed"
