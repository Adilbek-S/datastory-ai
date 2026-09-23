"""Индексация и поиск: PDF-документы и метаданные датасетов на ChromaDB (детерминированные офлайн-эмбеддинги)."""
import pandas as pd
import pytest

from datastory.file_processing.loader import read_table
from datastory.profiler.profiler import build_profile
from datastory.rag import api
from datastory.rag.embeddings import EmbeddingError, OfflineHashEmbeddings
from datastory.rag.knowledge_base import KnowledgeBase, SourceNotFoundError
from datastory.rag.models import SourceKind
from datastory.rag.pdf_parser import PdfError
from scripts import generate_demo_data as gen
from tests.helpers import make_pdf

DEMO_DIR = gen.DEFAULT_OUT
DEMO_PDF = (DEMO_DIR / "business_metrics.pdf").read_bytes()
DEMO_XLSX = (DEMO_DIR / "transactions_2026.xlsx").read_bytes()


class SpyEmbedder(OfflineHashEmbeddings):
    """Считает и запоминает тексты, отправленные на эмбеддинг; может имитировать сбой API."""

    def __init__(self):
        self.texts: list[str] = []
        self.fail = False

    def embed(self, texts):
        if self.fail:
            raise EmbeddingError("сбой сети")
        self.texts.extend(texts)
        return super().embed(texts)


class OtherEmbedder(OfflineHashEmbeddings):
    name = "other-provider-v1"


@pytest.fixture
def embedder() -> SpyEmbedder:
    return SpyEmbedder()


@pytest.fixture
def kb(tmp_path, embedder) -> KnowledgeBase:
    return KnowledgeBase(embedder, tmp_path / "kb")


@pytest.fixture(scope="module")
def demo_profile():
    raw = read_table(DEMO_XLSX, "transactions_2026.xlsx", "transactions")
    return build_profile(raw, "transactions_2026.xlsx", "transactions")[0]


def top_sections(result, n=1) -> list[str]:
    return [h.section for h in result.hits[:n]]


# ================================================================== документы: индексация
def test_index_demo_document(kb):
    result = kb.index_document(DEMO_PDF, "business_metrics.pdf")
    assert (result.status, result.chunk_count, result.filename) == ("indexed", 10, "business_metrics.pdf")
    assert kb.stats()["documents"] == 1 and kb.stats()["document_chunks"] == 10

    (info,) = kb.list_documents()
    assert info.filename == "business_metrics.pdf" and info.page_count == 2 and info.chunk_count == 10
    assert info.document_id == result.document_id and info.dataset_id is None
    assert info.sections[0].startswith("DemoPay KZ") and "9. Контекстное событие" in info.sections


def test_every_chunk_stores_required_metadata(kb):
    kb.index_document(DEMO_PDF, "business_metrics.pdf", dataset_id="abc123abc123")
    stored = kb._docs.get(include=["metadatas", "documents"])
    assert len(stored["ids"]) == 10
    for chunk_id, meta, text in zip(stored["ids"], stored["metadatas"], stored["documents"]):
        assert meta["chunk_id"] == chunk_id
        assert meta["document_id"] == chunk_id.rsplit("-", 1)[0]
        assert meta["filename"] == "business_metrics.pdf"
        assert meta["page"] in (1, 2) and meta["dataset_id"] == "abc123abc123"
        assert meta["section"] and text


def test_repeated_indexing_is_prevented(kb, embedder):
    first = kb.index_document(DEMO_PDF, "business_metrics.pdf")
    embedded_once = len(embedder.texts)
    second = kb.index_document(DEMO_PDF, "business_metrics.pdf")

    assert (first.status, second.status) == ("indexed", "already_indexed")
    assert second.document_id == first.document_id and second.chunk_count == 10
    assert len(embedder.texts) == embedded_once, "повторная индексация не должна вызывать эмбеддинги"
    assert kb.stats()["document_chunks"] == 10 and len(kb.list_documents()) == 1


def test_renamed_copy_is_detected_by_content(kb):
    kb.index_document(DEMO_PDF, "business_metrics.pdf")
    copy = kb.index_document(DEMO_PDF, "копия (1).pdf")
    assert copy.status == "already_indexed" and copy.filename == "business_metrics.pdf"
    assert "business_metrics.pdf" in copy.message
    assert len(kb.list_documents()) == 1


def test_reindex_with_other_dataset_keeps_original_link(kb):
    kb.index_document(DEMO_PDF, "b.pdf", dataset_id="aaaaaaaaaaaa")
    again = kb.index_document(DEMO_PDF, "b.pdf", dataset_id="bbbbbbbbbbbb")
    assert again.status == "already_indexed" and again.dataset_id == "aaaaaaaaaaaa"
    assert "Привязка" in again.message
    assert kb.list_documents()[0].dataset_id == "aaaaaaaaaaaa"


def test_different_documents_are_indexed_separately(kb):
    other = make_pdf([("h", "1. Возвраты"), ("p", "Возврат — отмена ранее проведённой транзакции.")])
    kb.index_document(DEMO_PDF, "a.pdf")
    assert kb.index_document(other, "refunds.pdf").status == "indexed"
    assert {d.filename for d in kb.list_documents()} == {"a.pdf", "refunds.pdf"}


def test_incomplete_indexing_is_repaired(kb):
    kb.index_document(DEMO_PDF, "b.pdf")
    kb._docs.delete(ids=[kb._docs.get(include=[])["ids"][3]])  # имитируем оборванную запись
    assert kb.stats()["document_chunks"] == 9

    repaired = kb.index_document(DEMO_PDF, "b.pdf")
    assert repaired.status == "indexed" and kb.stats()["document_chunks"] == 10


def test_embedding_failure_leaves_no_partial_document(kb, embedder):
    embedder.fail = True
    with pytest.raises(EmbeddingError):
        kb.index_document(DEMO_PDF, "b.pdf")
    assert kb.list_documents() == [] and kb.stats()["document_chunks"] == 0

    embedder.fail = False
    assert kb.index_document(DEMO_PDF, "b.pdf").status == "indexed"


def test_invalid_pdf_is_rejected_before_embedding(kb, embedder):
    with pytest.raises(PdfError):
        kb.index_document(b"not a pdf", "bad.pdf")
    assert embedder.texts == [] and kb.list_documents() == []


def test_index_survives_restart(tmp_path, embedder):
    path = tmp_path / "persist"
    KnowledgeBase(embedder, path).index_document(DEMO_PDF, "b.pdf")
    reopened = KnowledgeBase(SpyEmbedder(), path)
    assert reopened.stats()["documents"] == 1
    assert reopened.index_document(DEMO_PDF, "b.pdf").status == "already_indexed"


def test_delete_document(kb):
    document_id = kb.index_document(DEMO_PDF, "b.pdf").document_id
    assert kb.delete_document(document_id) == 10
    assert kb.list_documents() == [] and kb.delete_document(document_id) == 0
    assert kb.index_document(DEMO_PDF, "b.pdf").status == "indexed"  # после удаления можно индексировать снова


def test_providers_use_separate_collections(tmp_path):
    path = tmp_path / "shared"
    KnowledgeBase(OfflineHashEmbeddings(), path).index_document(DEMO_PDF, "b.pdf")
    other = KnowledgeBase(OtherEmbedder(), path)
    assert other.stats()["documents"] == 0  # векторы разных моделей не смешиваются
    assert other.index_document(DEMO_PDF, "b.pdf").status == "indexed"


# ================================================================== документы: поиск
@pytest.fixture
def indexed(kb):
    kb.index_document(DEMO_PDF, "business_metrics.pdf")
    return kb


def test_finds_success_rate_formula(indexed):
    result = indexed.search_business_context("формула Success Rate")
    top = result.hits[0]
    assert top.section == "4. Формула Success Rate"
    assert "Success Rate = Successful / Transactions × 100" in top.text
    assert result.found and top.relevant


def test_finds_transaction_volume_definition(indexed):
    top = indexed.search_business_context("Что такое Transaction Volume?").hits[0]
    assert top.section == "5. Определение Transaction Volume"
    assert "денежный объём платежей" in top.text and "Amount_KZT" in top.text


def test_finds_average_transaction_amount_formula(indexed):
    top = indexed.search_business_context("формула Average Transaction Amount").hits[0]
    assert top.section == "7. Формула Average Transaction Amount"
    assert "Average Transaction Amount = Amount_KZT / Transactions" in top.text


def test_finds_march_infrastructure_event(indexed):
    top = indexed.search_business_context("плановое обновление инфраструктуры в марте").hits[0]
    assert top.section == "9. Контекстное событие"
    assert "плановое обновление инфраструктуры обработки транзакций" in top.text
    assert "не доказывает причинную связь" in top.text  # оговорка о причинности приходит вместе с фактом


def test_formula_is_in_top3_for_natural_question(indexed):
    sections = top_sections(indexed.search_business_context("Как считается Success Rate?"), 3)
    assert "4. Формула Success Rate" in sections


def test_default_is_top3_sorted_with_sources(indexed):
    result = indexed.search_business_context("Success Rate")
    assert [h.rank for h in result.hits] == [1, 2, 3]
    assert [h.score for h in result.hits] == sorted((h.score for h in result.hits), reverse=True)
    for hit in result.hits:
        assert hit.source.kind is SourceKind.DOCUMENT and hit.source.filename == "business_metrics.pdf"
        assert hit.source.chunk_id and hit.source.page == hit.page and hit.source.section == hit.section
        assert hit.evidence_type.value == "document_fact"


def test_top_k_is_respected_and_capped(indexed):
    assert len(indexed.search_business_context("Success Rate", 1).hits) == 1
    assert len(indexed.search_business_context("Success Rate", 5).hits) == 5
    assert len(indexed.search_business_context("Success Rate", 50).hits) == 10  # больше, чем есть в индексе


def test_irrelevant_query_is_reported_as_not_found(indexed):
    result = indexed.search_business_context("рецепт борща со сметаной")
    assert not result.found and not result.relevant_hits
    assert "нельзя опирать на документы" in result.message


def test_empty_queries_and_empty_index(kb, indexed):
    assert indexed.search_business_context("   ").hits == []
    empty = KnowledgeBase(SpyEmbedder(), kb._client.get_settings().persist_directory + "_empty")
    result = empty.search_business_context("формула Success Rate")
    assert result.hits == [] and not result.found


def test_search_scope_by_dataset_and_document(kb):
    second = make_pdf([("h", "1. Формула Success Rate"), ("p", "Общая методика: Success Rate — доля успешных операций.")])
    attached = kb.index_document(DEMO_PDF, "business_metrics.pdf", dataset_id="aaaaaaaaaaaa")
    shared = kb.index_document(second, "general.pdf")  # без привязки — доступен всем датасетам

    for_a = kb.search_business_context("формула Success Rate", dataset_id="aaaaaaaaaaaa")
    assert {h.source.document_id for h in for_a.hits} == {attached.document_id, shared.document_id}

    for_b = kb.search_business_context("формула Success Rate", dataset_id="bbbbbbbbbbbb")
    assert {h.source.document_id for h in for_b.hits} == {shared.document_id}

    only_doc = kb.search_business_context("формула Success Rate", document_id=attached.document_id)
    assert {h.source.document_id for h in only_doc.hits} == {attached.document_id}
    assert only_doc.hits[0].source.dataset_id == "aaaaaaaaaaaa"


# ================================================================== источники
def test_get_source_reference_matches_hit_source(indexed):
    hit = indexed.search_business_context("формула Success Rate").hits[0]
    source = indexed.get_source_reference(hit.source.chunk_id)
    assert source == hit.source
    assert source.citation == "business_metrics.pdf, стр. 1, раздел «4. Формула Success Rate»"
    assert "Success Rate = Successful" in indexed.get_chunk_text(hit.source.chunk_id)


def test_source_of_second_page_and_unknown_id(indexed):
    hit = indexed.search_business_context("плановое обновление инфраструктуры").hits[0]
    assert hit.source.page == 2 and "стр. 2" in hit.source.citation
    for missing in ("nope", "", "bddc6a1ce878949d-9999"):
        with pytest.raises(SourceNotFoundError):
            indexed.get_source_reference(missing)


# ================================================================== датасеты: индексация
def test_dataset_is_indexed_as_metadata_not_rows(kb, demo_profile):
    result = kb.index_dataset_profile(demo_profile)
    assert (result.status, result.record_count) == ("indexed", 1 + 6)  # датасет + 6 колонок, а не 18 строк
    assert kb.stats()["datasets"] == 1 and kb.stats()["dataset_records"] == 7

    many_rows = pd.DataFrame({"a": range(1000), "b": ["x", "y"] * 500})
    profile, _ = build_profile(many_rows, "big.csv")
    assert kb.index_dataset_profile(profile).record_count == 3  # 1000 строк -> 3 записи


def test_dataset_description_contents(kb, demo_profile):
    kb.index_dataset_profile(demo_profile)
    text = kb._datasets.get(ids=[f"ds-{demo_profile.dataset_id}"], include=["documents"])["documents"][0]
    for expected in (
        "transactions_2026.xlsx",  # название датасета
        demo_profile.description,  # описание
        "Transactions — число", "Month — дата/время", "Channel — категория",  # колонки и типы
        "с 2026-01-01 по 2026-06-01",  # период
        "Предполагаемые измерения", "Channel (Mobile, Web, API)",  # измерения и безопасные значения
        "Предполагаемые показатели", "Amount_KZT",  # показатели
    ):
        assert expected in text, expected


def test_column_records_have_meaning_type_and_safe_examples(kb, demo_profile):
    kb.index_dataset_profile(demo_profile)
    records = kb._datasets.get(where={"entry_type": "column"}, include=["documents", "metadatas"])
    by_name = dict(zip((m["column_name"] for m in records["metadatas"]), records["documents"]))
    assert set(by_name) == {"Month", "Transactions", "Successful", "Failed", "Amount_KZT", "Channel"}
    assert "количество операций" in by_name["Transactions"] and "Тип: число" in by_name["Transactions"]
    assert "объём транзакций" in by_name["Amount_KZT"] and "тенге" in by_name["Amount_KZT"]
    assert "Mobile" in by_name["Channel"] and "Период: с 2026-01-01 по 2026-06-01" in by_name["Month"]
    assert "предполагаемый смысл" in by_name["Transactions"]  # синонимы подписаны как предположение


def test_personal_data_never_reaches_index_or_embedder(kb, embedder):
    frame = pd.DataFrame(
        {
            "email": ["ivan@example.kz", "aida@mail.ru", "bolat@corp.com"],
            "Full Name": ["Иванов Иван Иванович", "Петрова Анна Сергеевна", "Сидоров Пётр Олегович"],
            "notes": ["позвонить после шести вечера", "просил не беспокоить", "перезвонить завтра"],
            "amount": [100, 200, 300],
        }
    )
    profile, _ = build_profile(frame, "clients_ivanov.xlsx")
    kb.index_dataset_profile(profile)

    stored = " ".join(kb._datasets.get(include=["documents"])["documents"]) + " ".join(embedder.texts)
    for secret in ("ivan@example.kz", "aida@mail.ru", "Иванов", "Петрова", "Сидоров", "позвонить", "не беспокоить"):
        assert secret not in stored, secret
    assert "email" in stored and "Full Name" in stored  # названия колонок остаются


def test_dataset_reindex_is_idempotent_and_updates_changes(kb, embedder, demo_profile):
    first = kb.index_dataset_profile(demo_profile)
    embedded = len(embedder.texts)
    again = kb.index_dataset_profile(demo_profile)
    assert (first.status, again.status) == ("indexed", "unchanged") and len(embedder.texts) == embedded

    changed = demo_profile.model_copy(update={"description": "Новое описание набора"})
    assert kb.index_dataset_profile(changed).status == "indexed"
    assert kb.stats()["dataset_records"] == 7
    text = kb._datasets.get(ids=[f"ds-{demo_profile.dataset_id}"], include=["documents"])["documents"][0]
    assert "Новое описание набора" in text


def test_stale_column_records_are_removed(kb):
    wide, _ = build_profile(pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}), "t.csv", dataset_id="aaaaaaaaaaaa")
    narrow, _ = build_profile(pd.DataFrame({"a": [1, 2]}), "t.csv", dataset_id="aaaaaaaaaaaa")
    kb.index_dataset_profile(wide)
    assert kb.stats()["dataset_records"] == 4
    kb.index_dataset_profile(narrow)
    assert kb.stats()["dataset_records"] == 2


def test_dataset_embedding_failure_keeps_previous_index(kb, embedder, demo_profile):
    kb.index_dataset_profile(demo_profile)
    embedder.fail = True
    with pytest.raises(EmbeddingError):
        kb.index_dataset_profile(demo_profile.model_copy(update={"description": "другое"}))
    assert kb.stats()["dataset_records"] == 7  # старое описание не потеряно


def test_delete_dataset(kb, demo_profile):
    kb.index_dataset_profile(demo_profile)
    assert kb.delete_dataset(demo_profile.dataset_id) == 7 and kb.stats()["datasets"] == 0


# ================================================================== датасеты: поиск и неоднозначность
@pytest.fixture
def indexed_dataset(kb, demo_profile):
    kb.index_dataset_profile(demo_profile)
    return kb


@pytest.mark.parametrize(
    ("phrase", "column"),
    [
        ("Количество операций", "Transactions"),
        ("Объём транзакций", "Amount_KZT"),
        ("Сумма платежей", "Amount_KZT"),
        ("Число успешных операций", "Successful"),
        ("неуспешные транзакции", "Failed"),
        ("период", "Month"),
    ],
)
def test_phrases_are_mapped_to_columns(indexed_dataset, phrase, column):
    resolution = indexed_dataset.resolve_column(phrase)
    assert resolution.status == "confident" and resolution.chosen.column_name == column
    assert not resolution.confirmed_by_user and "предположение" in resolution.message
    assert resolution.chosen.source.kind is SourceKind.DATASET and resolution.chosen.source.column_name == column


def test_exact_synonym_beats_embedding_similarity(indexed_dataset):
    """«Объём транзакций» лексически ближе к Transactions, но это синоним Amount_KZT."""
    hit = indexed_dataset.resolve_column("Объём транзакций").chosen
    assert hit.matched_alias == "объём транзакций" and hit.raw_score < hit.score


def test_unrelated_phrase_is_not_found(indexed_dataset):
    resolution = indexed_dataset.resolve_column("погода в Астане")
    assert resolution.status == "not_found" and resolution.chosen is None


def test_similar_columns_require_user_confirmation(kb):
    frame = pd.DataFrame({"Amount_KZT": [1, 2, 3], "Amount_USD": [4, 5, 6], "Month": ["2026-01", "2026-02", "2026-03"]})
    profile, _ = build_profile(frame, "payments.xlsx")
    kb.index_dataset_profile(profile)

    resolution = kb.resolve_column("сумма")
    assert resolution.status == "ambiguous" and resolution.needs_confirmation and resolution.chosen is None
    assert {h.column_name for h in resolution.candidates} == {"Amount_KZT", "Amount_USD"}
    assert "Подтвердите" in resolution.message

    confirmed = kb.confirm_column(resolution, "Amount_USD")
    assert confirmed.status == "confirmed" and confirmed.confirmed_by_user
    assert confirmed.chosen.column_name == "Amount_USD" and not confirmed.needs_confirmation
    with pytest.raises(ValueError):
        kb.confirm_column(resolution, "Month")  # не входит в число кандидатов


def test_same_column_in_two_datasets_is_ambiguous_until_scoped(kb):
    first, _ = build_profile(pd.DataFrame({"Amount_KZT": [1, 2]}), "payments_2025.xlsx")
    second, _ = build_profile(pd.DataFrame({"Amount_KZT": [3, 4]}), "payments_2026.xlsx")
    kb.index_dataset_profile(first)
    kb.index_dataset_profile(second)

    assert kb.resolve_column("Объём транзакций").status == "ambiguous"
    scoped = kb.resolve_column("Объём транзакций", dataset_id=second.dataset_id)
    assert scoped.status == "confident" and scoped.chosen.dataset_id == second.dataset_id


def test_search_dataset_metadata_top3_and_entry_type(indexed_dataset, demo_profile):
    result = indexed_dataset.search_dataset_metadata("транзакции по каналам за месяц")
    assert len(result.hits) == 3 and [h.rank for h in result.hits] == [1, 2, 3]

    columns = indexed_dataset.search_dataset_metadata("канал", entry_type="column")
    assert columns.hits[0].column_name == "Channel" and all(h.entry_type == "column" for h in columns.hits)

    datasets = indexed_dataset.search_dataset_metadata("датасет transactions_2026", entry_type="dataset")
    assert len(datasets.hits) == 1 and datasets.hits[0].column_name is None
    assert datasets.hits[0].dataset_id == demo_profile.dataset_id
    assert datasets.hits[0].source.citation.startswith("датасет «transactions_2026.xlsx»")


def test_finds_the_right_dataset_among_several(kb, demo_profile):
    kb.index_dataset_profile(demo_profile)
    staff, _ = build_profile(
        pd.DataFrame({"Department": ["IT", "HR", "IT"], "Salary": [500, 400, 600], "Hired": ["2024-01-10", "2024-02-11", "2024-03-12"]}),
        "staff.xlsx",
    )
    kb.index_dataset_profile(staff)
    top = kb.search_dataset_metadata("успешные транзакции платежи по каналам", entry_type="dataset").hits[0]
    assert top.dataset_id == demo_profile.dataset_id


def test_source_reference_for_dataset_records(indexed_dataset, demo_profile):
    hit = indexed_dataset.resolve_column("Количество операций").chosen
    source = indexed_dataset.get_source_reference(hit.source.chunk_id)
    assert source.kind is SourceKind.DATASET and source.column_name == "Transactions"
    assert source.dataset_id == demo_profile.dataset_id
    assert source.citation == f"датасет «transactions_2026.xlsx» ({demo_profile.dataset_id}), колонка «Transactions»"


# ================================================================== функции верхнего уровня
def test_public_api_functions(kb, demo_profile):
    assert api.index_document(DEMO_PDF, "business_metrics.pdf", kb=kb).status == "indexed"
    assert api.index_dataset_profile(demo_profile, kb=kb).status == "indexed"

    context = api.search_business_context("формула Success Rate", kb=kb)
    assert context.hits[0].section == "4. Формула Success Rate" and len(context.hits) == 3

    metadata = api.search_dataset_metadata("количество операций", entry_type="column", kb=kb)
    assert metadata.hits[0].column_name == "Transactions"
    assert api.resolve_column("Количество операций", kb=kb).chosen.column_name == "Transactions"

    source = api.get_source_reference(context.hits[0].source.chunk_id, kb=kb)
    assert source.section == "4. Формула Success Rate"


def test_default_knowledge_base_uses_settings_and_offline_provider():
    api.index_document(DEMO_PDF, "b.pdf")
    kb = api.get_default_knowledge_base()
    assert kb.embedder.name == "offline-hash-v1" and kb.stats()["documents"] == 1
    assert api.search_business_context("формула Success Rate").hits[0].section == "4. Формула Success Rate"
