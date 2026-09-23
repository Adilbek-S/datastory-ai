"""Страница «База знаний»: PDF-документы, метаданные датасетов и проверка семантического поиска."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from datastory.config import PROJECT_ROOT
from datastory.models import EvidenceType
from datastory.rag.embeddings import EmbeddingError
from datastory.rag.evidence import EVIDENCE_LABELS
from datastory.rag.knowledge_base import DEFAULT_TOP_K, KnowledgeBase
from datastory.rag.pdf_parser import PdfError, chunk_document, parse_pdf
from datastory.storage.store import DatasetStore
from datastory.ui.kb import get_kb, index_profile_safely
from datastory.ui.theme import kpi_row
from datastory.models import KPI

DEMO_PDF = PROJECT_ROOT / "data" / "demo" / "business_metrics.pdf"
NO_DATASET = "Без привязки (общий документ)"
SECTIONS = ["Документы", "Проверка поиска", "Датасеты"]
EXAMPLES = {
    "context": [
        "Как считается Success Rate?",
        "Что такое Transaction Volume?",
        "Формула средней суммы транзакции",
        "Что было в марте с инфраструктурой?",
    ],
    "columns": ["Количество операций", "Объём транзакций", "Сумма платежей", "Количество"],
}


def render() -> None:
    st.title("База знаний")
    st.caption("Документы с описанием бизнес-показателей и метаданные датасетов — для семантического поиска с указанием источников.")

    try:
        kb = get_kb()
    except EmbeddingError as exc:
        st.error(exc.user_message, icon=":material/error:")
        return

    _provider_banner(kb)
    _show_flash(st.container())  # слот всегда есть: иначе появление сообщения сдвигает элементы и сбрасывает вкладку
    try:
        stats = kb.stats()
    except Exception as exc:  # noqa: BLE001 — хранилище не должно ронять страницу
        st.error(f"Не удалось открыть хранилище ChromaDB: {exc}")
        return
    kpi_row(
        [
            KPI(label="Документов", value=str(stats["documents"]), hint="PDF в индексе"),
            KPI(label="Фрагментов", value=str(stats["document_chunks"]), hint="из документов"),
            KPI(label="Датасетов", value=str(stats["datasets"]), hint="описания в индексе"),
            KPI(label="Записей метаданных", value=str(stats["dataset_records"]), hint="датасеты и колонки"),
        ]
    )
    st.write("")

    # Явный переключатель раздела (значение хранится в сессии) вместо st.tabs: выбор не сбрасывается при перерисовке.
    section = st.radio("Раздел", SECTIONS, horizontal=True, key="kb_section", label_visibility="collapsed")
    if section == SECTIONS[0]:
        _documents_tab(kb)
    elif section == SECTIONS[1]:
        _search_tab(kb)
    else:
        _datasets_tab(kb)

    with st.expander("Как различаются утверждения"):
        st.markdown(
            f"- **{EVIDENCE_LABELS[EvidenceType.DOCUMENT_FACT]}** — прямо сказано в загруженном PDF; всегда указан источник.\n"
            f"- **{EVIDENCE_LABELS[EvidenceType.COMPUTED]}** — получено расчётом по данным (Pandas); указана формула.\n"
            f"- **{EVIDENCE_LABELS[EvidenceType.ASSUMPTION]}** — догадка системы или модели; документом не подтверждена.\n\n"
            "RAG ищет по смыслу и показывает источники. Точные числа он не считает — это делает Pandas."
        )


def _flash(kind: str, message: str) -> None:
    """Запоминает сообщение и перерисовывает страницу, чтобы счётчики и таблицы обновились."""
    st.session_state["kb_flash"] = (kind, message)
    st.rerun()


def _show_flash(slot) -> None:
    flash = st.session_state.pop("kb_flash", None)
    if flash:
        kind, message = flash
        with slot:
            {"success": st.success, "info": st.info, "warning": st.warning}[kind](message)


def _provider_banner(kb: KnowledgeBase) -> None:
    if kb.embedder.is_semantic:
        st.caption(f"Эмбеддинги: `{kb.embedder.name}` · хранилище ChromaDB")
    else:
        st.warning(
            "Ключ OpenAI не задан, поэтому работает **офлайн-режим**: поиск лексический (по словам), а не семантический, "
            "модель text-embedding-3-small не используется. Добавьте `OPENAI_API_KEY` в файл `.env` и перезапустите приложение.",
            icon=":material/warning:",
        )


# --------------------------------------------------------------------------- документы
def _documents_tab(kb: KnowledgeBase) -> None:
    saved = DatasetStore().list()
    options = {NO_DATASET: None, **{f"{r.filename} ({r.dataset_id})": r.dataset_id for r in saved}}

    left, right = st.columns([3, 2])
    pdf = left.file_uploader("PDF-документ", type=["pdf"], help="Описание бизнес-показателей, метрик, регламентов")
    target = right.selectbox("Привязать к датасету", list(options), help="Документ будет искаться в контексте этого датасета")

    if pdf is not None and _pdf_preview(pdf.getvalue(), pdf.name):
        if st.button("Проиндексировать документ", type="primary"):
            _index(kb, pdf.getvalue(), pdf.name, options[target])

    if DEMO_PDF.exists():
        if st.button("Добавить демо-документ business_metrics.pdf", icon=":material/description:"):
            _index(kb, DEMO_PDF.read_bytes(), DEMO_PDF.name, options[target])

    documents = kb.list_documents()
    st.markdown("#### Индексированные документы")
    if not documents:
        st.info("Документов пока нет. Загрузите PDF или добавьте демо-документ.")
        return
    st.dataframe(
        pd.DataFrame(
            {
                "Файл": d.filename,
                "Страниц": d.page_count,
                "Фрагментов": d.chunk_count,
                "Датасет": d.dataset_id or "—",
                "Индексирован (UTC)": d.indexed_at.replace("T", " ")[:16],
                "ID": d.document_id,
            }
            for d in documents
        ),
        hide_index=True, width="stretch",
    )
    with st.expander("Разделы документов"):
        for d in documents:
            st.markdown(f"**{d.filename}**: " + "; ".join(d.sections or ["разделы не определены"]))

    chosen = st.selectbox("Удалить документ из индекса", [f"{d.filename} · {d.document_id}" for d in documents], index=None,
                          placeholder="Выберите документ")
    if chosen and st.button("Удалить", icon=":material/delete:"):
        removed = kb.delete_document(chosen.rsplit("·", 1)[1].strip())
        _flash("success", f"Документ удалён из индекса (фрагментов: {removed}).")


def _pdf_preview(data: bytes, filename: str) -> bool:
    """Показывает, что найдено в PDF. False — файл нельзя индексировать (ошибка уже показана)."""
    try:
        parsed = parse_pdf(data, filename)
    except PdfError as exc:
        st.error(exc.user_message, icon=":material/error:")
        return False
    chunks = chunk_document(parsed)
    st.caption(f"Страниц: {parsed.page_count} · разделов: {len(parsed.section_titles)} · фрагментов для индекса: {len(chunks)}")
    return True


def _index(kb: KnowledgeBase, data: bytes, filename: str, dataset_id: str | None) -> None:
    try:
        with st.spinner("Индексируем документ…"):
            result = kb.index_document(data, filename, dataset_id)
    except PdfError as exc:
        st.error(exc.user_message, icon=":material/error:")
        return
    except EmbeddingError as exc:
        st.error(f"Не удалось получить эмбеддинги: {exc.user_message}", icon=":material/error:")
        return
    _flash("success" if result.status == "indexed" else "info", result.message)


# --------------------------------------------------------------------------- поиск
def _search_tab(kb: KnowledgeBase) -> None:
    mode = st.radio("Что искать", ["Бизнес-контекст (PDF)", "Колонки и датасеты"], horizontal=True)
    context_mode = mode.startswith("Бизнес")

    saved = DatasetStore().list()
    scope = {"Все датасеты": None, **{f"{r.filename} ({r.dataset_id})": r.dataset_id for r in saved}}
    dataset_id = scope[st.selectbox("Область поиска", list(scope))]

    st.caption("Примеры запросов:")
    cols = st.columns(len(EXAMPLES["context" if context_mode else "columns"]))
    for col, example in zip(cols, EXAMPLES["context" if context_mode else "columns"]):
        col.button(example, key=f"ex::{mode}::{example}", width="stretch",
                   on_click=lambda q=example: st.session_state.update(kb_query=q))
    query = st.text_input("Запрос", key="kb_query", placeholder="Например: формула Success Rate")
    if not query.strip():
        return

    try:
        if context_mode:
            _show_context(kb, query, dataset_id)
        else:
            _show_columns(kb, query, dataset_id)
    except EmbeddingError as exc:
        st.error(f"Не удалось получить эмбеддинг запроса: {exc.user_message}", icon=":material/error:")


def _show_context(kb: KnowledgeBase, query: str, dataset_id: str | None) -> None:
    result = kb.search_business_context(query, DEFAULT_TOP_K, dataset_id=dataset_id)
    if not result.hits:
        st.info("В базе знаний пока нет документов.")
        return
    if not result.found:
        st.warning(result.message, icon=":material/warning:")
    st.markdown(f"#### Найденные фрагменты (Top-{DEFAULT_TOP_K})")
    for hit in result.hits:
        with st.container(border=True):
            badge = ":green[релевантно]" if hit.relevant else ":gray[слабое совпадение]"
            st.markdown(f"**№{hit.rank} · {hit.source.section or 'Без раздела'}** · близость {hit.score:.3f} · {badge}")
            st.markdown(f"{hit.text}")
            st.caption(f":material/verified: {EVIDENCE_LABELS[hit.evidence_type]} · источник: {hit.source.citation} · `{hit.source.chunk_id}`")


def _show_columns(kb: KnowledgeBase, query: str, dataset_id: str | None) -> None:
    resolution = st.session_state.get("kb_resolution")
    if resolution is None or resolution.query != query:
        resolution = kb.resolve_column(query, dataset_id=dataset_id)
        st.session_state["kb_resolution"] = resolution

    if resolution.status == "not_found":
        st.warning(resolution.message, icon=":material/search_off:")
    elif resolution.status == "ambiguous":
        st.warning(resolution.message, icon=":material/help:")
        labels = {f"{h.column_name} · {h.filename} · близость {h.score:.2f}": h for h in resolution.candidates}
        pick = st.radio("Какую колонку вы имели в виду?", list(labels))
        if st.button("Подтвердить выбор", type="primary"):
            hit = labels[pick]
            st.session_state["kb_resolution"] = kb.confirm_column(resolution, hit.column_name, hit.dataset_id)
            st.rerun()
        _document_hint(kb, query, dataset_id)
    else:
        chosen = resolution.chosen
        if resolution.confirmed_by_user:
            st.success(f"Колонка «{chosen.column_name}» — подтверждено пользователем.", icon=":material/check_circle:")
        else:
            st.info(
                f"Запрос сопоставлен с колонкой **{chosen.column_name}** ({chosen.filename}). "
                f"{EVIDENCE_LABELS[EvidenceType.ASSUMPTION]} системы: пользователь пока не подтверждал.",
                icon=":material/lightbulb:",
            )

    result = kb.search_dataset_metadata(query, DEFAULT_TOP_K, dataset_id=dataset_id, entry_type="column")
    st.markdown(f"#### Ближайшие колонки (Top-{DEFAULT_TOP_K})")
    for hit in result.hits:
        with st.container(border=True):
            badge = ":green[релевантно]" if hit.relevant else ":gray[слабое совпадение]"
            st.markdown(f"**№{hit.rank} · {hit.column_name}** ({hit.column_kind}) · близость {hit.score:.3f} · {badge}")
            st.caption(hit.text.replace("\n", " "))
            st.caption(f"Метаданные датасета (вычислены профайлером) · источник: {hit.source.citation}")


def _document_hint(kb: KnowledgeBase, query: str, dataset_id: str | None) -> None:
    """При неоднозначности показываем, что об этом говорят документы: это помогает выбрать колонку."""
    context = kb.search_business_context(query, 1, dataset_id=dataset_id)
    if context.found:
        hit = context.hits[0]
        st.info(f"Подсказка из документа ({hit.source.citation}):\n\n{hit.text}", icon=":material/menu_book:")


# --------------------------------------------------------------------------- датасеты
def _datasets_tab(kb: KnowledgeBase) -> None:
    store = DatasetStore()
    saved = store.list()
    if not saved:
        st.info("Подтверждённых датасетов пока нет. Загрузите и подтвердите таблицу на странице «Анализ данных».")
        return
    indexed = kb.indexed_dataset_ids()
    st.dataframe(
        pd.DataFrame(
            {
                "ID": r.dataset_id, "Файл": r.filename, "Лист": r.sheet_name or "",
                "Строк": r.row_count, "Колонок": r.column_count,
                "В индексе": "да" if r.dataset_id in indexed else "нет",
            }
            for r in saved
        ),
        hide_index=True, width="stretch",
    )
    st.caption("В индекс попадает описание датасета и колонок (названия, типы, безопасные примеры, период), а не строки таблицы.")
    if st.button("Проиндексировать все сохранённые датасеты"):
        results = [(ref.filename, *index_profile_safely(store.load_profile(ref.dataset_id))) for ref in saved]
        failed = [f"{name}: {message}" for name, ok, message in results if not ok]
        _flash("warning" if failed else "success", "\n\n".join(failed) if failed else f"Датасетов в индексе: {len(results)}.")
