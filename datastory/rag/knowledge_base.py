"""База знаний на ChromaDB: документы (PDF) и метаданные датасетов.

Две коллекции на каждого провайдера эмбеддингов (векторы разных моделей несовместимы):
  documents__<provider>  — фрагменты PDF
  datasets__<provider>   — описания датасетов и колонок (без строк таблицы)

RAG ищет по смыслу и возвращает источники. Точные числа он не считает и не хранит.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from datastory.config import get_settings
from datastory.models import DatasetProfile
from datastory.rag.dataset_descriptions import build_dataset_records, content_hash
from datastory.rag.embeddings import EmbeddingProvider, get_embedding_provider
from datastory.rag.models import (
    ColumnResolution,
    ContextHit,
    ContextSearchResult,
    DatasetIndexResult,
    DocumentChunk,
    DocumentIndexResult,
    DocumentInfo,
    MetadataHit,
    MetadataSearchResult,
    SourceKind,
    SourceReference,
)
from datastory.rag.pdf_parser import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, PdfError, chunk_document, parse_pdf

DEFAULT_TOP_K = 3  # для MVP: три лучших фрагмента
EXACT_MATCH_BOOST = 0.15  # бонус за точное совпадение запроса с синонимом/названием колонки


def _phrase(text: str) -> str:
    """Нормализация фразы для точного сравнения: регистр, «ё», знаки препинания, подчёркивания."""
    return " ".join(re.sub(r"[^0-9a-zа-я]+", " ", text.lower().replace("ё", "е")).split())


class SourceNotFoundError(LookupError):
    """Фрагмент с таким chunk_id не найден в индексе."""


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")



class KnowledgeBase:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        chroma_dir: Path | None = None,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap: int = DEFAULT_OVERLAP,
    ):
        self.embedder = embedder or get_embedding_provider()
        self.max_chars, self.overlap = max_chars, overlap
        path = Path(chroma_dir) if chroma_dir is not None else get_settings().chroma_dir
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        suffix = _slug(self.embedder.name)
        cosine = {"hnsw": {"space": "cosine"}}
        self._docs = self._client.get_or_create_collection(f"documents__{suffix}", configuration=cosine, embedding_function=None)
        self._datasets = self._client.get_or_create_collection(f"datasets__{suffix}", configuration=cosine, embedding_function=None)

    # ------------------------------------------------------------------ документы
    def index_document(self, data: bytes, filename: str, dataset_id: str | None = None) -> DocumentIndexResult:
        """Индексирует PDF. Повторная загрузка того же содержимого не индексируется заново.

        Идентификатор документа — хеш содержимого, поэтому переименованный файл тоже считается дублем.
        Эмбеддинги считаются до записи: при ошибке API в индексе не остаётся половины документа.
        """
        parsed = parse_pdf(data, filename)  # PdfError, если файл нельзя прочитать
        existing = self._docs.get(where={"document_id": parsed.document_id}, include=["metadatas"])
        if existing["ids"]:
            expected = existing["metadatas"][0]["chunk_count"]
            if len(existing["ids"]) == expected:
                first = existing["metadatas"][0]
                note = ""
                if dataset_id and (first.get("dataset_id") or "") != dataset_id:
                    note = " Привязка к датасету не изменена: документ уже привязан иначе."
                return DocumentIndexResult(
                    status="already_indexed",
                    document_id=parsed.document_id,
                    filename=first["filename"],
                    chunk_count=expected,
                    dataset_id=first.get("dataset_id") or None,
                    message=f"Документ уже в базе знаний (файл «{first['filename']}»), повторная индексация не нужна.{note}",
                )
            self._docs.delete(where={"document_id": parsed.document_id})  # недописанная индексация — начинаем заново

        chunks = chunk_document(parsed, max_chars=self.max_chars, overlap=self.overlap, dataset_id=dataset_id)
        if not chunks:
            raise PdfError("В PDF не найдено текста для индексации.")
        vectors = self.embedder.embed([c.embedding_text for c in chunks])

        indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._docs.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[self._chunk_metadata(c, len(chunks), parsed.page_count, indexed_at) for c in chunks],
        )
        return DocumentIndexResult(
            status="indexed",
            document_id=parsed.document_id,
            filename=filename,
            chunk_count=len(chunks),
            dataset_id=dataset_id,
            message=f"Документ проиндексирован: {len(chunks)} фрагментов.",
        )

    @staticmethod
    def _chunk_metadata(chunk: DocumentChunk, count: int, pages: int, indexed_at: str) -> dict:
        return {
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page": chunk.page,
            "page_end": chunk.page_end,
            "section": chunk.section or "",
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "chunk_count": count,
            "page_count": pages,
            "dataset_id": chunk.dataset_id or "",
            "indexed_at": indexed_at,
        }

    def list_documents(self) -> list[DocumentInfo]:
        records = self._docs.get(include=["metadatas"])
        grouped: dict[str, list[dict]] = defaultdict(list)
        for meta in records["metadatas"]:
            grouped[meta["document_id"]].append(meta)
        documents = []
        for document_id, metas in grouped.items():
            metas.sort(key=lambda m: m["chunk_index"])
            sections = list(dict.fromkeys(m["section"] for m in metas if m["section"]))
            documents.append(
                DocumentInfo(
                    document_id=document_id,
                    filename=metas[0]["filename"],
                    page_count=metas[0]["page_count"],
                    chunk_count=len(metas),
                    sections=sections,
                    dataset_id=metas[0]["dataset_id"] or None,
                    indexed_at=metas[0]["indexed_at"],
                )
            )
        return sorted(documents, key=lambda d: d.indexed_at, reverse=True)

    def delete_document(self, document_id: str) -> int:
        found = self._docs.get(where={"document_id": document_id}, include=[])["ids"]
        if found:
            self._docs.delete(ids=found)
        return len(found)

    # ------------------------------------------------------------------ датасеты
    def index_dataset_profile(self, profile: DatasetProfile) -> DatasetIndexResult:
        """Индексирует описание датасета и его колонок. Повторный вызов с тем же профилем ничего не меняет."""
        records = build_dataset_records(profile)
        digest = content_hash(records)
        existing = self._datasets.get(where={"dataset_id": profile.dataset_id}, include=["metadatas"])
        if existing["ids"] and len(existing["ids"]) == len(records) and all(
            m.get("content_hash") == digest for m in existing["metadatas"]
        ):
            return DatasetIndexResult(
                status="unchanged", dataset_id=profile.dataset_id, record_count=len(records),
                message="Описание датасета уже в индексе.",
            )

        vectors = self.embedder.embed([r.text for r in records])  # до удаления: при ошибке старый индекс сохраняется
        if existing["ids"]:
            self._datasets.delete(where={"dataset_id": profile.dataset_id})
        self._datasets.add(
            ids=[r.record_id for r in records],
            embeddings=vectors,
            documents=[r.text for r in records],
            metadatas=[{**r.metadata, "content_hash": digest} for r in records],
        )
        return DatasetIndexResult(
            status="indexed", dataset_id=profile.dataset_id, record_count=len(records),
            message=f"Проиндексировано записей: {len(records)} (датасет и колонки).",
        )

    def indexed_dataset_ids(self) -> set[str]:
        return {m["dataset_id"] for m in self._datasets.get(include=["metadatas"])["metadatas"]}

    def delete_dataset(self, dataset_id: str) -> int:
        found = self._datasets.get(where={"dataset_id": dataset_id}, include=[])["ids"]
        if found:
            self._datasets.delete(ids=found)
        return len(found)

    # ------------------------------------------------------------------ поиск
    def search_business_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        dataset_id: str | None = None,
        document_id: str | None = None,
        include_global: bool = True,
    ) -> ContextSearchResult:
        """Top-k фрагментов документов с источниками.

        dataset_id: документы этого датасета (и, если include_global, документы без привязки).
        """
        provider = self.embedder.name
        query = (query or "").strip()
        if not query:
            return ContextSearchResult(query=query, provider=provider, message="Пустой запрос.")

        where = self._where(document_id=document_id, dataset_id=dataset_id, include_global=include_global)
        raw = self._query(self._docs, query, top_k, where)
        hits = []
        for rank, (text, meta, distance) in enumerate(zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]), 1):
            score = round(1.0 - distance, 4)
            hits.append(
                ContextHit(
                    rank=rank, score=score, relevant=score >= self.embedder.min_score, text=text,
                    section=meta["section"] or None, page=meta["page"], source=self._document_source(meta),
                )
            )
        found = any(h.relevant for h in hits)
        message = "" if found else "В базе знаний нет достаточно релевантных фрагментов: ответ нельзя опирать на документы."
        return ContextSearchResult(query=query, provider=provider, hits=hits, found=found, message=message)

    def search_dataset_metadata(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        dataset_id: str | None = None,
        entry_type: str | None = None,
    ) -> MetadataSearchResult:
        """Семантический поиск датасета или колонки по описанию (entry_type: 'dataset' | 'column')."""
        provider = self.embedder.name
        query = (query or "").strip()
        if not query:
            return MetadataSearchResult(query=query, provider=provider, message="Пустой запрос.")

        clauses = [c for c in ({"dataset_id": dataset_id} if dataset_id else None, {"entry_type": entry_type} if entry_type else None) if c]
        where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
        raw = self._query(self._datasets, query, top_k, where)
        hits = []
        for rank, (chunk_id, text, meta, distance) in enumerate(
            zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]), 1
        ):
            score = round(1.0 - distance, 4)
            hits.append(
                MetadataHit(
                    rank=rank, score=score, relevant=score >= self.embedder.min_score,
                    entry_type=meta["entry_type"], dataset_id=meta["dataset_id"], filename=meta["filename"],
                    column_name=meta.get("column_name"), column_kind=meta.get("column_kind"),
                    aliases=[a for a in (meta.get("aliases") or "").split("|") if a], text=text,
                    source=self._dataset_source(meta, raw_id=chunk_id),
                )
            )
        found = any(h.relevant for h in hits)
        return MetadataSearchResult(
            query=query, provider=provider, hits=hits, found=found,
            message="" if found else "Подходящих датасетов или колонок не найдено.",
        )

    def resolve_column(self, query: str, *, dataset_id: str | None = None, top_k: int = DEFAULT_TOP_K) -> ColumnResolution:
        """Сопоставляет фразу («количество операций») с колонкой.

        Оценка = близость эмбеддингов + бонус за точное совпадение запроса с синонимом или названием
        колонки. Если лучшие кандидаты почти равны (или колонки из разных датасетов), статус «ambiguous»:
        интерфейс должен попросить пользователя выбрать. Найденное соответствие — предположение
        системы, пока пользователь его не подтвердил.
        """
        result = self.search_dataset_metadata(query, top_k * 3, dataset_id=dataset_id, entry_type="column")
        phrase = _phrase(query)
        ranked = []
        for hit in result.hits:
            alias = next((a for a in [*hit.aliases, hit.column_name or ""] if a and _phrase(a) == phrase), None)
            if alias:
                hit = hit.model_copy(update={
                    "raw_score": hit.score, "score": round(min(1.0, hit.score + EXACT_MATCH_BOOST), 4),
                    "matched_alias": alias, "relevant": True,
                })
            ranked.append(hit)
        ranked.sort(key=lambda h: h.score, reverse=True)
        relevant = [h for h in ranked if h.relevant]

        if not relevant:
            return ColumnResolution(
                query=query, status="not_found", candidates=ranked[:top_k],
                message="Не удалось сопоставить запрос ни с одной колонкой. Уточните формулировку.",
            )
        best = relevant[0]
        close = [h for h in relevant if best.score - h.score < self.embedder.ambiguity_margin][:top_k]
        if len(close) > 1:
            names = ", ".join(f"«{h.column_name}»" for h in close)
            return ColumnResolution(
                query=query, status="ambiguous", candidates=close,
                message=f"Запрос подходит нескольким колонкам: {names}. Подтвердите, какую использовать.",
            )
        return ColumnResolution(
            query=query, status="confident", chosen=best, candidates=[best],
            message=f"Запрос сопоставлен с колонкой «{best.column_name}» (предположение системы, не подтверждено пользователем).",
        )

    @staticmethod
    def confirm_column(resolution: ColumnResolution, column_name: str, dataset_id: str | None = None) -> ColumnResolution:
        """Фиксирует выбор пользователя среди кандидатов."""
        for hit in resolution.candidates:
            if hit.column_name == column_name and (dataset_id is None or hit.dataset_id == dataset_id):
                return resolution.model_copy(
                    update={
                        "status": "confirmed", "chosen": hit, "confirmed_by_user": True,
                        "message": f"Пользователь подтвердил колонку «{column_name}».",
                    }
                )
        raise ValueError(f"Колонка «{column_name}» не входит в число кандидатов.")

    # ------------------------------------------------------------------ источники и статистика
    def get_source_reference(self, chunk_id: str) -> SourceReference:
        """Источник по chunk_id (фрагмент документа или запись метаданных датасета)."""
        found = self._docs.get(ids=[chunk_id], include=["metadatas"])
        if found["ids"]:
            return self._document_source(found["metadatas"][0])
        found = self._datasets.get(ids=[chunk_id], include=["metadatas"])
        if found["ids"]:
            return self._dataset_source(found["metadatas"][0], raw_id=chunk_id)
        raise SourceNotFoundError(f"Фрагмент {chunk_id!r} не найден в индексе.")

    def get_chunk_text(self, chunk_id: str) -> str:
        for collection in (self._docs, self._datasets):
            found = collection.get(ids=[chunk_id], include=["documents"])
            if found["ids"]:
                return found["documents"][0]
        raise SourceNotFoundError(f"Фрагмент {chunk_id!r} не найден в индексе.")

    def stats(self) -> dict:
        docs = self.list_documents()
        return {
            "provider": self.embedder.name,
            "documents": len(docs),
            "document_chunks": self._docs.count(),
            "datasets": len(self.indexed_dataset_ids()),
            "dataset_records": self._datasets.count(),
        }

    # ------------------------------------------------------------------ внутреннее
    def _query(self, collection, query: str, top_k: int, where: dict | None) -> dict:
        matching = collection.get(where=where, include=[])["ids"] if where else collection.get(include=[])["ids"]
        if not matching:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        vector = self.embedder.embed([query])[0]
        return collection.query(
            query_embeddings=[vector],
            n_results=min(max(top_k, 1), len(matching)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    @staticmethod
    def _where(*, document_id: str | None, dataset_id: str | None, include_global: bool) -> dict | None:
        clauses: list[dict] = []
        if document_id:
            clauses.append({"document_id": document_id})
        if dataset_id:
            scope = [{"dataset_id": dataset_id}] + ([{"dataset_id": ""}] if include_global else [])
            clauses.append(scope[0] if len(scope) == 1 else {"$or": scope})
        return None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _document_source(meta: dict) -> SourceReference:
        return SourceReference(
            kind=SourceKind.DOCUMENT, chunk_id=meta["chunk_id"], filename=meta["filename"],
            document_id=meta["document_id"], page=meta["page"], page_end=meta["page_end"],
            section=meta["section"] or None, dataset_id=meta["dataset_id"] or None,
        )

    @staticmethod
    def _dataset_source(meta: dict, raw_id: str) -> SourceReference:
        column = meta.get("column_name")
        return SourceReference(
            kind=SourceKind.DATASET, chunk_id=raw_id, filename=meta["filename"],
            dataset_id=meta["dataset_id"], column_name=column,
        )
