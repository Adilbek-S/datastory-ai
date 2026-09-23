"""Pydantic-модели RAG: источники, фрагменты, результаты поиска и индексации."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from datastory.models import EvidenceType


class SourceKind(str, Enum):
    DOCUMENT = "document"
    DATASET = "dataset"


class SourceReference(BaseModel):
    """Ссылка на источник найденного фрагмента (для цитирования и проверки)."""

    kind: SourceKind
    chunk_id: str
    filename: str
    document_id: str | None = None
    page: int | None = None
    page_end: int | None = None
    section: str | None = None
    dataset_id: str | None = None
    column_name: str | None = None

    @property
    def citation(self) -> str:
        if self.kind is SourceKind.DOCUMENT:
            pages = f"стр. {self.page}" if self.page_end in (None, self.page) else f"стр. {self.page}–{self.page_end}"
            section = f", раздел «{self.section}»" if self.section else ""
            return f"{self.filename}, {pages}{section}"
        column = f", колонка «{self.column_name}»" if self.column_name else ""
        return f"датасет «{self.filename}» ({self.dataset_id}){column}"


# --------------------------------------------------------------------------- PDF
class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page: int
    page_end: int
    section: str | None = None
    text: str
    dataset_id: str | None = None

    @property
    def embedding_text(self) -> str:
        """Текст для эмбеддинга: заголовок раздела помогает найти фрагмент по теме."""
        return f"{self.section}\n{self.text}" if self.section else self.text


class Paragraph(BaseModel):
    page: int
    text: str


class DocumentSection(BaseModel):
    title: str | None
    paragraphs: list[Paragraph] = Field(default_factory=list)

    @property
    def page_start(self) -> int:
        return self.paragraphs[0].page if self.paragraphs else 1


class ParsedDocument(BaseModel):
    document_id: str
    filename: str
    page_count: int
    sections: list[DocumentSection]

    @property
    def section_titles(self) -> list[str]:
        return [s.title for s in self.sections if s.title]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    sections: list[str] = Field(default_factory=list)
    dataset_id: str | None = None
    indexed_at: str = ""


class DocumentIndexResult(BaseModel):
    status: Literal["indexed", "already_indexed"]
    document_id: str
    filename: str
    chunk_count: int
    dataset_id: str | None = None
    message: str = ""


class DatasetIndexResult(BaseModel):
    status: Literal["indexed", "unchanged"]
    dataset_id: str
    record_count: int
    message: str = ""


# --------------------------------------------------------------------------- поиск
class ContextHit(BaseModel):
    """Фрагмент документа, найденный по смыслу. Это факт из документа: у него всегда есть источник."""

    rank: int
    score: float
    relevant: bool
    text: str
    section: str | None = None
    page: int
    source: SourceReference
    evidence_type: EvidenceType = EvidenceType.DOCUMENT_FACT


class ContextSearchResult(BaseModel):
    query: str
    provider: str
    hits: list[ContextHit] = Field(default_factory=list)
    found: bool = False  # есть ли хотя бы один достаточно релевантный фрагмент
    message: str = ""

    @property
    def relevant_hits(self) -> list[ContextHit]:
        return [h for h in self.hits if h.relevant]


class MetadataHit(BaseModel):
    """Найденное описание датасета или колонки (метаданные, а не строки таблицы)."""

    rank: int
    score: float
    relevant: bool
    entry_type: Literal["dataset", "column"]
    dataset_id: str
    filename: str
    column_name: str | None = None
    column_kind: str | None = None
    aliases: list[str] = Field(default_factory=list)  # предполагаемые синонимы колонки
    matched_alias: str | None = None  # точное совпадение запроса с синонимом или названием колонки
    raw_score: float | None = None  # близость эмбеддингов до бонуса за точное совпадение
    text: str
    source: SourceReference


class MetadataSearchResult(BaseModel):
    query: str
    provider: str
    hits: list[MetadataHit] = Field(default_factory=list)
    found: bool = False
    message: str = ""


class ColumnResolution(BaseModel):
    """Итог сопоставления фразы пользователя с колонкой.

    Соответствие — предположение системы, пока пользователь его не подтвердил.
    """

    query: str
    status: Literal["confident", "ambiguous", "not_found", "confirmed"]
    chosen: MetadataHit | None = None
    candidates: list[MetadataHit] = Field(default_factory=list)
    confirmed_by_user: bool = False
    message: str = ""

    @property
    def needs_confirmation(self) -> bool:
        return self.status == "ambiguous"
