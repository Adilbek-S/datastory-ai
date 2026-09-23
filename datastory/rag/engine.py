"""RAG Engine: база знаний на ChromaDB.

Реализованы разбиение на фрагменты и работа с хранилищем. Эмбеддинги
(text-embedding-3-small) и поиск будут подключены на следующем этапе.
"""
from __future__ import annotations

import chromadb

from datastory.config import get_settings
from datastory.models import KnowledgeChunk

COLLECTION_NAME = "business_knowledge"


def split_text(chunk: KnowledgeChunk, size: int = 800, overlap: int = 100) -> list[KnowledgeChunk]:
    """Режет страницу на перекрывающиеся фрагменты фиксированной длины."""
    if size <= overlap:
        raise ValueError("size должен быть больше overlap")
    text, step = chunk.text, size - overlap
    return [
        KnowledgeChunk(source=chunk.source, page=chunk.page, text=text[start : start + size])
        for start in range(0, len(text), step)
    ]


class RagEngine:
    def __init__(self) -> None:
        settings = get_settings()
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        # embedding_function=None: векторы будут передаваться явно (OpenAI Embeddings).
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME, embedding_function=None
        )

    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        raise NotImplementedError("Индексация появится на следующем этапе.")

    def search(self, query: str, top_k: int = 4) -> list[KnowledgeChunk]:
        raise NotImplementedError("Семантический поиск появится на следующем этапе.")
