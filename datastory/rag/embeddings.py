"""Провайдеры эмбеддингов.

* OpenAIEmbeddings — text-embedding-3-small (основной режим).
* OfflineHashEmbeddings — детерминированные лексические векторы без сети. Нужны для тестов и как
  явно помеченный офлайн-режим, когда ключ OpenAI не задан. Это НЕ семантическая модель.

Векторы разных провайдеров несовместимы, поэтому хранилище держит отдельные коллекции
для каждого провайдера (см. KnowledgeBase).
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from abc import ABC, abstractmethod
from typing import Callable

from datastory.config import Settings, get_settings


class EmbeddingError(RuntimeError):
    """Ошибка получения эмбеддингов; user_message показывается пользователю."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class EmbeddingProvider(ABC):
    name: str
    is_semantic: bool
    # Пороги косинусной близости; подбираются под конкретную модель.
    min_score: float
    ambiguity_margin: float

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Векторы для списка текстов (в том же порядке)."""


# --------------------------------------------------------------------------- OpenAI
class OpenAIEmbeddings(EmbeddingProvider):
    is_semantic = True
    min_score = 0.30  # по замерам на демо-данных: нерелевантное <=0.18, релевантное >=0.33
    ambiguity_margin = 0.05
    MAX_CHARS = 6000  # запас до лимита модели по токенам

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        *,
        client=None,
        batch_size: int = 64,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not (api_key or "").strip() and client is None:
            raise EmbeddingError("Ключ OpenAI не задан. Укажите OPENAI_API_KEY в файле .env.")
        self.model = model
        self.name = f"openai-{model}"
        self.batch_size, self.max_retries, self._sleep = batch_size, max_retries, sleep
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t[: self.MAX_CHARS] or " " for t in texts[start : start + self.batch_size]]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        import openai

        retryable = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
            except retryable as exc:
                if attempt == self.max_retries:
                    raise EmbeddingError(
                        "Сервис OpenAI временно недоступен или превышен лимит запросов. Повторите позже."
                    ) from exc
                self._sleep(2**attempt)
                continue
            except openai.AuthenticationError as exc:
                raise EmbeddingError("OpenAI отклонил ключ API. Проверьте OPENAI_API_KEY в файле .env.") from exc
            except openai.APIError as exc:
                raise EmbeddingError(f"Ошибка OpenAI при получении эмбеддингов: {exc}") from exc

            items = sorted(response.data, key=lambda d: d.index)
            if len(items) != len(batch):
                raise EmbeddingError("OpenAI вернул неполный ответ. Повторите попытку.")
            return [list(item.embedding) for item in items]
        raise AssertionError("unreachable")


# --------------------------------------------------------------------------- офлайн
_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")
_STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "о", "об", "от", "для", "из", "за", "не", "что", "как", "это",
    "the", "of", "a", "an", "is", "are", "to", "in", "on", "for", "and", "or", "by", "what",
}


class OfflineHashEmbeddings(EmbeddingProvider):
    """Лексические векторы: слова, основы и символьные триграммы хешируются в фиксированное пространство."""

    is_semantic = False
    name = "offline-hash-v1"
    min_score = 0.20
    ambiguity_margin = 0.06
    DIM = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _features(self, text: str):
        for token in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
            if token in _STOPWORDS:
                continue
            yield token, 1.0
            if len(token) > 5:
                yield "~" + token[:5], 0.8  # общая основа: «транзакций» ~ «транзакции»
            if len(token) > 3:
                padded = f"#{token}#"
                for i in range(len(padded) - 2):
                    yield padded[i : i + 3], 0.3

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIM
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


# --------------------------------------------------------------------------- выбор провайдера
def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """auto: OpenAI при заданном ключе, иначе офлайн; openai / offline — принудительно."""
    settings = settings or get_settings()
    mode = settings.embedding_provider.strip().lower()
    if mode not in {"auto", "openai", "offline"}:
        raise EmbeddingError(f"Неизвестный EMBEDDING_PROVIDER: {mode!r} (допустимо: auto, openai, offline).")
    if mode == "offline" or (mode == "auto" and not settings.has_openai_key):
        return OfflineHashEmbeddings()
    return OpenAIEmbeddings(settings.openai_api_key, settings.embedding_model)
