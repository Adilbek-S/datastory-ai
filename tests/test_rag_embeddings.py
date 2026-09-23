"""Провайдеры эмбеддингов: OpenAI (на подменённом клиенте, без сети) и офлайн-режим."""
import math
from types import SimpleNamespace

import openai
import pytest

from datastory.config import Settings
from datastory.rag.embeddings import (
    EmbeddingError,
    OfflineHashEmbeddings,
    OpenAIEmbeddings,
    get_embedding_provider,
)

try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def response(status: int) -> "httpx.Response":
    return httpx.Response(status, request=REQUEST)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


class FakeClient:
    """Подменяет OpenAI(): каждый вызов create() берёт следующее поведение (исключение или успех)."""

    def __init__(self, behaviours=None, shuffle=False):
        self.behaviours = list(behaviours or [])
        self.calls: list[dict] = []
        self.shuffle = shuffle
        self.embeddings = self

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": list(input)})
        behaviour = self.behaviours.pop(0) if self.behaviours else "ok"
        if isinstance(behaviour, Exception):
            raise behaviour
        items = [SimpleNamespace(index=i, embedding=[float(len(text)), 1.0]) for i, text in enumerate(input)]
        if behaviour == "incomplete":
            items = items[:-1]
        if self.shuffle:
            items.reverse()
        return SimpleNamespace(data=items)


def provider(client, **kwargs) -> OpenAIEmbeddings:
    sleeps: list[float] = []
    instance = OpenAIEmbeddings("sk-test", client=client, sleep=sleeps.append, **kwargs)
    instance.sleeps = sleeps
    return instance


# ------------------------------------------------------------------ OpenAI
def test_openai_uses_text_embedding_3_small_by_default():
    client = FakeClient()
    embedder = provider(client)
    assert embedder.name == "openai-text-embedding-3-small" and embedder.is_semantic
    embedder.embed(["один", "два"])
    assert client.calls == [{"model": "text-embedding-3-small", "input": ["один", "два"]}]


def test_openai_batches_requests_and_keeps_order():
    client = FakeClient()
    vectors = provider(client, batch_size=2).embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [len(c["input"]) for c in client.calls] == [2, 2, 1]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_openai_reorders_response_by_index():
    vectors = provider(FakeClient(shuffle=True)).embed(["a", "bb", "ccc"])
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


def test_openai_truncates_long_texts_and_replaces_empty():
    client = FakeClient()
    embedder = provider(client)
    embedder.embed(["x" * 20_000, ""])
    sent = client.calls[0]["input"]
    assert len(sent[0]) == OpenAIEmbeddings.MAX_CHARS and sent[1] == " "


def test_openai_retries_on_rate_limit_with_backoff():
    client = FakeClient([openai.RateLimitError("limit", response=response(429), body=None), "ok"])
    embedder = provider(client)
    assert len(embedder.embed(["текст"])) == 1
    assert len(client.calls) == 2 and embedder.sleeps == [1]


def test_openai_gives_up_after_max_retries():
    errors = [openai.APIConnectionError(request=REQUEST) for _ in range(5)]
    client = FakeClient(errors)
    embedder = provider(client, max_retries=2)
    with pytest.raises(EmbeddingError, match="недоступен"):
        embedder.embed(["текст"])
    assert len(client.calls) == 3 and embedder.sleeps == [1, 2]


def test_openai_bad_key_fails_fast_with_plain_message():
    client = FakeClient([openai.AuthenticationError("bad key", response=response(401), body=None)])
    embedder = provider(client)
    with pytest.raises(EmbeddingError, match="ключ"):
        embedder.embed(["текст"])
    assert len(client.calls) == 1 and embedder.sleeps == []


def test_openai_other_api_errors_are_wrapped():
    client = FakeClient([openai.BadRequestError("bad input", response=response(400), body=None)])
    with pytest.raises(EmbeddingError, match="Ошибка OpenAI"):
        provider(client).embed(["текст"])


def test_openai_incomplete_response_is_an_error():
    with pytest.raises(EmbeddingError, match="неполный"):
        provider(FakeClient(["incomplete"])).embed(["a", "b"])


def test_openai_requires_key():
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAIEmbeddings("   ")


# ------------------------------------------------------------------ офлайн
def test_offline_vectors_are_deterministic_and_normalized():
    a, b = OfflineHashEmbeddings(), OfflineHashEmbeddings()
    v1, v2 = a.embed(["Формула Success Rate"])[0], b.embed(["Формула Success Rate"])[0]
    assert v1 == v2 and len(v1) == OfflineHashEmbeddings.DIM
    assert math.sqrt(sum(x * x for x in v1)) == pytest.approx(1.0)
    assert not a.is_semantic


def test_offline_similar_texts_are_closer_than_unrelated():
    embedder = OfflineHashEmbeddings()
    query, close, far = embedder.embed(["количество транзакций", "число транзакций за месяц", "рецепт борща со сметаной"])
    assert cosine(query, close) > cosine(query, far) + 0.2


def test_offline_handles_russian_word_forms():
    embedder = OfflineHashEmbeddings()
    a, b, c = embedder.embed(["операций", "операции", "погода"])
    assert cosine(a, b) > 0.5 > cosine(a, c)


def test_offline_empty_text_does_not_crash():
    assert OfflineHashEmbeddings().embed([""])[0] == [0.0] * OfflineHashEmbeddings.DIM


# ------------------------------------------------------------------ выбор провайдера
def test_provider_selection_modes():
    with_key = Settings(openai_api_key="sk-test", embedding_provider="auto", _env_file=None)
    without_key = Settings(openai_api_key="", embedding_provider="auto", _env_file=None)
    assert get_embedding_provider(with_key).name == "openai-text-embedding-3-small"
    assert get_embedding_provider(without_key).name == "offline-hash-v1"

    assert get_embedding_provider(Settings(openai_api_key="sk-test", embedding_provider="offline", _env_file=None)).name == "offline-hash-v1"
    assert get_embedding_provider(Settings(openai_api_key="sk-test", embedding_provider=" OpenAI ", _env_file=None)).is_semantic


def test_forced_openai_without_key_and_unknown_mode():
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        get_embedding_provider(Settings(openai_api_key="", embedding_provider="openai", _env_file=None))
    with pytest.raises(EmbeddingError, match="EMBEDDING_PROVIDER"):
        get_embedding_provider(Settings(embedding_provider="magic", _env_file=None))


def test_default_environment_in_tests_never_uses_real_key():
    """conftest переопределяет .env: тесты не должны тратить запросы OpenAI."""
    assert get_embedding_provider().name == "offline-hash-v1"
