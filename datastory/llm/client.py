"""Обращения к языковой модели со структурным ответом (OpenAI через langchain-openai).

Модель всегда возвращает объект заданной Pydantic-схемы, а не свободный текст. Все сбои (нет ключа, сеть,
лимиты, невалидный ответ) превращаются в LLMError с понятным сообщением: воркфлоу переходит на детерминированные
правила и предупреждает пользователя, а не падает.
"""
from __future__ import annotations

import base64
import logging
from contextlib import nullcontext
from typing import Protocol, TypeVar

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from datastory.config import Settings, get_settings
from datastory.errors import LLMError, LLMUnavailableError
from datastory.llm.usage import UsageHandler, UsageRecorder
from datastory.observability import trace_config, untraced

logger = logging.getLogger("datastory.llm")
T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    """Всё, что нужно воркфлоу от модели: ответ по схеме на пару «системная инструкция + запрос»."""

    name: str

    def generate(self, schema: type[T], *, system: str, user: str) -> T: ...


class VisionLLM(Protocol):
    """Модель, читающая изображение: ответ по схеме на «инструкция + запрос + картинка»."""

    name: str

    def generate_from_image(self, schema: type[T], *, system: str, user: str, image: bytes, mime: str) -> T: ...


LENGTH_LIMIT_MESSAGE = "Ответ модели достиг лимита max output tokens и обрезан."


class OpenAIStructuredLLM:
    """Структурные ответы OpenAI; та же модель принимает изображения (gpt-4o-mini Vision)."""

    def __init__(
        self, api_key: str, model: str, *, timeout: float = 60.0, max_retries: int = 2, chat: ChatOpenAI | None = None,
        usage: UsageRecorder | None = None, max_tokens: int | None = None, temperature: float = 0.0,
    ):
        self.name = f"openai-{model}"
        self.usage = usage  # если задан, токены каждого ответа складываются сюда
        self.temperature, self.max_tokens = temperature, max_tokens  # параметры генерации фиксируются явно (эксперименты записывают их в результат)
        self._chat = chat or ChatOpenAI(
            model=model, api_key=api_key, temperature=temperature, timeout=timeout, max_retries=max_retries, max_tokens=max_tokens
        )

    def generate(self, schema: type[T], *, system: str, user: str) -> T:
        return self._invoke(schema, [SystemMessage(system), HumanMessage(user)], traced=True)

    def generate_from_image(self, schema: type[T], *, system: str, user: str, image: bytes, mime: str) -> T:
        url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        content = [{"type": "text", "text": user}, {"type": "image_url", "image_url": {"url": url, "detail": "high"}}]
        # Изображение в LangSmith не отправляется: вызов исключён из трассировки, а запись о нём делает span распознавания.
        return self._invoke(schema, [SystemMessage(system), HumanMessage(content)], traced=False)

    def _invoke(self, schema: type[T], messages: list, traced: bool) -> T:
        config = trace_config(
            f"llm.{schema.__name__}", tags=("llm", "structured-output"),
            metadata={"ls_provider": "openai", "ls_model_name": self.name.removeprefix("openai-"), "schema": schema.__name__},
        )
        if self.usage is not None:
            config["callbacks"] = [UsageHandler(self.usage)]
        try:
            with nullcontext() if traced else untraced():
                answer = self._chat.with_structured_output(schema).invoke(messages, config=config)
        except openai.AuthenticationError:
            raise LLMError("OpenAI отклонил ключ API. Проверьте OPENAI_API_KEY в файле .env.") from None
        except openai.LengthFinishReasonError:
            raise LLMError(LENGTH_LIMIT_MESSAGE) from None
        except openai.RateLimitError:
            raise LLMError("Превышен лимит запросов OpenAI. Повторите позже.") from None
        except (openai.APITimeoutError, openai.APIConnectionError):
            raise LLMError("Не удалось связаться с OpenAI: проверьте подключение к сети.") from None
        except openai.APIError as exc:
            raise LLMError(f"Ошибка OpenAI: {getattr(exc, 'message', exc)}") from None
        except (OutputParserException, ValidationError, ValueError) as exc:
            logger.warning("Невалидный структурированный ответ: %s", exc)
            raise LLMError("Модель вернула ответ, не соответствующий ожидаемой структуре.") from None
        if answer is None:
            raise LLMError("Модель не вернула структурированный ответ.")
        return answer if isinstance(answer, schema) else schema.model_validate(answer)


def get_llm(
    settings: Settings | None = None, usage: UsageRecorder | None = None, max_tokens: int | None = None, temperature: float | None = None
) -> OpenAIStructuredLLM:
    """Модель и параметры генерации из настроек (LLM_TEMPERATURE, LLM_MAX_OUTPUT_TOKENS); max_tokens/temperature переопределяют их для экспериментов.

    Без ключа бросает LLMUnavailableError — вызывающий код переходит на правила.
    """
    settings = settings or get_settings()
    if not settings.has_openai_key:
        raise LLMUnavailableError("Ключ OpenAI не задан (OPENAI_API_KEY): используются детерминированные правила.")
    return OpenAIStructuredLLM(
        settings.openai_api_key, settings.openai_model, usage=usage,
        max_tokens=settings.llm_max_output_tokens if max_tokens is None else max_tokens,
        temperature=settings.llm_temperature if temperature is None else temperature,
    )


def get_vision_llm(settings: Settings | None = None) -> OpenAIStructuredLLM:
    """Модель для распознавания таблиц на изображениях. Без ключа бросает LLMUnavailableError."""
    settings = settings or get_settings()
    if not settings.has_openai_key:
        raise LLMUnavailableError("Для распознавания изображений нужен ключ OpenAI: задайте OPENAI_API_KEY в файле .env.")
    return OpenAIStructuredLLM(settings.openai_api_key, settings.vision_model, timeout=120.0)
