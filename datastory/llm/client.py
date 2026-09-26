"""Обращения к языковой модели со структурным ответом (OpenAI через langchain-openai).

Модель всегда возвращает объект заданной Pydantic-схемы, а не свободный текст. Все сбои (нет ключа, сеть,
лимиты, невалидный ответ) превращаются в LLMError с понятным сообщением: воркфлоу переходит на детерминированные
правила и предупреждает пользователя, а не падает.
"""
from __future__ import annotations

import logging
from typing import Protocol, TypeVar

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from datastory.config import Settings, get_settings
from datastory.errors import LLMError, LLMUnavailableError

logger = logging.getLogger("datastory.llm")
T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    """Всё, что нужно воркфлоу от модели: ответ по схеме на пару «системная инструкция + запрос»."""

    name: str

    def generate(self, schema: type[T], *, system: str, user: str) -> T: ...


class OpenAIStructuredLLM:
    def __init__(self, api_key: str, model: str, *, timeout: float = 60.0, max_retries: int = 2, chat: ChatOpenAI | None = None):
        self.name = f"openai-{model}"
        self._chat = chat or ChatOpenAI(model=model, api_key=api_key, temperature=0, timeout=timeout, max_retries=max_retries)

    def generate(self, schema: type[T], *, system: str, user: str) -> T:
        try:
            answer = self._chat.with_structured_output(schema).invoke([SystemMessage(system), HumanMessage(user)])
        except openai.AuthenticationError:
            raise LLMError("OpenAI отклонил ключ API. Проверьте OPENAI_API_KEY в файле .env.") from None
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


def get_llm(settings: Settings | None = None) -> StructuredLLM:
    """Модель из настроек. Без ключа бросает LLMUnavailableError — вызывающий код переходит на правила."""
    settings = settings or get_settings()
    if not settings.has_openai_key:
        raise LLMUnavailableError("Ключ OpenAI не задан (OPENAI_API_KEY): используются детерминированные правила.")
    return OpenAIStructuredLLM(settings.openai_api_key, settings.openai_model)
