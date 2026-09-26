"""Модели чата по дашборду: тип вопроса, структурные ответы LLM, ответ пользователю."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from datastory.models import NumericEvidence

ChatIntent = Literal[
    "metric_definition",  # что означает показатель
    "metric_change",  # как изменился показатель
    "extreme_period",  # какой период максимальный / минимальный
    "top_category",  # какая категория (канал) лидирует
    "notable_changes",  # какие изменения заслуживают внимания
    "context_events",  # какие события упоминаются в документации
    "cause_question",  # почему изменился показатель (причины по данным не определяются)
    "unsupported",  # вне возможностей MVP
]
SUPPORTED_QUESTIONS = (
    "Что означает показатель?",
    "Как изменился показатель?",
    "Какой месяц имеет максимальное значение?",
    "Какой канал имеет наибольшее количество операций?",
    "Какие изменения заслуживают внимания?",
    "Какие события упоминаются в сопроводительной документации?",
)
MAX_QUESTION = 500


class QuestionPlan(BaseModel):
    """Структурный ответ LLM на этапе classify_question."""

    intent: ChatIntent = Field(description="Тип вопроса; unsupported, если вопрос не подходит ни под один из типов")
    metric: str | None = Field(description="Идентификатор показателя из списка доступных или null, если показатель не назван")
    comment: str = Field(description="Одно предложение: как понят вопрос или почему он не поддерживается")


class Citation(BaseModel):
    chunk_id: str = Field(description="chunk_id фрагмента документа из запроса")
    quote: str = Field(description="Дословная цитата из этого фрагмента")


class ChatAnswerDraft(BaseModel):
    """Структурный ответ LLM на этапе compose_answer: текст и ссылки, но не числа как данные."""

    answer: str = Field(description="Ответ на вопрос, 1–4 предложения; числа только из числовых доказательств; id доказательств в текст не вставляй")
    evidence_ids: list[str] = Field(description="id числовых доказательств (например, success_rate.max), на которые опирается ответ; пустой список, если чисел нет")
    limitation: str | None = Field(description="Ограничение интерпретации или null")
    citations: list[Citation] = Field(description="Фрагменты документов, использованные в ответе, с дословными цитатами; пустой список, если документы не использованы")


class ChatAnswer(BaseModel):
    question: str
    intent: ChatIntent
    answer: str
    supported: bool = True
    metrics: list[str] = Field(default_factory=list)
    evidence: list[NumericEvidence] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list, description="Откуда числа: вызовы MCP или результаты дашборда")
    context_source: str | None = None
    limitation: str | None = None
    tools: list[str] = Field(default_factory=list, description="Что вызывалось для ответа: MCP calculate_metrics, RAG")
    generated_by: Literal["llm", "rules"] = "rules"
    classified_by: Literal["llm", "rules"] = "rules"
    violations: list[str] = Field(default_factory=list, description="Замечания проверки достоверности к отклонённым попыткам")
    error: bool = False
