"""Промпты чата. Методика (Skill) подгружается из SKILL.md, здесь только формат ответа и данные вопроса."""
from __future__ import annotations

from typing import TYPE_CHECKING

from datastory.analytics.metrics import get_metric
from datastory.skills import Skill
from datastory.workflow.methodology import methodology_prompt
from datastory.workflow.models import AnalysisResult

if TYPE_CHECKING:
    from datastory.chat.evidence import Bundle

CLASSIFY_SYSTEM = (
    "Ты определяешь тип вопроса пользователя по готовому дашборду. Типы:\n"
    "- metric_definition: что означает показатель, как он считается;\n"
    "- metric_change: как изменился показатель во времени;\n"
    "- extreme_period: какой месяц или период имеет максимальное или минимальное значение;\n"
    "- top_category: какая категория (канал) лидирует по количеству операций или другому показателю;\n"
    "- notable_changes: какие изменения заслуживают внимания;\n"
    "- context_events: какие события упоминаются в сопроводительной документации;\n"
    "- cause_question: почему показатель изменился (причины по данным не определяются);\n"
    "- unsupported: всё остальное (прогнозы, произвольные вычисления, код, изменение данных, вопросы не о дашборде).\n"
    "В metric верни идентификатор показателя из списка доступных, если вопрос называет показатель, иначе null."
)
CHAT_TASK = (
    "Ты отвечаешь на вопрос пользователя по готовому дашборду и возвращаешь структурный ответ: текст (answer, 1–4 предложения), "
    "id использованных числовых доказательств (evidence_ids), ограничение интерпретации (limitation). "
    "В evidence_ids указывай только id числовых доказательств из таблицы, а chunk_id фрагментов документов — только в citations вместе с дословной цитатой. "
    "Не вставляй id в текст ответа. Если показателей несколько, ответь по каждому. Отвечай только на заданный вопрос."
)


INTENT_HINTS = {
    "metric_definition": "Объясни, что означает показатель, по найденному определению и приведи формулу расчёта; определения не придумывай.",
    "context_events": "Перечисли только события, названные в документах, с цитатой. Не делай выводов о динамике показателей.",
    "cause_question": "Скажи, что причина по данным не определяется; назови сам факт изменения по доказательствам и событие из документа как совпадающее по времени.",
    "extreme_period": "Назови период максимума и минимума по каждому показателю.",
    "top_category": "Назови лидирующую категорию, её значение и долю, затем остальные.",
}


def _metric_line(metric: str) -> str:
    definition = get_metric(metric)
    unit = definition.unit or "без единицы"
    return f"{metric} — «{definition.label}», формула {definition.formula}, единица {unit}"


def classify_prompt(question: str, result: AnalysisResult) -> str:
    metrics = "\n".join(f"- {m}: {get_metric(m).label}" for m in result.summary.available_metrics)
    return f"Вопрос: «{question}»\n\nДоступные показатели:\n{metrics}"


def chat_system(skill: Skill) -> str:
    return "\n\n".join([CHAT_TASK, methodology_prompt(skill, "chat")])


def chat_prompt(question: str, bundle: "Bundle", feedback: list[str] | None = None) -> str:
    parts = [f"Вопрос: «{question}» (тип: {bundle.intent})."]
    if bundle.intent in INTENT_HINTS:
        parts.append(INTENT_HINTS[bundle.intent])
    if bundle.metrics:
        parts.append("Показатели: " + "; ".join(_metric_line(m) for m in bundle.metrics) + ".")
    if bundle.labels and bundle.inputs:
        parts.append("Периоды и категории в данных: " + ", ".join(dict.fromkeys(bundle.labels)) + ". Не упоминай других.")
    if bundle.intent == "metric_definition":
        lines = []
        for metric, fragment in bundle.definitions.items():
            lines.append(
                f"- {metric}: найдено [{fragment.chunk_id}] ({fragment.citation}): «{fragment.text}»" if fragment
                else f"- {metric}: в документации НЕ найдено; используй только формулу расчёта"
            )
        parts.append("Определения из документации:\n" + "\n".join(lines))
    events = [f for f in bundle.events]
    if bundle.intent in ("context_events", "cause_question"):
        parts.append("Фрагменты документации:\n" + ("\n".join(f"[{f.chunk_id}] ({f.citation}): «{f.text}»" for f in events) or "не найдены"))
    if bundle.facts:
        parts.append("Числовые доказательства (id | описание | значение) — единственный допустимый источник чисел:\n" + "\n".join(f"{f.id} | {f.label} | {f.formatted}" for f in bundle.facts))
    if feedback:
        parts.append("Предыдущий ответ отклонён проверкой, исправьте:\n" + "\n".join(f"- {v}" for v in feedback))
    return "\n\n".join(parts)
