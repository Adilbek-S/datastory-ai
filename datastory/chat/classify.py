"""Разбор вопроса без LLM: тип вопроса и показатель по ключевым словам.

Используется, когда модель недоступна, а также для проверки того, что показатель, названный моделью, действительно
существует в датасете. Никакого кода из вопроса не выполняется: тип вопроса — один из фиксированного набора.
"""
from __future__ import annotations

import re

from datastory.chat.models import ChatIntent

# порядок важен: более специфичные типы проверяются раньше
_INTENT_PATTERNS: tuple[tuple[ChatIntent, str], ...] = (
    (
        "unsupported",
        r"прогноз|предскаж|спрогноз|что будет|в будущем|построй|нарисуй|\bкод\b|python|sql|скрипт|корреляц|регресс|медиан|"
        r"стандартн\w* отклон|с прошл\w* год|удали|измени данные|загрузи",
    ),
    ("cause_question", r"почему|из-за чего|по какой причине|причин[аыу]\b|что вызвал|чем вызван"),
    ("context_events", r"событи|документаци|упоминаются|упоминается|что было|инцидент|обновлен|регламент"),
    ("notable_changes", r"заслужива|вниман|аномал|существенн|важн|что интересн|выделя"),
    (
        "extreme_period",
        r"максимальн|минимальн|наибольш\w* значени|наименьш\w* значени|\bпик\b|лучш\w* месяц|худш\w* месяц|какой месяц|"
        r"в какой месяц|какой период|какой квартал",
    ),
    (
        "top_category",
        r"какой канал|какая категор|какой из каналов|по каналам|канал\w*.*(больш|наиболь|лидир|меньш|наименьш)|"
        r"(больш|наиболь|лидир|меньш|наименьш)\w*.*канал",
    ),
    ("metric_definition", r"что означает|что значит|что такое|определени|как считается|как рассчитывается|формула|что показывает"),
    ("metric_change", r"как изменил|изменени|динамик|\bрос\w*|вырос|снизил|упал|тренд|как менял"),
)
_METRIC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("failed_count", r"неуспешн|отклонен|отказ|ошибок|failed"),
    ("average_transaction_amount", r"средн\w* сумм|средн\w* чек|average"),
    ("transaction_volume", r"объ[её]м|оборот|volume|сумм\w* транзакц"),
    ("successful_count", r"количеств\w* успешн|числ\w* успешн|успешных транзакц|successful"),
    ("success_rate", r"успешност|success|доля успешн|процент успешн"),
    ("transaction_count", r"количеств|числ\w* (операц|транзакц)|операц|транзакц"),
)
_UNSUPPORTED_REASON = "Вопрос требует расчётов или действий, которых нет в текущем MVP (прогнозы, произвольные вычисления, код, изменение данных)."


# «в марте», «за январь», «2026-03»: расчёт с фильтром по периоду чат не выполняет
_PERIOD_FILTER = re.compile(
    r"\b(?:в|за|на|для|по)\s+(?:январ|феврал|март|апрел|ма[йяе]\b|июн|июл|август|сентябр|октябр|ноябр|декабр)|\b20\d\d-\d\d\b"
)
FILTER_SENSITIVE: tuple[ChatIntent, ...] = ("metric_change", "extreme_period", "top_category")
PERIOD_FILTER_REASON = "Чат отвечает по всему периоду данных: расчёты с фильтром по месяцу или периоду в текущем MVP не поддерживаются."


def has_period_filter(question: str) -> bool:
    return bool(_PERIOD_FILTER.search(_norm(question)))


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def classify_by_rules(question: str) -> tuple[ChatIntent, str]:
    """(тип вопроса, пояснение). Вопрос без узнаваемого типа — unsupported."""
    text = _norm(question)
    for intent, pattern in _INTENT_PATTERNS:
        if re.search(pattern, text):
            return intent, _UNSUPPORTED_REASON if intent == "unsupported" else ""
    return "unsupported", "Не удалось отнести вопрос ни к одному из поддерживаемых типов."


def detect_metric(question: str, available: list[str]) -> str | None:
    """Показатель, названный в вопросе, если он доступен в датасете."""
    text = _norm(question)
    for metric, pattern in _METRIC_PATTERNS:
        if metric in available and re.search(pattern, text):
            return metric
    return None
