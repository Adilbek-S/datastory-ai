"""Исключения приложения. У каждого есть понятное пользователю сообщение (user_message)."""
from __future__ import annotations


class DataLoadError(ValueError):
    """Базовая ошибка загрузки данных."""

    code = "data_error"

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class UnsupportedFileError(DataLoadError):
    code = "unsupported_format"


class EmptyFileError(DataLoadError):
    code = "empty_file"


class CorruptFileError(DataLoadError):
    code = "corrupt_file"


class NoDataError(DataLoadError):
    code = "no_data"


class DatasetNotFoundError(LookupError):
    """Датасет с таким ID отсутствует в рабочем хранилище."""


class AnalyticsError(ValueError):
    """Некорректный запрос к аналитическим инструментам (метрика, group_by, фильтры, график).

    Сообщение написано для пользователя и для LLM-агента: в нём указано, что именно неверно и что допустимо.
    """

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class LLMError(RuntimeError):
    """Сбой обращения к языковой модели (сеть, ключ, лимиты, невалидный структурированный ответ)."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class LLMUnavailableError(LLMError):
    """Языковая модель не настроена (нет ключа OpenAI)."""


class WorkflowError(RuntimeError):
    """Некорректное обращение к workflow (неизвестная сессия, анализ не ждёт подтверждения)."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message
