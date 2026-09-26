"""Подмена Vision-модели в тестах: возвращает структурный ответ, как настоящая модель, и запоминает запрос."""
from datastory.errors import LLMError
from datastory.file_processing.image_loader import TableExtraction
from scripts import generate_demo_data as gen

COLUMNS = ["Month", "Transactions", "Successful", "Failed", "Amount_KZT", "Channel"]


def screenshot_extraction() -> TableExtraction:
    """То, что видно на transactions_screenshot.png: февраль и март, строки как текст."""
    rows = [[str(r[c]) for c in COLUMNS] for r in gen.build_rows() if r["Month"] in gen.SCREENSHOT_MONTHS]
    return TableExtraction(contains_table=True, columns=COLUMNS, rows=rows, unreadable_cells=[], comment="Таблица транзакций.")


class FakeVision:
    name = "fake-vision"

    def __init__(self, answer=None):
        self.answer = answer if answer is not None else screenshot_extraction()
        self.calls: list[dict] = []

    def generate_from_image(self, schema, *, system, user, image, mime):
        self.calls.append({"schema": schema.__name__, "system": system, "user": user, "image": image, "mime": mime})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def failing_vision(message="Превышен лимит запросов OpenAI. Повторите позже.") -> FakeVision:
    return FakeVision(LLMError(message))
