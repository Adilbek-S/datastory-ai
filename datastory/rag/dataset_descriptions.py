"""Семантические описания датасета для индекса (метаданные, а не строки таблицы).

Индексируются только: запись про датасет целиком и по одной записи на колонку. Строки Excel
не индексируются, а RAG не используется для точных расчётов — их выполняет Pandas.

Примеры значений берутся из краткого профиля для LLM (build_llm_profile), поэтому персональные
данные сюда не попадают. Синонимы колонок — предположения по названию (подписаны как таковые).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from datastory.models import ColumnKind, DatasetProfile
from datastory.profiler.llm_profile import LLMColumnSummary, build_llm_profile

KIND_RU = {
    ColumnKind.NUMERIC: "число",
    ColumnKind.CATEGORICAL: "категория",
    ColumnKind.DATETIME: "дата/время",
    ColumnKind.BOOLEAN: "да/нет",
    ColumnKind.TEXT: "текст",
}
ROLE_RU = {
    ColumnKind.NUMERIC: "показатель",
    ColumnKind.CATEGORICAL: "измерение",
    ColumnKind.BOOLEAN: "измерение",
    ColumnKind.DATETIME: "временное измерение",
    ColumnKind.TEXT: "текстовое поле",
}

# (регулярное выражение по названию колонки, предполагаемые синонимы). Правила «успешные»,
# «неуспешные» и «сумма» исключают общее правило «количество операций».
_GLOSSARY: list[tuple[str, list[str]]] = [
    (r"success|succeed|успеш", ["успешные транзакции", "успешные операции", "количество успешных операций", "успешно завершённые платежи"]),
    (r"fail|error|declin|reject|неуспеш|ошибк|отказ", ["неуспешные транзакции", "неуспешные операции", "количество ошибок", "отказы", "неудачные операции"]),
    (r"amount|sum|volume|revenue|turnover|сумм|объ[её]м|выруч|оборот", ["сумма", "объём транзакций", "объём платежей", "денежный объём", "оборот", "сумма платежей"]),
]
_COUNT_RULE = (r"transaction|trx|txn|operation|payment|транзакц|операц|платеж", ["количество транзакций", "количество операций", "число транзакций", "число операций"])
_EXTRA: list[tuple[str, list[str]]] = [
    (r"count|number|qty|quantity|количеств|число|кол-во", ["количество", "число"]),
    (r"month|date|day|year|period|week|дата|месяц|период|год|день|недел", ["месяц", "дата", "период", "временная ось"]),
    (r"channel|source|канал|источник", ["канал", "канал платежа", "источник"]),
    (r"region|city|country|регион|город|стран", ["регион", "город", "география"]),
    (r"rate|percent|share|pct|ratio|доля|процент|коэфф", ["доля", "процент", "коэффициент"]),
    (r"avg|average|mean|средн", ["среднее значение", "средняя сумма"]),
    (r"price|cost|цена|стоимост", ["цена", "стоимость"]),
    (r"customer|client|user|клиент|пользоват", ["клиент", "пользователь"]),
    (r"product|sku|item|товар|продукт", ["товар", "продукт"]),
    (r"status|state|статус", ["статус", "состояние"]),
    (r"type|category|тип|категори", ["тип", "категория"]),
    (r"kzt|тенге", ["валюта: тенге"]),
    (r"usd|доллар", ["валюта: доллары США"]),
    (r"eur|евро", ["валюта: евро"]),
    (r"rub|рубл", ["валюта: рубли"]),
]


def _normalize_name(name: str) -> str:
    """«AmountKZT» и «Amount_KZT» -> «amount kzt»."""
    spaced = re.sub(r"([a-zа-я0-9])([A-ZА-Я])", r"\1 \2", name)
    return re.sub(r"[^0-9a-zа-яё]+", " ", spaced.lower()).strip()


def guess_synonyms(column_name: str) -> list[str]:
    """Предполагаемые синонимы колонки по её названию (правила, без LLM)."""
    name = _normalize_name(column_name)
    phrases: list[str] = []
    specific = False
    for pattern, values in _GLOSSARY:
        if re.search(pattern, name):
            phrases += values
            specific = True
    if not specific and re.search(_COUNT_RULE[0], name):
        phrases += _COUNT_RULE[1]
    for pattern, values in _EXTRA:
        if re.search(pattern, name):
            phrases += values
    seen: set[str] = set()
    return [p for p in phrases if not (p in seen or seen.add(p))]


@dataclass
class IndexRecord:
    record_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _number(value: float | str) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.0f}".replace(",", " ") if abs(value) >= 1000 else f"{value:g}"
    return str(value)


def _column_text(title: str, summary: LLMColumnSummary, kind: ColumnKind) -> str:
    """Смысл колонки (название и предполагаемые синонимы) идёт первым: короткий фокусный текст лучше ищется."""
    if summary.note and "персональные" in summary.note:
        return (
            f"Колонка «{summary.name}» датасета «{title}». Тип: {KIND_RU[kind]}. "
            "Возможны персональные данные — значения не индексируются."
        )

    synonyms = guess_synonyms(summary.name)
    head = f"Колонка «{summary.name}»"
    if synonyms:
        head += " — предполагаемый смысл по названию: " + ", ".join(synonyms)
    lines = [head + ".", f"Тип: {KIND_RU[kind]}, роль: {ROLE_RU[kind]}. Датасет «{title}»."]

    stats = summary.stats
    if kind is ColumnKind.NUMERIC and "min" in stats:
        lines.append(f"Значения от {_number(stats['min'])} до {_number(stats['max'])}, среднее {_number(stats.get('mean', ''))}.")
    if kind is ColumnKind.DATETIME and "min" in stats:
        lines.append(f"Период: с {stats['min']} по {stats['max']}.")
    if summary.top_values:
        lines.append("Значения: " + ", ".join(summary.top_values) + ".")
    elif summary.sample_values and kind is not ColumnKind.NUMERIC:
        lines.append("Примеры значений: " + ", ".join(summary.sample_values) + ".")
    if summary.missing_pct:
        lines.append(f"Пропусков: {summary.missing_pct:g}%.")
    return "\n".join(lines)


def build_dataset_records(profile: DatasetProfile) -> list[IndexRecord]:
    """Запись про датасет + по записи на каждую колонку."""
    safe = build_llm_profile(profile)
    title = profile.filename
    by_name = {c.name: c for c in safe.columns}
    sheet = f", лист «{profile.sheet_name}»" if profile.sheet_name else ""

    periods = [
        f"с {c.date_min} по {c.date_max} (колонка {c.name})"
        for c in profile.columns
        if c.kind is ColumnKind.DATETIME and c.date_min and not c.is_sensitive
    ]
    dimensions = [
        f"{c.name} ({', '.join(t.value for t in c.top_values[:5])})" if c.top_values and not c.is_sensitive else c.name
        for c in profile.columns
        if c.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN, ColumnKind.DATETIME)
    ]
    measures = [c.name for c in profile.columns if c.kind is ColumnKind.NUMERIC]

    lines = [f"Датасет «{title}»{sheet}.", profile.description, f"Размер: {profile.row_count} строк, {profile.column_count} колонок."]
    if periods:
        lines.append("Период данных: " + "; ".join(periods) + ".")
    if dimensions:
        lines.append("Предполагаемые измерения (группировка и фильтры): " + "; ".join(dimensions) + ".")
    if measures:
        lines.append("Предполагаемые показатели (числовые): " + ", ".join(measures) + ".")
    lines.append("Колонки: " + "; ".join(f"{c.name} — {KIND_RU[c.kind]}" for c in profile.columns) + ".")

    base = {"dataset_id": profile.dataset_id, "filename": profile.filename, "sheet_name": profile.sheet_name or ""}
    records = [IndexRecord(f"ds-{profile.dataset_id}", "\n".join(lines), {**base, "entry_type": "dataset"})]
    for column in profile.columns:
        records.append(
            IndexRecord(
                f"col-{profile.dataset_id}-{column.position:03d}",
                _column_text(title, by_name[column.name], column.kind),
                {
                    **base,
                    "entry_type": "column",
                    "column_name": column.name,
                    "column_kind": column.kind.value,
                    "aliases": "|".join(guess_synonyms(column.name)),
                },
            )
        )
    return records


def content_hash(records: list[IndexRecord]) -> str:
    joined = "\x1f".join(f"{r.record_id}\x1e{r.text}" for r in records)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
