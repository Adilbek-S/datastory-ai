"""Эвристики для персональных данных. Такие колонки и значения не передаются в LLM."""
from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as pdt

NAME_PATTERN = re.compile(
    r"e-?mail|почт|phone|телефон|(?<![а-я])тел(?![а-я])|mobile|(?<![a-zа-я])iin(?![a-zа-я])|иин|инн|ssn"
    r"|passport|паспорт|(?<![a-zа-я])fio(?![a-zа-я])|фио|full_?name|first_?name|last_?name|middle_?name"
    r"|surname|фамили|отчеств|address|адрес|(?<![a-z])card(?![a-z])|номер\s*карт|iban|birth|рожд"
    r"|(?<![a-zа-я])name(?![a-zа-я])|(?<![а-я])имя(?![а-я])",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
PHONE_RE = re.compile(r"^\+?[\d\s\-()]{10,20}$")
CARD_RE = re.compile(r"^\d{4}([ -]?\d{4}){2,3}(\d{0,3})$")

VALUE_MATCH_RATIO = 0.6
SAMPLE_SIZE = 200


def _luhn_ok(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if flip:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        flip = not flip
    return total % 10 == 0


def _looks_like_phone(text: str, digits: str) -> bool:
    if not PHONE_RE.match(text) or not 10 <= len(digits) <= 12:
        return False
    formatted = text.startswith("+") or "(" in text or "-" in text
    return formatted or (len(digits) == 11 and digits[0] in "78")


def looks_like_pii_value(value: str) -> bool:
    """Значение похоже на e-mail, телефон или номер банковской карты."""
    text = value.strip()
    if EMAIL_RE.match(text):
        return True
    digits = re.sub(r"\D", "", text)
    if CARD_RE.match(text) and 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return True
    return _looks_like_phone(text, digits)


def detect_sensitive(name: str, raw: pd.Series) -> str:
    """Причина, по которой колонка считается персональной, либо пустая строка."""
    if NAME_PATTERN.search(str(name)):
        return "название колонки"
    if pdt.is_numeric_dtype(raw) or pdt.is_datetime64_any_dtype(raw) or pdt.is_bool_dtype(raw):
        return ""
    values = raw.dropna().astype(str).head(SAMPLE_SIZE)
    if values.empty:
        return ""
    matches = sum(looks_like_pii_value(v) for v in values)
    if matches / len(values) >= VALUE_MATCH_RATIO:
        return "значения похожи на e-mail, телефон или номер карты"
    return ""
