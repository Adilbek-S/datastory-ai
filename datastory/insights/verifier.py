"""Проверка достоверности выводов (этап verify_insights).

Проверки не доверяют LLM:
- каждое число в тексте должно совпадать с одним из числовых доказательств (с точностью до округления, которое
  указал сам автор текста), иначе это «выдуманное» число;
- причинные утверждения без оговорки «не доказано / не установлено» запрещены: совпадение по времени — не причина;
- слова о росте или снижении допустимы только при сравнении чисел (в доказательствах есть разность), причём
  направление слов должно совпадать со знаком разности.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datastory.models import Insight, NumericEvidence

# --- идентификаторы нарушений (используются в тестах и в описании методики)
NUMBERS_GROUNDED = "numbers-grounded"
NO_CAUSAL_CLAIM = "no-causal-claim"
TREND_NEEDS_COMPARISON = "trend-needs-comparison"
TREND_DIRECTION = "trend-direction-mismatch"
EVIDENCE_REQUIRED = "evidence-required"
UNKNOWN_EVIDENCE = "unknown-evidence"
CONTEXT_NOT_IN_SOURCE = "context-not-in-source"
PERIOD_NOT_IN_DATA = "period-not-in-data"


# какое правило методики (Skill datastory-analysis) нарушено
METHODOLOGY_RULE = {
    NUMBERS_GROUNDED: "A1", EVIDENCE_REQUIRED: "A1", UNKNOWN_EVIDENCE: "A1", NO_CAUSAL_CLAIM: "A3",
    TREND_NEEDS_COMPARISON: "A4", TREND_DIRECTION: "A4", PERIOD_NOT_IN_DATA: "A6", CONTEXT_NOT_IN_SOURCE: "A7",
}


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message} (правило методики {METHODOLOGY_RULE[self.rule]})"


_MULTIPLIERS = {"млрд": 1e9, "млн": 1e6, "тыс": 1e3}
_NUMBER = re.compile(
    r"(?<![\w.,])(?P<int>\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](?P<frac>\d+))?(?:[  ]?(?P<mult>млрд|млн|тыс)(?![а-яё]))?"
)

_STRONG_CAUSE = re.compile(
    r"вызван|вызвал|вызыва|из-за|привел|привёл|приводит|обусловлен|по вине|следстви|в результате|"
    r"caused|because|due to|led to|result of",
    re.IGNORECASE,
)
# «Причина — обновление», «причиной является…», «основная причина»: утверждение причины. «Нужны данные для анализа причин» — нет.
_WEAK_CAUSE = re.compile(
    r"причин\w*\s*(?:—|-|:|является|являются|был[аио]?|состо\w+|заключа\w+)|(?:основн|главн|вероятн|возможн)\w*\s+причин",
    re.IGNORECASE,
)
_DISCLAIMER = re.compile(
    r"не\s+(?:доказыва|доказан|подтвержд|означа|установлен|определен|определён|объясня|позвол|следует)|нельзя\s+утвержд|"
    r"неизвестн|не\s+известн|без\s+подтвержд|невозможно",
    re.IGNORECASE,
)
_NEGATION_BEFORE = re.compile(r"\bне\s+\w*\s*$", re.IGNORECASE)
_RISE = re.compile(r"\bрост\w*|\bвыр[оа]с\w*|\bувелич\w*|\bповыс\w*|\bприрост\w*", re.IGNORECASE)
_DECLINE = re.compile(r"\bсниж\w*|\bснизил\w*|\bупал\w*|\bпадени\w*|\bсократ\w*|\bуменьш\w*|\bпросе[лд]\w*|\bпросадк\w*", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")
_MONTHS = {
    1: r"\bянвар\w*", 2: r"\bфеврал\w*", 3: r"\bмарт\w*", 4: r"\bапрел\w*", 5: r"\bма(?:й|я|е|ем)\b", 6: r"\bиюн\w*",
    7: r"\bиюл\w*", 8: r"\bавгуст\w*", 9: r"\bсентябр\w*", 10: r"\bоктябр\w*", 11: r"\bноябр\w*", 12: r"\bдекабр\w*",
}
_LABEL_MONTH = re.compile(r"(?<!\d)\d{4}-(\d{2})(?!\d)|(?<!\d)(\d{2})\.\d{4}(?!\d)")


def _mask(text: str, mask: list[str]) -> str:
    for item in sorted({m for m in mask if m}, key=len, reverse=True):
        text = text.replace(item, " ")
    return text


def _allowed_numbers(facts: list[NumericEvidence]) -> list[float]:
    return [v for f in facts for v in (f.value, abs(f.value))]


def ungrounded_numbers(text: str, facts: list[NumericEvidence], mask: list[str]) -> list[str]:
    """Числа текста, которых нет среди доказательств. Даты и подписи групп (mask) и формулы не считаются числами вывода."""
    allowed = _allowed_numbers(facts)
    bad = []
    for match in _NUMBER.finditer(_mask(text, mask)):
        integer = re.sub(r"[   ]", "", match["int"])
        frac = match["frac"] or ""
        number = float(f"{integer}.{frac}" if frac else integer)
        scale = _MULTIPLIERS.get(match["mult"] or "", 1.0)
        tolerance = (0.5 * 10 ** -len(frac) + 1e-9) * scale  # допуск — округление до указанного знака
        if not any(abs(number * scale - a) <= tolerance for a in allowed):
            bad.append(match.group(0).strip())
    return bad


def causal_sentences(text: str) -> list[str]:
    """Предложения, утверждающие причину без оговорки. «Причина не установлена» и «не вызвано» допустимы."""
    flagged = []
    for sentence in _SENTENCE.split(text):
        if _DISCLAIMER.search(sentence):
            continue
        strong = _STRONG_CAUSE.search(sentence)
        if strong and not _NEGATION_BEFORE.search(sentence[: strong.start()]):
            flagged.append(sentence.strip())
        elif not strong and _WEAK_CAUSE.search(sentence) and not re.search(r"\bне\b|\bнет\b", sentence, re.IGNORECASE):
            flagged.append(sentence.strip())
    return flagged


def _data_months(labels: list[str]) -> set[int]:
    return {int(m[1] or m[2]) for label in labels for m in _LABEL_MONTH.finditer(label)}


def check_insight(
    insight: Insight, facts: list[NumericEvidence], mask: list[str], labels: list[str] | None = None
) -> list[Violation]:
    """Проверки текста вывода (название, краткий вывод, ограничение) относительно набора доказательств.

    labels — подписи групп графика (периоды, категории): названия месяцев в тексте должны соответствовать этим периодам.
    """
    violations: list[Violation] = []
    text = " ".join(filter(None, [insight.title, insight.text, insight.limitation]))

    if not insight.evidence:
        violations.append(Violation(EVIDENCE_REQUIRED, "вывод не ссылается ни на одно числовое доказательство"))
    known = {f.id for f in facts}
    for item in insight.evidence:
        if item.id not in known:
            violations.append(Violation(UNKNOWN_EVIDENCE, f"доказательство {item.id!r} не входит в набор фактов графика"))

    bad = ungrounded_numbers(text, facts, mask)
    if bad:
        violations.append(
            Violation(NUMBERS_GROUNDED, f"числа {', '.join(bad)} отсутствуют среди числовых доказательств: используйте только их")
        )

    if labels is not None:
        present = _data_months(labels)
        for number, pattern in _MONTHS.items():
            if number not in present and re.search(pattern, text, re.IGNORECASE):
                name = re.search(pattern, text, re.IGNORECASE).group(0)
                violations.append(Violation(PERIOD_NOT_IN_DATA, f"в тексте упомянут период «{name}», которого нет в данных этого графика"))

    for sentence in causal_sentences(text):
        violations.append(
            Violation(NO_CAUSAL_CLAIM, f"утверждение о причине без подтверждения: «{sentence}». Пишите «совпадает по времени», причина не установлена")
        )

    rises, declines = bool(_RISE.search(text)), bool(_DECLINE.search(text))
    if rises or declines:
        differences = [e for e in insight.evidence if e.kind == "difference"]
        if not differences:
            violations.append(
                Violation(TREND_NEEDS_COMPARISON, "рост или снижение заявлены без сравнения чисел: сошлитесь на доказательство-разность")
            )
        elif declines and not rises and not any(e.value < 0 for e in differences):
            violations.append(Violation(TREND_DIRECTION, "текст говорит о снижении, а приведённые сравнения не показывают снижения"))
        elif rises and not declines and not any(e.value > 0 for e in differences):
            violations.append(Violation(TREND_DIRECTION, "текст говорит о росте, а приведённые сравнения не показывают роста"))
    return violations
