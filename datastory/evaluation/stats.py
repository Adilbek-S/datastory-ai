"""Статистика для A/B сравнения: парные подсчёты, точный тест знаков, бутстрап-интервал и правило вывода.

Вывод «конфигурация лучше» допускается только если 95% доверительный интервал разницы целиком по одну сторону от нуля;
иначе результат называется неподтверждённым. Правило — в verdict(), тесты проверяют его границы.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

BOOTSTRAP_SAMPLES = 10_000
SEED = 20260927


def paired_counts(a: list[bool], b: list[bool]) -> dict[str, int]:
    """Парные наблюдения одних и тех же (случай, повтор): обе успешны, только A, только B, обе неуспешны."""
    if len(a) != len(b):
        raise ValueError("Ряды A и B должны быть одной длины: наблюдения парные.")
    return {
        "both": sum(x and y for x, y in zip(a, b)), "only_a": sum(x and not y for x, y in zip(a, b)),
        "only_b": sum(y and not x for x, y in zip(a, b)), "neither": sum(not x and not y for x, y in zip(a, b)),
    }


def sign_test_pvalue(only_a: int, only_b: int) -> float:
    """Точный двусторонний тест знаков по расходящимся парам (McNemar): p при равенстве шансов 1/2."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class Difference:
    mean: float  # B − A (доли или единицы измерения)
    low: float
    high: float
    n_units: int  # число случаев, по которым считался интервал


def bootstrap_difference(per_case_a: list[float], per_case_b: list[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = SEED, alpha: float = 0.05) -> Difference:
    """Разница средних B − A с бутстрап-интервалом; единица перевыборки — случай (повторы усредняются внутри случая)."""
    if len(per_case_a) != len(per_case_b) or not per_case_a:
        raise ValueError("Нужны непустые ряды одинаковой длины: по одному значению на случай.")
    diffs = [b - a for a, b in zip(per_case_a, per_case_b)]
    n, rng = len(diffs), random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    low, high = means[int(samples * alpha / 2)], means[min(samples - 1, int(samples * (1 - alpha / 2)))]
    return Difference(sum(diffs) / n, low, high, n)


def verdict(diff: Difference, higher_is_better: bool = True) -> str:
    """confirmed_better | confirmed_worse | not_confirmed | no_difference — по положению интервала относительно нуля."""
    if diff.low == 0 == diff.high == diff.mean:
        return "no_difference"
    sign = 1 if higher_is_better else -1
    low, high = (diff.low, diff.high) if sign == 1 else (-diff.high, -diff.low)
    if low > 0:
        return "confirmed_better"
    if high < 0:
        return "confirmed_worse"
    return "not_confirmed"
