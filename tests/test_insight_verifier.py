"""Проверка достоверности выводов: числа только из расчётов, без причин «по совпадению», без роста без сравнения."""
import pytest

from datastory.analytics.models import MetricResult, MetricRow
from datastory.insights.facts import build_facts, group_labels
from datastory.insights.verifier import (
    EVIDENCE_REQUIRED,
    NO_CAUSAL_CLAIM,
    PERIOD_NOT_IN_DATA,
    NUMBERS_GROUNDED,
    TREND_DIRECTION,
    TREND_NEEDS_COMPARISON,
    causal_sentences,
    check_insight,
    ungrounded_numbers,
)
from datastory.models import Insight

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04"]


def series(values, metric="success_rate", unit="%", label="Success Rate", axis="Month", names=MONTHS) -> MetricResult:
    rows = [MetricRow(group={axis: n}, value=v) for n, v in zip(names, values)]
    return MetricResult(dataset_id="d", metric=metric, label=label, unit=unit, formula="SUM(Successful) / SUM(Transactions) × 100",
                        group_by=[axis], rows=rows, overall=MetricRow(value=sum(present) / len(present) if (present := [v for v in values if v is not None]) else None))


RATE = series([97.5, 97.8, 92.0, 97.9])
FACTS = build_facts(RATE, "time")
MASK = [*group_labels(RATE), RATE.formula, "Success Rate"]


def insight(text, ids=("first", "last", "change"), title="Динамика", limitation=None) -> Insight:
    by_id = {f.id: f for f in FACTS}
    return Insight(title=title, text=text, evidence=[by_id[i] for i in ids], limitation=limitation)


def rules(found) -> set[str]:
    return {v.rule for v in found}


# ================================================================== факты
def test_facts_contain_values_and_comparisons_computed_in_python():
    facts = {f.id: f for f in FACTS}
    assert facts["first"].value == 97.5 and facts["last"].value == 97.9 and facts["min"].group == "2026-03"
    assert facts["change"].value == pytest.approx(0.4) and facts["change"].kind == "difference"
    assert facts["drop"].value == pytest.approx(-5.8) and facts["drop"].group == "2026-02 → 2026-03"
    assert facts["rise"].value == pytest.approx(5.9)
    assert facts["drop"].formatted == "5,80 п.п." and facts["periods"].kind == "count"


def test_facts_for_sums_include_relative_change_and_category_shares():
    counts = series([100.0, 110.0, 121.0], metric="transaction_count", unit="шт.", label="Количество", names=MONTHS[:3])
    assert {f.id: f for f in build_facts(counts, "time")}["change_pct"].formatted == "21,0%"

    channels = MetricResult(
        dataset_id="d", metric="transaction_count", label="Количество транзакций", unit="шт.", formula="SUM(Transactions)",
        group_by=["Channel"], rows=[MetricRow(group={"Channel": "Mobile"}, value=300), MetricRow(group={"Channel": "API"}, value=100)],
        overall=MetricRow(value=400),
    )
    facts = {f.id: f for f in build_facts(channels, "category")}
    assert facts["cat1_share"].value == 75.0 and facts["cat1_share"].group == "Mobile" and facts["cat2_share"].formatted == "25,0%"


def test_series_with_gaps_and_single_period_do_not_crash():
    gap = series([97.0, None, 96.0])
    assert {f.id for f in build_facts(gap, "time")} >= {"first", "last", "change"}
    assert {f.id for f in build_facts(series([97.0], names=MONTHS[:1]), "time")} == {"overall", "first", "periods"}
    assert build_facts(series([]), "time") == []


# ================================================================== числа
def test_numbers_that_match_evidence_are_accepted():
    text = "В периоде 2026-01 значение 97,50%, в 2026-04 — 97,90%; изменение 0,40 п.п."
    assert ungrounded_numbers(text, FACTS, MASK) == []


def test_rounding_to_fewer_decimals_is_accepted_but_other_numbers_are_not():
    assert ungrounded_numbers("Значение около 97,5% и 5,8 п.п.", FACTS, MASK) == []
    assert ungrounded_numbers("Значение выросло до 98,3%", FACTS, MASK) == ["98,3"]
    assert ungrounded_numbers("Падение на 5,80 п.п. и ещё на 7,1", FACTS, MASK) == ["7,1"]


def test_period_labels_formulas_and_years_from_labels_are_not_numbers():
    assert ungrounded_numbers("Период 2026-03: SUM(Successful) / SUM(Transactions) × 100", FACTS, [*MASK, "2026"]) == []


def test_scaled_numbers_are_matched_by_the_stated_multiplier():
    volume = series([4_185_883_345.0, 4_325_251_527.0], metric="transaction_volume", unit="KZT", label="Объём", names=MONTHS[:2])
    facts = build_facts(volume, "time")
    assert ungrounded_numbers("Объём вырос с 4,19 млрд KZT до 4,33 млрд KZT", facts, []) == []
    assert ungrounded_numbers("Объём составил 4,19 млн KZT", facts, []) == ["4,19 млн"]  # множитель указан неверно


def test_thousand_separators_are_read_as_one_number():
    counts = series([317_164.0, 330_360.0], metric="transaction_count", unit="шт.", label="Количество", names=MONTHS[:2])
    facts = build_facts(counts, "time")
    assert ungrounded_numbers("С 317 164 до 330 360 транзакций", facts, []) == []
    assert ungrounded_numbers("Было 317 165 транзакций", facts, []) == ["317 165"]


def test_check_insight_reports_invented_number_with_the_rule_id():
    found = check_insight(insight("Успешность достигла 99,9%."), FACTS, MASK)
    assert NUMBERS_GROUNDED in rules(found) and "99,9" in str(found[0])


# ================================================================== причинность
@pytest.mark.parametrize(
    "sentence",
    [
        "Снижение вызвано обновлением инфраструктуры.",
        "Падение произошло из-за обновления.",
        "Обновление привело к снижению успешности.",
        "Это результат обновления системы: в результате упала успешность.",
        "Причина — плановое обновление.",
        "Основная причина снижения: сбой.",
        "Возможно, это следствие работ.",
        "The drop was caused by the upgrade.",
    ],
)
def test_causal_claims_are_flagged(sentence):
    assert causal_sentences(sentence), sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "Причина изменения по данным не установлена.",
        "Совпадает по времени с плановым обновлением, но это не доказывает причинную связь.",
        "Снижение не вызвано обновлением: данных для такого вывода нет.",
        "Причина снижения неизвестна.",
        "Необходимы дополнительные данные для анализа причин снижения.",
        "Данные о причинах отсутствуют.",
        "Нельзя утверждать, что снижение вызвано обновлением.",
        "Успешность снизилась в марте.",
    ],
)
def test_neutral_or_disclaiming_sentences_are_allowed(sentence):
    assert causal_sentences(sentence) == [], sentence


def test_check_insight_rejects_causal_text_in_any_field():
    text = "Успешность упала на 5,80 п.п. из-за обновления."
    assert NO_CAUSAL_CLAIM in rules(check_insight(insight(text, ("drop",)), FACTS, MASK))
    hidden = insight("Успешность упала на 5,80 п.п.", ("drop",), limitation="Падение вызвано обновлением.")
    assert NO_CAUSAL_CLAIM in rules(check_insight(hidden, FACTS, MASK))


# ================================================================== рост и снижение
def test_trend_words_need_a_comparison():
    plain = insight("Значение выросло.", ids=("first",))
    assert TREND_NEEDS_COMPARISON in rules(check_insight(plain, FACTS, MASK))
    assert check_insight(insight("Значение выросло на 0,40 п.п."), FACTS, MASK) == []


def test_trend_direction_must_match_the_comparison():
    assert TREND_DIRECTION in rules(check_insight(insight("Значение снизилось на 0,40 п.п."), FACTS, MASK))
    assert TREND_DIRECTION in rules(check_insight(insight("Значение выросло на 5,80 п.п.", ("drop",)), FACTS, MASK))
    assert check_insight(insight("Наибольшее падение — на 5,80 п.п.", ("drop",)), FACTS, MASK) == []


def test_word_prefixes_do_not_cause_false_trend_matches():
    assert check_insight(insight("Это просто значение 97,50%.", ("first",)), FACTS, MASK) == []


def test_an_insight_without_evidence_is_rejected():
    empty = Insight(title="Вывод", text="Значение стабильно.", evidence=[])
    assert rules(check_insight(empty, FACTS, MASK)) == {EVIDENCE_REQUIRED}


# ================================================================== периоды
def test_month_names_must_match_periods_present_in_the_data():
    labels = group_labels(RATE)  # 2026-01 … 2026-04
    assert PERIOD_NOT_IN_DATA not in rules(check_insight(insight("Падение в марте на 5,80 п.п.", ("drop",)), FACTS, MASK, labels))
    found = check_insight(insight("Падение в июне на 5,80 п.п.", ("drop",)), FACTS, MASK, labels)
    assert PERIOD_NOT_IN_DATA in rules(found) and "июне" in str(found[0])


def test_a_chart_without_periods_cannot_mention_a_month():
    """Распределение по каналам посчитано по всем строкам: «в марте 2026 года» — выдуманный период."""
    found = check_insight(insight("В марте 2026 года значение составило 97,50%.", ("first",)), FACTS, MASK, ["Mobile", "Web", "API"])
    assert PERIOD_NOT_IN_DATA in rules(found)
    assert check_insight(insight("Значение составило 97,50%.", ("first",)), FACTS, MASK, ["Mobile", "Web", "API"]) == []
