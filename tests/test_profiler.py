"""Профайлер: типы колонок, качество данных, ручная настройка типов, персональные данные."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from datastory.errors import NoDataError
from datastory.models import ColumnKind, DatasetProfile, Severity
from datastory.profiler.pii import detect_sensitive, looks_like_pii_value
from datastory.profiler.profiler import build_profile

N = 20


def kinds(profile: DatasetProfile) -> dict[str, ColumnKind]:
    return {c.name: c.kind for c in profile.columns}


def issue_codes(profile: DatasetProfile, column: str | None = None) -> list[str]:
    return [i.code for i in profile.quality_issues if column is None or i.column == column]


@pytest.fixture
def mixed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount_text": [f"{i} {i:03d},5" for i in range(1, N + 1)],  # «1 001,5»
            "amount_en": [f"{i},000.25" for i in range(1, N + 1)],  # «1,000.25»
            "date_dmy": [f"{(i % 28) + 1:02d}.03.2026" for i in range(N)],
            "date_ym": [f"2026-{(i % 12) + 1:02d}" for i in range(N)],
            "date_iso_time": [f"2026-01-{(i % 28) + 1:02d} 10:30:00" for i in range(N)],
            "cat": ["A", "B"] * (N // 2),
            "free_text": [f"unique comment number {i}" for i in range(N)],
            "flag": ["да", "нет"] * (N // 2),
            "count": list(range(N)),
            "ratio": np.linspace(0.1, 0.9, N),
            "when": pd.date_range("2026-01-01", periods=N),
        }
    )


# ------------------------------------------------------------------ типы
def test_column_kinds(mixed):
    profile, _ = build_profile(mixed, "t.csv")
    assert kinds(profile) == {
        "amount_text": ColumnKind.NUMERIC,
        "amount_en": ColumnKind.NUMERIC,
        "date_dmy": ColumnKind.DATETIME,
        "date_ym": ColumnKind.DATETIME,
        "date_iso_time": ColumnKind.DATETIME,
        "cat": ColumnKind.CATEGORICAL,
        "free_text": ColumnKind.TEXT,
        "flag": ColumnKind.BOOLEAN,
        "count": ColumnKind.NUMERIC,
        "ratio": ColumnKind.NUMERIC,
        "when": ColumnKind.DATETIME,
    }
    assert profile.detected_category_columns == ["cat"]
    assert set(profile.detected_date_columns) == {"date_dmy", "date_ym", "date_iso_time", "when"}
    assert profile.detected_numeric_columns == ["amount_text", "amount_en", "count", "ratio"]


def test_text_numbers_and_dates_are_converted(mixed):
    _, typed = build_profile(mixed, "t.csv")
    assert typed["amount_text"].iloc[0] == pytest.approx(1001.5)
    assert typed["amount_en"].iloc[0] == pytest.approx(1000.25)
    assert typed["date_dmy"].iloc[0] == pd.Timestamp("2026-01-01") + pd.DateOffset(months=2)
    assert typed["date_ym"].iloc[0] == pd.Timestamp("2026-01-01")
    assert typed["date_iso_time"].iloc[0] == pd.Timestamp("2026-01-01 10:30:00")
    assert typed["flag"].iloc[0] is True or bool(typed["flag"].iloc[0]) is True
    assert pd.api.types.is_numeric_dtype(typed["amount_text"])
    assert pd.api.types.is_datetime64_any_dtype(typed["date_ym"])


def test_year_like_integers_are_numbers_not_dates():
    profile, _ = build_profile(pd.DataFrame({"year": [2021, 2022, 2023, 2024]}), "t.csv")
    assert kinds(profile) == {"year": ColumnKind.NUMERIC}


def test_null_tokens_are_missing_not_invalid():
    frame = pd.DataFrame({"v": ["10", "-", "20", "n/a", "30", "", "40", "50", "60", "70"]})
    profile, typed = build_profile(frame, "t.csv")
    col = profile.column("v")
    assert col.kind is ColumnKind.NUMERIC
    assert col.missing_count == 3
    assert "invalid_numeric" not in issue_codes(profile)
    assert typed["v"].isna().sum() == 3


# ------------------------------------------------------------------ счётчики
def test_counts_rows_missing_duplicates():
    frame = pd.DataFrame({"a": [1, 1, 2, None, 3, 3], "b": ["x", "x", "y", "z", None, None]})
    profile, _ = build_profile(frame, "t.xlsx", "Sheet1")
    assert profile.row_count == 6 and profile.column_count == 2
    assert profile.missing_cell_count == 3
    assert profile.duplicate_row_count == 2  # (1,x) и (3,None)
    assert profile.sheet_name == "Sheet1" and profile.filename == "t.xlsx"
    assert "duplicate_rows" in issue_codes(profile)
    assert len(profile.dataset_id) == 12


def test_missing_value_severity_levels():
    frame = pd.DataFrame(
        {
            "few": [1.0] * 99 + [None],  # 1%  -> info
            "some": [1.0] * 90 + [None] * 10,  # 10% -> warning
            "most": [1.0] * 40 + [None] * 60,  # 60% -> error
        }
    )
    profile, _ = build_profile(frame, "t.csv")
    severity = {i.column: i.severity for i in profile.quality_issues if i.code == "missing_values"}
    assert severity == {"few": Severity.INFO, "some": Severity.WARNING, "most": Severity.ERROR}
    assert profile.column("some").missing_pct == 10.0


def test_empty_constant_and_unnamed_columns():
    frame = pd.DataFrame({"empty": [None] * 5, "const": ["x"] * 5, "Unnamed: 2": range(5), "ok": range(5)})
    profile, _ = build_profile(frame, "t.csv")
    assert "empty_column" in issue_codes(profile, "empty")
    assert "constant_column" in issue_codes(profile, "const")
    assert "unnamed_column" in issue_codes(profile, "Unnamed: 2")
    assert issue_codes(profile, "ok") == []


def test_clean_data_has_no_issues():
    profile, _ = build_profile(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}), "t.csv")
    assert profile.quality_issues == []


# ------------------------------------------------------------------ некорректные числа
def test_invalid_numeric_values_are_reported():
    frame = pd.DataFrame({"v": ["10", "20", "abc", "30", "40", "50", "60", "70", "80", "90"]})
    profile, typed = build_profile(frame, "t.csv")
    col = profile.column("v")
    assert col.kind is ColumnKind.NUMERIC
    assert col.missing_count == 0  # нераспознанное значение — не пропуск
    issue = next(i for i in profile.quality_issues if i.code == "invalid_numeric")
    assert issue.count == 1 and issue.severity is Severity.WARNING
    assert "abc" in issue.message
    assert np.isnan(typed["v"].iloc[2]) and typed["v"].iloc[3] == 30


def test_many_invalid_numbers_are_an_error():
    frame = pd.DataFrame({"v": ["1", "2", "3", "4", "5", "6", "7", "8", "x", "y"]})
    profile, _ = build_profile(frame, "t.csv", type_overrides={"v": ColumnKind.NUMERIC})
    issue = next(i for i in profile.quality_issues if i.code == "invalid_numeric")
    assert issue.severity is Severity.ERROR and issue.count == 2


def test_forced_numeric_on_text_marks_everything_invalid():
    frame = pd.DataFrame({"city": ["Almaty", "Astana", "Shymkent"]})
    profile, typed = build_profile(frame, "t.csv", type_overrides={"city": ColumnKind.NUMERIC})
    issue = next(i for i in profile.quality_issues if i.code == "invalid_numeric")
    assert issue.count == 3 and issue.severity is Severity.ERROR
    assert typed["city"].isna().all()


def test_invalid_dates_are_reported():
    frame = pd.DataFrame({"d": ["2026-01-01"] * 9 + ["31.31.2026"]})
    profile, typed = build_profile(frame, "t.csv")
    assert profile.column("d").kind is ColumnKind.DATETIME
    assert any(i.code == "invalid_datetime" and i.count == 1 for i in profile.quality_issues)
    assert typed["d"].isna().sum() == 1


# ------------------------------------------------------------------ ручная настройка типов
def test_type_override_changes_profile_and_data():
    frame = pd.DataFrame({"code": [101, 102, 103, 101, 102, 103], "amount": [1.5, 2, 3, 4, 5, 6]})
    auto, _ = build_profile(frame, "t.csv")
    assert kinds(auto)["code"] is ColumnKind.NUMERIC

    profile, typed = build_profile(frame, "t.csv", type_overrides={"code": ColumnKind.CATEGORICAL})
    col = profile.column("code")
    assert col.kind is ColumnKind.CATEGORICAL and col.detected_kind is ColumnKind.NUMERIC
    assert col.is_overridden
    assert profile.detected_category_columns == ["code"]
    assert profile.detected_numeric_columns == ["amount"]
    assert typed["code"].tolist() == ["101", "102", "103", "101", "102", "103"]
    assert col.top_values and col.numeric_stats is None


def test_override_text_to_numeric_and_to_date():
    frame = pd.DataFrame({"n": ["001", "002", "003"], "d": ["1/2/2026", "2/3/2026", "10/12/2026"]})
    profile, typed = build_profile(
        frame, "t.csv", type_overrides={"n": ColumnKind.NUMERIC, "d": ColumnKind.DATETIME}
    )
    assert typed["n"].tolist() == [1, 2, 3]
    assert pd.api.types.is_datetime64_any_dtype(typed["d"])
    assert profile.detected_date_columns == ["d"]


def test_unknown_override_columns_are_ignored():
    profile, _ = build_profile(pd.DataFrame({"a": [1, 2]}), "t.csv", type_overrides={"nope": ColumnKind.TEXT})
    assert kinds(profile) == {"a": ColumnKind.NUMERIC}


# ------------------------------------------------------------------ персональные данные
@pytest.mark.parametrize(
    "name", ["Email", "e-mail клиента", "Телефон", "phone_number", "ФИО", "Full Name", "customer_name", "IIN", "Адрес", "card_number"]
)
def test_sensitive_by_column_name(name):
    assert detect_sensitive(name, pd.Series([1, 2, 3]))


@pytest.mark.parametrize("name", ["Channel", "Month", "Transactions", "Successful", "Failed", "Amount_KZT", "region", "filename"])
def test_regular_columns_are_not_sensitive(name):
    assert not detect_sensitive(name, pd.Series(["a", "b", "c"]))


def test_sensitive_by_values():
    emails = pd.Series(["ivan@example.kz", "aida@mail.ru", "x.y@corp.com"])
    phones = pd.Series(["+7 (701) 123-45-67", "8 777 123 45 67", "+7-702-555-11-22"])
    cards = pd.Series(["4111 1111 1111 1111", "5500 0000 0000 0004", "4012 8888 8888 1881"])
    assert "e-mail" in detect_sensitive("contact", emails)
    assert detect_sensitive("col1", phones)
    assert detect_sensitive("col2", cards)


def test_amounts_and_dates_are_not_mistaken_for_pii():
    assert not looks_like_pii_value("1 234 567 890")
    assert not looks_like_pii_value("2026-03-01")
    assert not looks_like_pii_value("2026-03-01 10:30:00")
    assert not detect_sensitive("value", pd.Series([1234567890, 9876543210]))


def test_sensitive_flag_in_profile_and_manual_override():
    frame = pd.DataFrame({"email": ["a@b.kz", "c@d.kz"], "amount": [1, 2], "note": ["x", "y"]})
    profile, _ = build_profile(frame, "t.csv")
    assert profile.column("email").is_sensitive and not profile.column("amount").is_sensitive
    assert "possible_personal_data" in issue_codes(profile, "email")

    profile, _ = build_profile(frame, "t.csv", sensitive_overrides={"email": False, "note": True})
    assert not profile.column("email").is_sensitive
    assert profile.column("note").is_sensitive and profile.column("note").sensitive_reason == "отмечено пользователем"


# ------------------------------------------------------------------ прочее
def test_profile_is_json_serializable():
    profile, _ = build_profile(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), "t.csv")
    restored = DatasetProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile


def test_description_default_and_custom():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    profile, _ = build_profile(frame, "t.csv")
    assert "2 строк" in profile.description and "a" in profile.description
    custom, _ = build_profile(frame, "t.csv", description="Мои данные")
    assert custom.description == "Мои данные"


def test_empty_frame_raises_no_data():
    with pytest.raises(NoDataError):
        build_profile(pd.DataFrame(), "t.csv")
    with pytest.raises(NoDataError):
        build_profile(pd.DataFrame({"a": []}), "t.csv")


def test_stats_are_computed_by_pandas():
    profile, _ = build_profile(pd.DataFrame({"v": [1, 2, 3, 4, 10]}), "t.csv")
    stats = profile.column("v").numeric_stats
    assert (stats.min, stats.max, stats.mean, stats.median) == (1, 10, 4, 3)
    assert stats.std == pytest.approx(3.5355, abs=1e-3)


def test_profiling_does_not_load_llm_libraries():
    code = (
        "import sys, pandas as pd;"
        "from datastory.profiler.profiler import build_profile;"
        "build_profile(pd.DataFrame({'a':[1,2]}), 't.csv');"
        "assert not {'openai','langchain_openai'} & set(sys.modules), 'LLM-библиотека загружена'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
