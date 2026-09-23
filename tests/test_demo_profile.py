"""Главный демонстрационный файл: все колонки и типы данных определяются правильно."""
import io
import json

import pandas as pd
import pytest

from datastory.errors import UnsupportedFileError
from datastory.file_processing.loader import list_sheets, read_table
from datastory.models import ColumnKind
from datastory.profiler.profiler import build_profile
from scripts import generate_demo_data as gen

DEMO = gen.DEFAULT_OUT
XLSX = DEMO / "transactions_2026.xlsx"

EXPECTED_KINDS = {
    "Month": ColumnKind.DATETIME,
    "Transactions": ColumnKind.NUMERIC,
    "Successful": ColumnKind.NUMERIC,
    "Failed": ColumnKind.NUMERIC,
    "Amount_KZT": ColumnKind.NUMERIC,
    "Channel": ColumnKind.CATEGORICAL,
}


@pytest.fixture(scope="module")
def loaded():
    data = XLSX.read_bytes()
    sheet = list_sheets(data)[0]
    raw = read_table(data, XLSX.name, sheet)
    return build_profile(raw, XLSX.name, sheet)


def test_demo_workbook_has_single_sheet():
    assert list_sheets(XLSX.read_bytes()) == ["transactions"]


def test_all_columns_are_found_in_order(loaded):
    profile, _ = loaded
    assert [c.name for c in profile.columns] == gen.COLUMNS
    assert profile.column_count == 6 and profile.sheet_name == "transactions"


def test_all_column_types_are_detected_correctly(loaded):
    profile, _ = loaded
    assert {c.name: c.kind for c in profile.columns} == EXPECTED_KINDS
    assert all(not c.is_overridden for c in profile.columns)  # определено автоматически, без ручной правки
    assert profile.detected_date_columns == ["Month"]
    assert profile.detected_numeric_columns == ["Transactions", "Successful", "Failed", "Amount_KZT"]
    assert profile.detected_category_columns == ["Channel"]


def test_dtypes_after_typing(loaded):
    _, typed = loaded
    assert pd.api.types.is_datetime64_any_dtype(typed["Month"])
    for column in ("Transactions", "Successful", "Failed", "Amount_KZT"):
        assert pd.api.types.is_integer_dtype(typed[column]), column
    assert pd.api.types.is_string_dtype(typed["Channel"])


def test_counts_and_quality(loaded):
    profile, _ = loaded
    assert profile.row_count == 18
    assert profile.missing_cell_count == 0 and profile.duplicate_row_count == 0
    assert profile.quality_issues == []
    assert not any(c.is_sensitive for c in profile.columns)


def test_column_details(loaded):
    profile, _ = loaded
    month, channel = profile.column("Month"), profile.column("Channel")
    assert (month.date_min, month.date_max, month.unique_count) == ("2026-01-01", "2026-06-01", 6)
    assert channel.unique_count == 3
    assert {t.value: t.count for t in channel.top_values} == {"Mobile": 6, "Web": 6, "API": 6}


def test_statistics_match_expected_metrics(loaded):
    profile, typed = loaded
    expected = json.loads((DEMO / "expected_metrics.json").read_text(encoding="utf-8"))
    assert profile.row_count == expected["row_count"]
    assert [c.name for c in profile.columns] == expected["columns"]

    for column, key in (("Transactions", "transactions"), ("Successful", "successful"), ("Failed", "failed"), ("Amount_KZT", "amount_kzt")):
        stats = profile.column(column).numeric_stats
        assert stats.mean * profile.row_count == pytest.approx(expected["totals"][key], rel=1e-9)
        assert typed[column].sum() == expected["totals"][key]


def test_same_result_from_csv_export():
    raw = read_table(XLSX.read_bytes(), XLSX.name)
    csv_bytes = raw.to_csv(index=False, sep=";").encode("utf-8-sig")
    profile, typed = build_profile(read_table(csv_bytes, "transactions_2026.csv"), "transactions_2026.csv")
    assert {c.name: c.kind for c in profile.columns} == EXPECTED_KINDS
    assert profile.row_count == 18 and typed["Transactions"].sum() == raw["Transactions"].sum()


def test_demo_images_and_pdf_are_not_tables():
    for name in ("transactions_screenshot.png", "business_metrics.pdf"):
        with pytest.raises(UnsupportedFileError):
            read_table((DEMO / name).read_bytes(), name)


def test_demo_profile_survives_json(loaded):
    profile, _ = loaded
    assert json.loads(io.StringIO(profile.model_dump_json()).read())["detected_date_columns"] == ["Month"]
