"""profile_dataset: сводка датасета (строки, колонки, типы, показатели, измерения, статистики, пропуски)."""
import json

import numpy as np
import pandas as pd
import pytest

from datastory.analytics.summary import build_dataset_summary
from datastory.file_processing.loader import read_table
from datastory.profiler.profiler import build_profile
from scripts import generate_demo_data as gen


@pytest.fixture(scope="module")
def demo():
    raw = read_table((gen.DEFAULT_OUT / "transactions_2026.xlsx").read_bytes(), "transactions_2026.xlsx", "transactions")
    profile, typed = build_profile(raw, "transactions_2026.xlsx", "transactions")
    return build_dataset_summary(profile), typed


def test_rows_columns_and_types(demo):
    summary, _ = demo
    assert (summary.row_count, summary.column_count) == (18, 6)
    assert summary.column_names == gen.COLUMNS
    assert summary.column_types == {
        "Month": "datetime", "Transactions": "numeric", "Successful": "numeric", "Failed": "numeric",
        "Amount_KZT": "numeric", "Channel": "categorical",
    }
    assert (summary.filename, summary.sheet_name) == ("transactions_2026.xlsx", "transactions")


def test_measures_and_dimensions(demo):
    summary, _ = demo
    assert summary.numeric_measures == ["Transactions", "Successful", "Failed", "Amount_KZT"]
    assert summary.dimensions == ["Month", "Channel"]
    assert summary.sensitive_columns == []


def test_all_six_metrics_are_available_for_the_demo_dataset(demo):
    assert demo[0].available_metrics == [
        "transaction_count", "successful_count", "failed_count", "transaction_volume", "success_rate", "average_transaction_amount",
    ]


def test_numeric_statistics_match_pandas(demo):
    summary, typed = demo
    stats = next(c for c in summary.columns if c.name == "Transactions").statistics
    column = typed["Transactions"]
    assert stats["min"] == column.min() and stats["max"] == column.max()
    assert stats["mean"] == pytest.approx(column.mean()) and stats["median"] == pytest.approx(column.median())
    assert stats["std"] == pytest.approx(column.std())


def test_datetime_and_category_statistics(demo):
    summary, _ = demo
    by_name = {c.name: c for c in summary.columns}
    assert by_name["Month"].statistics == {"min": "2026-01-01", "max": "2026-06-01"}
    assert by_name["Channel"].statistics == {"top_values": {"Mobile": 6, "Web": 6, "API": 6}}
    assert by_name["Channel"].unique_count == 3 and by_name["Month"].unique_count == 6


def test_missing_information_for_clean_data(demo):
    summary, _ = demo
    assert summary.missing_cell_count == 0 and summary.duplicate_row_count == 0 and summary.missing_by_column == {}
    assert all(c.missing_count == 0 and c.missing_pct == 0 for c in summary.columns)


def test_missing_values_are_reported_per_column():
    frame = pd.DataFrame({"Transactions": [1, 2, np.nan, 4], "Channel": ["a", None, "a", None], "Month": ["2026-01"] * 4})
    summary = build_dataset_summary(build_profile(frame, "gaps.csv")[0])
    assert summary.missing_cell_count == 3
    assert {k: (v.count, v.pct) for k, v in summary.missing_by_column.items()} == {"Transactions": (1, 25.0), "Channel": (2, 50.0)}
    assert next(c for c in summary.columns if c.name == "Channel").missing_pct == 50.0


def test_duplicates_are_reported():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert build_dataset_summary(build_profile(frame, "d.csv")[0]).duplicate_row_count == 1


def test_available_metrics_depend_on_columns():
    frame = pd.DataFrame({"Transactions": [10, 20], "Successful": [9, 18], "Region": ["a", "b"]})
    summary = build_dataset_summary(build_profile(frame, "partial.csv")[0])
    assert summary.available_metrics == ["transaction_count", "successful_count", "success_rate"]
    # нет колонок платёжной системы: доступны суммы числовых колонок; без числовых колонок — ничего
    assert build_dataset_summary(build_profile(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "none.csv")[0]).available_metrics == ["sum:x", "sum:y"]
    assert build_dataset_summary(build_profile(pd.DataFrame({"a": ["u", "v"], "b": ["p", "q"]}), "text.csv")[0]).available_metrics == []


def test_non_numeric_metric_column_does_not_enable_metric():
    frame = pd.DataFrame({"Transactions": ["много", "мало", "средне"], "Successful": [1, 2, 3]})
    assert build_dataset_summary(build_profile(frame, "text.csv")[0]).available_metrics == ["successful_count"]


def test_personal_data_is_hidden_and_excluded_from_dimensions_and_measures():
    frame = pd.DataFrame(
        {
            "email": ["ivan@example.kz", "aida@mail.ru", "bolat@corp.com"],
            "Full Name": ["Иванов Иван", "Петрова Анна", "Сидоров Пётр"],
            "Transactions": [1, 2, 3],
            "Channel": ["Web", "Web", "API"],
        }
    )
    summary = build_dataset_summary(build_profile(frame, "clients.csv")[0])
    assert summary.sensitive_columns == ["email", "Full Name"]
    assert "email" not in summary.dimensions and "Full Name" not in summary.dimensions
    hidden = [c for c in summary.columns if c.is_sensitive]
    assert [c.name for c in hidden] == ["email", "Full Name"] and all(c.statistics is None for c in hidden)
    dump = summary.model_dump_json()
    for secret in ("ivan@example.kz", "aida@mail.ru", "Иванов", "Петрова", "Сидоров"):
        assert secret not in dump


def test_summary_is_json_serializable_and_roundtrips(demo):
    summary, _ = demo
    restored = type(summary).model_validate(json.loads(summary.model_dump_json()))
    assert restored == summary
