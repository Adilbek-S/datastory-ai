import asyncio

import pandas as pd
import pytest

from datastory.evaluation.pipeline import evaluate_result
from datastory.file_processing.loader import (
    UnsupportedFileError,
    file_kind,
    load_table,
)
from datastory.profiler.profiler import build_profile
from datastory.workflow.graph import run_analysis


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "region": ["Москва", "Москва", "Казань", "Казань"],
            "revenue": [100.0, 120.0, 90.0, None],
            "orders": [10, 12, 9, 11],
        }
    )


def test_file_kind():
    assert file_kind("a.CSV") == "table"
    assert file_kind("a.png") == "image"
    assert file_kind("a.pdf") == "pdf"
    assert file_kind("a.exe") == "unknown"


def test_load_csv_cp1251_and_semicolon():
    data = "город;выручка\nМосква;10\nКазань;20\n".encode("cp1251")
    out = load_table(data, "data.csv")
    assert list(out.columns) == ["город", "выручка"]
    assert len(out) == 2


def test_load_image_not_supported_yet():
    with pytest.raises(UnsupportedFileError):
        load_table(b"", "table.png")


def test_profile_kinds(df):
    profile, _ = build_profile(df, "test.xlsx")
    assert profile.row_count == 4 and profile.column_count == 4
    assert profile.missing_cell_count == 1
    assert profile.detected_date_columns == ["date"]
    assert profile.detected_category_columns == ["region"]
    assert profile.detected_numeric_columns == ["revenue", "orders"]


def test_workflow_and_evaluation(df):
    result = run_analysis(df)
    assert len(result.kpis) == 4
    assert {c.kind for c in result.charts} >= {"histogram", "bar", "line", "scatter"}
    assert all(case.passed for case in evaluate_result(result))


def test_mcp_tool_registered():
    from datastory.mcp_server.server import mcp

    tools = asyncio.run(mcp.list_tools())
    assert {"profile_file", "list_datasets", "get_dataset_profile", "search_business_context", "find_columns", "get_source"} <= {t.name for t in tools}
