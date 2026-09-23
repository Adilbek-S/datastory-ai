"""Проверки синтетических демо-данных и их генератора."""
import json
from pathlib import Path

import pandas as pd
import pymupdf
import pytest
from PIL import Image

from datastory.file_processing.loader import load_table
from datastory.models import ColumnKind
from datastory.profiler.profiler import profile_dataframe
from datastory.workflow.graph import run_analysis
from scripts import generate_demo_data as gen

REPO_DEMO = gen.DEFAULT_OUT
FILES = ("transactions_2026.xlsx", "business_metrics.pdf", "transactions_screenshot.png", "expected_metrics.json")


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    gen.generate_all(out)
    return out


@pytest.fixture(scope="module")
def df(generated) -> pd.DataFrame:
    return pd.read_excel(generated / "transactions_2026.xlsx")


@pytest.fixture(scope="module")
def expected(generated) -> dict:
    return json.loads((generated / "expected_metrics.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ датасет
def test_structure(df):
    assert list(df.columns) == gen.COLUMNS
    assert len(df) == 18
    assert set(df["Channel"]) == {"Mobile", "Web", "API"}
    assert list(df["Month"].unique()) == gen.MONTHS
    assert (df.groupby("Month")["Channel"].nunique() == 3).all()


def test_no_missing_values(df):
    assert not df.isna().any().any()
    assert (df[["Transactions", "Successful", "Failed", "Amount_KZT"]] > 0).all().all()


def test_transactions_equal_successful_plus_failed(df):
    assert (df["Transactions"] == df["Successful"] + df["Failed"]).all()


def test_march_success_rate_drop(df):
    rate = df.assign(sr=df["Successful"] / df["Transactions"] * 100)
    for channel, part in rate.groupby("Channel"):
        march = part.loc[part["Month"] == gen.DROP_MONTH, "sr"].iloc[0]
        others = part.loc[part["Month"] != gen.DROP_MONTH, "sr"]
        assert march < others.min() - 2, channel  # заметное снижение относительно всех других месяцев
        assert others.max() - others.min() < 1.0, channel  # в остальные месяцы — лишь небольшие колебания


def test_volumes_fluctuate_slightly(df):
    for _, part in df.groupby("Channel"):
        changes = part["Transactions"].pct_change().dropna().abs()
        assert changes.max() < 0.12 and changes.max() > 0.0


# ------------------------------------------------------------------ контрольные показатели
def test_expected_matches_independent_pandas_calc(df, expected):
    totals = expected["totals"]
    assert totals["transactions"] == df["Transactions"].sum()
    assert totals["success_rate_pct"] == pytest.approx(df["Successful"].sum() / df["Transactions"].sum() * 100, abs=1e-4)
    assert totals["avg_transaction_amount_kzt"] == pytest.approx(df["Amount_KZT"].sum() / df["Transactions"].sum(), abs=1e-3)

    for month, values in expected["by_month"].items():
        part = df[df["Month"] == month]
        assert values["success_rate_pct"] == pytest.approx(part["Successful"].sum() / part["Transactions"].sum() * 100, abs=1e-4)
    for channel, values in expected["by_channel"].items():
        part = df[df["Channel"] == channel]
        assert values["amount_kzt"] == part["Amount_KZT"].sum()


def test_expected_anomaly_and_leaders(expected):
    assert expected["anomaly"]["lowest_success_rate_month"] == "2026-03"
    assert expected["anomaly"]["march_vs_february_pp"] < -3
    assert expected["leaders"]["highest_transactions_channel"] == "Mobile"
    assert expected["leaders"]["highest_avg_amount_channel"] == "API"
    assert expected["leaders"]["highest_success_rate_channel"] == "API"
    assert expected["meta"]["synthetic"] is True


# ------------------------------------------------------------------ PDF
def test_pdf_is_readable_with_cyrillic(generated):
    with pymupdf.open(generated / "business_metrics.pdf") as doc:
        assert doc.page_count >= 1
        text = " ".join(page.get_text() for page in doc)
        fonts = [f for page in doc for f in page.get_fonts()]
    assert "Описание платёжной системы" in text
    assert "�" not in text
    assert fonts and all(f[2] == "Type0" for f in fonts)  # шрифты встроены, а не подменяются


def test_pdf_contains_required_definitions(generated):
    with pymupdf.open(generated / "business_metrics.pdf") as doc:
        text = " ".join(" ".join(page.get_text().split()) for page in doc)
    for phrase in (
        "Success Rate = Successful / Transactions × 100",
        "Average Transaction Amount = Amount_KZT / Transactions",
        "Transaction Volume",
        "Mobile", "Web", "API",
        "плановое обновление инфраструктуры обработки транзакций",
        "не доказывает причинную связь",
        "Transactions = Successful + Failed",
    ):
        assert phrase in text, phrase
    for column in gen.COLUMNS:
        assert column in text, column
    for number in range(1, 10):
        assert f"{number}. " in text


# ------------------------------------------------------------------ PNG
def test_png_created_and_usable(generated, expected):
    path = generated / "transactions_screenshot.png"
    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.width >= 1000 and img.height >= 300
        assert len(img.convert("L").getcolors(maxcolors=1 << 16)) > 3  # не пустой
    assert expected["screenshot"]["row_count"] == 6
    assert {r["Month"] for r in expected["screenshot"]["rows"]} == set(gen.SCREENSHOT_MONTHS)


# ------------------------------------------------------------------ воспроизводимость
def test_generator_is_rerunnable_and_deterministic(tmp_path, generated):
    gen.generate_all(tmp_path)
    gen.generate_all(tmp_path)  # повторный запуск поверх существующих файлов не падает
    for name in FILES:
        assert (tmp_path / name).read_bytes() == (generated / name).read_bytes(), name


def test_committed_demo_files_are_up_to_date(generated):
    for name in FILES:
        committed = REPO_DEMO / name
        assert committed.exists(), f"{name} не найден в data/demo — запустите scripts/generate_demo_data.py"
        assert committed.read_bytes() == (generated / name).read_bytes(), f"{name} устарел — перегенерируйте"


# ------------------------------------------------------------------ совместимость с приложением
def test_app_pipeline_on_demo_xlsx():
    frame = load_table((REPO_DEMO / "transactions_2026.xlsx").read_bytes(), "transactions_2026.xlsx")
    profile = profile_dataframe(frame)
    assert profile.columns_of(ColumnKind.NUMERIC) == ["Transactions", "Successful", "Failed", "Amount_KZT"]
    result = run_analysis(frame)
    assert result.profile.rows == 18 and result.kpis
