"""Краткий профиль для LLM: компактность и отсутствие персональных данных."""
import numpy as np
import pandas as pd
import pytest

from datastory.profiler.llm_profile import REDACTED, LLMDatasetProfile, build_llm_profile
from datastory.profiler.profiler import build_profile

EMAILS = ["ivan.petrov@example.kz", "aida.s@mail.ru", "bolat@corp.com", "dana@site.kz", "erlan@x.org", "fariza@y.kz"]
PHONES = ["+7 (701) 111-22-33", "+7 (702) 222-33-44", "+7 (705) 333-44-55", "+7 (707) 444-55-66", "+7 (708) 555-66-77", "+7 (747) 666-77-88"]
NAMES = ["Иванов Иван Иванович", "Петрова Анна Сергеевна", "Сидоров Пётр Олегович", "Ким Дана Ермековна", "Ахметов Бахыт Нурланович", "Жаксылык Айгуль Маратовна"]
NOTES = [f"клиент просил перезвонить после {h}:00 по личному вопросу" for h in range(6)]


@pytest.fixture
def pii_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email": EMAILS,
            "contact": PHONES,  # безымянная колонка: телефоны распознаются по значениям
            "Full Name": NAMES,
            "notes": NOTES,
            "city": ["Almaty", "Astana", "Almaty", "Astana", "Almaty", "Shymkent"],
            "amount": [1500.5, 2300, 870.25, 4100, 990, 1200],
            "day": pd.date_range("2026-01-01", periods=6),
        }
    )


def test_prompt_contains_structure_but_no_personal_data(pii_frame):
    profile, _ = build_profile(pii_frame, "clients.xlsx")
    prompt = build_llm_profile(profile).to_prompt()

    for secret in EMAILS + PHONES + NAMES + NOTES:
        assert secret not in prompt, secret
    assert "clients.xlsx" not in prompt  # имя файла тоже может содержать персональные данные

    for name in ("email", "contact", "Full Name", "notes", "city", "amount", "day"):
        assert name in prompt
    assert "6 строк" in prompt and "7 колонок" in prompt
    assert "Almaty" in prompt  # обычные категории передаются
    assert "amount (numeric" in prompt and "day (datetime" in prompt


def test_sensitive_columns_are_hidden_completely(pii_frame):
    profile, _ = build_profile(pii_frame, "c.xlsx")
    llm = build_llm_profile(profile)
    assert set(llm.hidden_columns) == {"email", "contact", "Full Name"}
    for col in llm.columns:
        if col.name in llm.hidden_columns:
            assert not col.sample_values and not col.stats and not col.top_values
            assert col.unique_count is None and "скрыты" in col.note


def test_free_text_examples_are_not_sent(pii_frame):
    profile, _ = build_profile(pii_frame, "c.xlsx")
    notes = next(c for c in build_llm_profile(profile).columns if c.name == "notes")
    assert notes.kind == "text" and not notes.sample_values


def test_user_marked_column_is_hidden(pii_frame):
    profile, _ = build_profile(pii_frame, "c.xlsx", sensitive_overrides={"city": True})
    prompt = build_llm_profile(profile).to_prompt()
    assert "Almaty" not in prompt and "city" in prompt


def test_pii_like_values_in_regular_column_are_masked():
    frame = pd.DataFrame({"tag": ["u@x.kz", "u@x.kz", "alpha", "alpha", "beta", "beta", "gamma", "gamma", "delta", "delta"]})
    profile, _ = build_profile(frame, "t.csv")
    assert not profile.column("tag").is_sensitive  # 20% значений — ниже порога автоопределения
    llm = build_llm_profile(profile)
    prompt = llm.to_prompt()
    assert "u@x.kz" not in prompt and REDACTED in prompt
    assert "alpha" in prompt


def test_samples_are_limited_and_numeric_stats_present():
    frame = pd.DataFrame({"v": np.arange(1, 501), "cat": ["a", "b", "c", "d", "e"] * 100})
    llm = build_llm_profile(build_profile(frame, "t.csv")[0], max_samples=2)
    col = {c.name: c for c in llm.columns}
    assert len(col["v"].sample_values) <= 2 and len(col["cat"].top_values) <= 2
    assert col["v"].stats["min"] == 1 and col["v"].stats["max"] == 500 and col["v"].stats["mean"] == 250.5


def test_profile_size_does_not_grow_with_row_count():
    def prompt_for(rows: int) -> str:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"amount": rng.normal(1000, 100, rows).round(2), "kind": rng.choice(["a", "b", "c"], rows)})
        return build_llm_profile(build_profile(frame, "t.csv")[0]).to_prompt()

    small, large = prompt_for(20), prompt_for(20_000)
    assert len(large) < len(small) * 1.6 and len(large) < 1500  # в промпт не попадают строки таблицы


def test_llm_profile_is_built_from_profile_only(pii_frame):
    """Функция принимает DatasetProfile: DataFrame с данными в неё передать нельзя."""
    with pytest.raises(AttributeError):
        build_llm_profile(pii_frame)  # type: ignore[arg-type]
    profile, _ = build_profile(pii_frame, "c.xlsx")
    assert isinstance(build_llm_profile(profile), LLMDatasetProfile)


def test_quality_summary_has_codes_without_values():
    frame = pd.DataFrame({"v": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "секретное-значение"]})
    llm = build_llm_profile(build_profile(frame, "t.csv")[0])
    assert any(line.startswith("invalid_numeric [v]") for line in llm.quality_summary)
    assert "секретное" not in llm.to_prompt()
