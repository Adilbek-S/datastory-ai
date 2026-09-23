"""Рабочее хранилище датасетов: сохранение, поиск по ID, интеграция с workflow и MCP."""
import pandas as pd
import pytest

from datastory.errors import DatasetNotFoundError
from datastory.profiler.profiler import build_profile
from datastory.storage.store import DatasetStore
from datastory.workflow.graph import run_analysis_for_dataset


@pytest.fixture
def store(tmp_path) -> DatasetStore:
    return DatasetStore(tmp_path / "workspace")


@pytest.fixture
def prepared():
    raw = pd.DataFrame(
        {
            "Month": ["2026-01", "2026-02", "2026-03", "2026-01", "2026-02", "2026-03"],
            "Channel": ["Web", "Web", "Web", "API", "API", "API"],
            "Amount": ["1 000,5", "2 000", "3 000,25", "500", "600", "700"],
            "Ok": ["да", "нет", "да", "да", "да", "нет"],
        }
    )
    return build_profile(raw, "sales.xlsx", "Q1")


def test_save_and_load_roundtrip_keeps_types(store, prepared):
    profile, typed = prepared
    store.save(typed, profile)
    loaded = store.load_dataframe(profile.dataset_id)
    pd.testing.assert_frame_equal(loaded, typed)
    assert pd.api.types.is_datetime64_any_dtype(loaded["Month"])
    assert pd.api.types.is_float_dtype(loaded["Amount"])
    assert str(loaded["Ok"].dtype) == "boolean"


def test_reference_lets_components_find_dataset_by_id(store, prepared):
    profile, typed = prepared
    ref = store.save(typed, profile)
    assert ref.dataset_id == profile.dataset_id
    assert (ref.filename, ref.sheet_name, ref.row_count, ref.column_count) == ("sales.xlsx", "Q1", 6, 4)
    assert ref.confirmed and ref.created_at.tzinfo is not None
    assert store.get(profile.dataset_id) == ref

    other = DatasetStore(store.root)  # новый экземпляр (как другой компонент/перезапуск) находит тот же датасет
    assert other.get(profile.dataset_id).storage_path == ref.storage_path
    assert other.load_profile(profile.dataset_id) == profile


def test_profile_description_is_persisted(store):
    profile, typed = build_profile(pd.DataFrame({"a": [1, 2]}), "t.csv", description="Тестовое описание")
    store.save(typed, profile)
    assert store.load_profile(profile.dataset_id).description == "Тестовое описание"


def test_list_newest_first_and_delete(store):
    ids = []
    for name in ("one.csv", "two.csv"):
        profile, typed = build_profile(pd.DataFrame({"a": [1, 2]}), name)
        store.save(typed, profile)
        ids.append(profile.dataset_id)
    assert [r.dataset_id for r in store.list()] == ids[::-1]

    store.delete(ids[0])
    assert [r.dataset_id for r in store.list()] == [ids[1]]
    with pytest.raises(DatasetNotFoundError):
        store.get(ids[0])
    with pytest.raises(DatasetNotFoundError):
        store.delete(ids[0])


def test_missing_and_invalid_ids(store):
    assert store.list() == []
    for bad in ("aaaaaaaaaaaa", "", "../etc", "..", "A" * 12, "abc"):
        with pytest.raises(DatasetNotFoundError):
            store.get(bad)
        with pytest.raises(DatasetNotFoundError):
            store.load_dataframe(bad)


def test_saving_same_id_replaces_dataset(store):
    profile, typed = build_profile(pd.DataFrame({"a": [1, 2]}), "t.csv")
    store.save(typed, profile)
    bigger, bigger_typed = build_profile(pd.DataFrame({"a": [1, 2, 3]}), "t.csv", dataset_id=profile.dataset_id)
    store.save(bigger_typed, bigger)
    assert len(store.load_dataframe(profile.dataset_id)) == 3
    assert len(store.list()) == 1
    assert not list((store.root / profile.dataset_id).glob("*.tmp"))


def test_workflow_analyzes_dataset_by_id(store, prepared):
    profile, typed = prepared
    store.save(typed, profile)
    result = run_analysis_for_dataset(profile.dataset_id, store)
    assert result.profile.dataset_id == profile.dataset_id
    assert result.profile.row_count == 6
    kinds = {c.kind for c in result.charts}
    assert {"histogram", "bar", "line"} <= kinds
    line = next(c for c in result.charts if c.kind == "line")
    assert line.x == "Month" and line.color == "Channel"


def test_workflow_unknown_id(store):
    with pytest.raises(DatasetNotFoundError):
        run_analysis_for_dataset("0123456789ab", store)


def test_mcp_tools_return_profile_by_id_without_personal_data(store, monkeypatch):
    from datastory.mcp_server import server

    frame = pd.DataFrame({"email": ["a@b.kz", "c@d.kz", "e@f.kz"], "amount": [1, 2, 3]})
    profile, typed = build_profile(frame, "clients.csv")
    store.save(typed, profile)
    monkeypatch.setattr(server, "DatasetStore", lambda: store)

    listed = server.list_datasets()
    assert [d["dataset_id"] for d in listed] == [profile.dataset_id]
    assert "storage_path" not in listed[0]

    llm = server.get_dataset_profile(profile.dataset_id)
    assert llm["row_count"] == 3 and llm["hidden_columns"] == ["email"]
    assert "a@b.kz" not in str(llm)
