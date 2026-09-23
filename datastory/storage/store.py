"""Рабочее хранилище подтверждённых датасетов (файлы на диске, без внешней БД).

Структура:  <root>/<dataset_id>/{data.parquet, profile.json, reference.json}
Другие компоненты находят датасет по dataset_id через DatasetStore.get / load_dataframe.
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from datastory.config import get_settings
from datastory.errors import DatasetNotFoundError
from datastory.models import DatasetProfile, DatasetReference

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
DATA_FILE, PROFILE_FILE, REFERENCE_FILE = "data.parquet", "profile.json", "reference.json"


def _check_id(dataset_id: str) -> str:
    if not _ID_RE.fullmatch(dataset_id or ""):
        raise DatasetNotFoundError(f"Некорректный идентификатор датасета: {dataset_id!r}")
    return dataset_id


class DatasetStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else get_settings().workspace_dir

    def _dir(self, dataset_id: str) -> Path:
        return self.root / _check_id(dataset_id)

    def save(self, df: pd.DataFrame, profile: DatasetProfile, *, confirmed: bool = True) -> DatasetReference:
        """Сохраняет таблицу (с типами из профиля) и профиль; возвращает ссылку на датасет."""
        folder = self._dir(profile.dataset_id)
        folder.mkdir(parents=True, exist_ok=True)

        data_path = folder / DATA_FILE
        tmp_path = folder / (DATA_FILE + ".tmp")
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(data_path)  # атомарная замена: не оставляем недописанный файл

        reference = DatasetReference(
            dataset_id=profile.dataset_id,
            filename=profile.filename,
            sheet_name=profile.sheet_name,
            row_count=profile.row_count,
            column_count=profile.column_count,
            storage_path=str(data_path),
            created_at=datetime.now(timezone.utc),
            confirmed=confirmed,
        )
        (folder / PROFILE_FILE).write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        (folder / REFERENCE_FILE).write_text(reference.model_dump_json(indent=2), encoding="utf-8")
        return reference

    def get(self, dataset_id: str) -> DatasetReference:
        path = self._dir(dataset_id) / REFERENCE_FILE
        if not path.exists():
            raise DatasetNotFoundError(f"Датасет {dataset_id} не найден в рабочем хранилище.")
        return DatasetReference.model_validate_json(path.read_text(encoding="utf-8"))

    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        self.get(dataset_id)
        return pd.read_parquet(self._dir(dataset_id) / DATA_FILE)

    def load_profile(self, dataset_id: str) -> DatasetProfile:
        self.get(dataset_id)
        return DatasetProfile.model_validate_json((self._dir(dataset_id) / PROFILE_FILE).read_text(encoding="utf-8"))

    def list(self) -> list[DatasetReference]:
        if not self.root.exists():
            return []
        refs = []
        for folder in self.root.iterdir():
            if _ID_RE.fullmatch(folder.name) and (folder / REFERENCE_FILE).exists():
                refs.append(self.get(folder.name))
        return sorted(refs, key=lambda r: r.created_at, reverse=True)

    def delete(self, dataset_id: str) -> None:
        folder = self._dir(dataset_id)
        if not folder.exists():
            raise DatasetNotFoundError(f"Датасет {dataset_id} не найден в рабочем хранилище.")
        shutil.rmtree(folder)
