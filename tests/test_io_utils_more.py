from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.io_utils import read_config, read_csv_table, read_csv_tables


def test_read_config_accepts_string_path_argument(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"outputs": "abc"}), encoding="utf-8")

    cfg = read_config(str(path))

    assert cfg == {"outputs": "abc"}


def test_read_config_preserves_nested_mapping_structure(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    payload = {
        "outputs": "out",
        "parallel": {"n_jobs": 2, "temp_folder": "/tmp/work"},
        "load_generation": {"mode": "plet_rusle"},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    cfg = read_config(path)

    assert cfg == payload
    assert isinstance(cfg["parallel"], dict)
    assert isinstance(cfg["load_generation"], dict)


def test_read_csv_table_accepts_string_path_argument(tmp_path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)

    df = read_csv_table(str(path))

    assert df.to_dict(orient="records") == [{"x": 1}, {"x": 2}]


def test_read_csv_tables_accepts_tuple_of_paths(tmp_path) -> None:
    path1 = tmp_path / "a.csv"
    path2 = tmp_path / "b.csv"
    pd.DataFrame({"id": [1]}).to_csv(path1, index=False)
    pd.DataFrame({"id": [2]}).to_csv(path2, index=False)

    frames = read_csv_tables((path1, path2))

    assert len(frames) == 2
    assert frames[0].iloc[0]["id"] == 1
    assert frames[1].iloc[0]["id"] == 2


def test_read_csv_tables_returns_empty_list_for_empty_sequence() -> None:
    frames = read_csv_tables([])

    assert frames == []


def test_read_csv_tables_propagates_missing_file_error(tmp_path) -> None:
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        read_csv_tables([path])