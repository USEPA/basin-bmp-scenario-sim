from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.io_utils import read_config, read_csv_table, read_csv_tables


def test_read_config_returns_empty_mapping_for_empty_yaml(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    cfg = read_config(path)

    assert cfg == {}


def test_read_config_reads_yaml_mapping(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"outputs": "out", "n_scenarios": 5}, sort_keys=False),
        encoding="utf-8",
    )

    cfg = read_config(path)

    assert cfg == {"outputs": "out", "n_scenarios": 5}


def test_read_config_rejects_non_mapping_yaml_root(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(["a", "b"]), encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        read_config(path)


def test_read_config_raises_for_missing_file(tmp_path) -> None:
    path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Input YAML file not found"):
        read_config(path)


def test_read_csv_table_reads_single_csv(tmp_path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(path, index=False)

    df = read_csv_table(path)

    assert list(df.columns) == ["a", "b"]
    assert df.to_dict(orient="records") == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_read_csv_tables_accepts_single_path_and_returns_single_frame_list(tmp_path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)

    frames = read_csv_tables(path)

    assert isinstance(frames, list)
    assert len(frames) == 1
    assert frames[0].to_dict(orient="records") == [{"a": 1}]


def test_read_csv_tables_preserves_input_order_for_multiple_paths(tmp_path) -> None:
    path1 = tmp_path / "a.csv"
    path2 = tmp_path / "b.csv"
    pd.DataFrame({"name": ["first"]}).to_csv(path1, index=False)
    pd.DataFrame({"name": ["second"]}).to_csv(path2, index=False)

    frames = read_csv_tables([path1, path2])

    assert len(frames) == 2
    assert frames[0].iloc[0]["name"] == "first"
    assert frames[1].iloc[0]["name"] == "second"