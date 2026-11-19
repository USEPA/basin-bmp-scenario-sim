from __future__ import annotations

import pandas as pd
import pytest

from src.constants import (
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    COL_OID,
    COL_PID,
    COL_PID_DOWN,
    COL_PROBABILITY,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
)
import src.input_config as input_config
from src.input_config import (
    _build_parcel_down_map,
    _complete_delivery_ratio_defaults,
)


class RecordingLogger:
    def warning(self, message: str) -> None:
        pass

    def verbose(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass


def test_selection_weight_table_is_normalized(monkeypatch) -> None:
    parcels = pd.DataFrame({COL_PID: [1, 2, 3]})
    weights = pd.DataFrame({COL_PID: [1, 2, 3], "value": [1.0, 2.0, 1.0]})
    monkeypatch.setattr(input_config, "_load_fixed_numeric_variable_table", lambda *args, **kwargs: weights)
    loaded = input_config._load_parcel_selection({}, parcels, RecordingLogger())
    assert loaded[COL_PID].tolist() == [1, 2, 3]
    assert loaded[COL_PROBABILITY].tolist() == pytest.approx([0.25, 0.50, 0.25])


def test_missing_selection_weight_means_equal_weights(monkeypatch) -> None:
    parcels = pd.DataFrame({COL_PID: [1, 2, 3]})
    monkeypatch.setattr(input_config, "_load_fixed_numeric_variable_table", lambda *args, **kwargs: None)
    loaded = input_config._load_parcel_selection({}, parcels, RecordingLogger())
    assert loaded[COL_PROBABILITY].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_parcel_down_has_one_downstream_per_source():
    edges = pd.DataFrame([
        {COL_PID: 1, COL_PID_DOWN: 3},
        {COL_PID: 2, COL_PID_DOWN: 3},
        {COL_PID: 3, COL_PID_DOWN: 4},
    ])
    assert _build_parcel_down_map(edges, [1, 2, 3, 4]) == {
        "1": "3", "2": "3", "3": "4", "4": None,
    }


def test_parcel_down_rejects_legacy_compound_relationships():
    edges = pd.DataFrame([{COL_PID: 3, COL_PID_DOWN: "1,2"}])
    with pytest.raises(ValueError, match="comma-separated"):
        _build_parcel_down_map(edges, [1, 2, 3])


def test_missing_delivery_ratio_columns_receive_neutral_defaults() -> None:
    out = _complete_delivery_ratio_defaults(
        None,
        {"1": ["1"], "2": ["1", "2"]},
    )
    got = {
        (str(row[COL_PID]), str(row[COL_OID])): (
            float(row[COL_SDR_F_TO_S]),
            float(row[COL_SDR_S_TO_O]),
            float(row[COL_NDR_F_TO_S]),
            float(row[COL_NDR_S_TO_O]),
        )
        for _, row in out.iterrows()
    }
    assert got == {
        ("1", "1"): (1.0, 1.0, 1.0, 1.0),
        ("2", "1"): (1.0, 1.0, 1.0, 1.0),
        ("2", "2"): (1.0, 1.0, 1.0, 1.0),
    }
