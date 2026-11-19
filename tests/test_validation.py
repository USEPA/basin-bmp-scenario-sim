from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.bmp import _get_bmp_selection_probs
from src.constants import DATA_CPS, DATA_PARCELS
import src.input_config as input_config
from src.input_config import _build_parcel_down_map
from src.model import Model


class DummyLogger:
    def log(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None
    def verbose(self, *args, **kwargs):
        return None


def test_load_parcel_selection_rejects_empty_parcels() -> None:
    with pytest.raises(ValueError, match="No parcels available"):
        input_config._load_parcel_selection({}, pd.DataFrame(columns=["pid"]), DummyLogger())


def test_load_parcel_selection_normalizes_selection_weights(monkeypatch) -> None:
    parcels = pd.DataFrame({"pid": [1, 2, 3]})
    weights = pd.DataFrame({"pid": [1, 2, 3], "value": [1.0, 3.0, 1.0]})
    monkeypatch.setattr(input_config, "_load_fixed_numeric_variable_table", lambda *args, **kwargs: weights)
    loaded = input_config._load_parcel_selection({}, parcels, DummyLogger())
    probs = dict(zip(loaded["pid"].astype(str), loaded["probability"]))
    assert probs == pytest.approx({"1": 0.2, "2": 0.6, "3": 0.2})


def test_load_parcel_selection_rejects_negative_weight(monkeypatch) -> None:
    parcels = pd.DataFrame({"pid": [1, 2]})
    weights = pd.DataFrame({"pid": [1, 2], "value": [1.0, -1.0]})
    monkeypatch.setattr(input_config, "_load_fixed_numeric_variable_table", lambda *args, **kwargs: weights)
    with pytest.raises(ValueError, match="finite and >= 0"):
        input_config._load_parcel_selection({}, parcels, DummyLogger())

def test_downstream_single_destination_and_reverse_in_memory():
    edges = pd.DataFrame({'pid':['4','5','9'], 'pid_down':['9','9','10']})
    mapping = _build_parcel_down_map(edges, ['4','5','9','10'])
    assert mapping == {'4':'9', '5':'9', '9':'10', '10':None}


def test_downstream_rejects_compound_ids():
    rows = pd.DataFrame({'pid':['9'], 'pid_down':['4,5']})
    with pytest.raises(ValueError, match='comma-separated'):
        _build_parcel_down_map(rows, ['4','5','9'])


def test_downstream_rejects_two_targets_even_when_duplicated():
    rows = pd.DataFrame({'pid':['9','9'], 'pid_down':['4','5']})
    with pytest.raises(ValueError, match='more than one downstream'):
        _build_parcel_down_map(rows, ['4','5','9'])


def test_downstream_unknown_ids_rejected():
    rows = pd.DataFrame({'pid':['9'], 'pid_down':['missing']})
    with pytest.raises(ValueError, match='missing'):
        _build_parcel_down_map(rows, ['4','9'])


def test_downstream_integer_ids():
    rows = pd.DataFrame({'pid':[1], 'pid_down':[2]})
    assert _build_parcel_down_map(rows, ['1','2']) == {'1':'2','2':None}


def test_get_bmp_selection_probs_rejects_invalid_probabilities(tmp_path) -> None:
    bmp_sel = tmp_path / "bmp_sel.csv"
    pd.DataFrame({"cps": [329, 412], "probability": [0.8, -0.2]}).to_csv(bmp_sel, index=False)

    model = SimpleNamespace(
        data={DATA_CPS: [329, 412]},
        cfg={},
        logger=DummyLogger(),
    )

    with pytest.raises(ValueError, match="nonnegative"):
        _get_bmp_selection_probs(model, str(bmp_sel))

def test_get_bmp_selection_probs_rejects_missing_cps_rows(tmp_path) -> None:
    bmp_sel = tmp_path / "bmp_sel.csv"
    pd.DataFrame({"cps": [329], "probability": [1.0]}).to_csv(bmp_sel, index=False)

    model = SimpleNamespace(
        data={DATA_CPS: [329, 412]},
        cfg={},
        logger=DummyLogger(),
    )

    with pytest.raises(ValueError, match="missing probability rows"):
        _get_bmp_selection_probs(model, str(bmp_sel))

def test_prepare_lookup_tables_rejects_duplicate_parcel_ids() -> None:
    model = Model.__new__(Model)
    model.data = {
        DATA_PARCELS: pd.DataFrame(
            {
                "pid": ["p1", "p1"],
                "area_ha": [1.0, 2.0],
                "perim_m": [10.0, 20.0],
            }
        )
    }
    model.logger = DummyLogger()

    with pytest.raises(ValueError, match="Duplicate parcel IDs"):
        Model._prepare_lookup_tables(model)


def test_config_rejects_legacy_per_file_parcel_and_outlet_keys() -> None:
    from src.input_validation import validate_config

    cfg = {
        "n_scenarios": 1,
        "buffer_depth_ft": 25.0,
        "bmp_fail_rate": 0.0,
        "bmp_fail_reduction": 0.25,
        "parcel_out": "parcel_out.csv",
    }
    with pytest.raises(ValueError, match="Legacy per-file parcel/outlet configuration keys"):
        validate_config(cfg)


def test_config_rejects_legacy_plet_file_keys() -> None:
    from src.input_validation import validate_config

    cfg = {
        "n_scenarios": 1,
        "buffer_depth_ft": 25.0,
        "bmp_fail_rate": 0.0,
        "bmp_fail_reduction": 0.25,
        "load_generation": {"mode": "plet_rusle", "plet_inputs": "plet_inputs.csv"},
    }
    with pytest.raises(ValueError, match="Legacy load_generation file keys"):
        validate_config(cfg)
