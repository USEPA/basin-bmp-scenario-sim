from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.bmp import _get_bmp_selection_probs
from src.constants import CFG_PARCEL_P, DATA_CPS, DATA_PARCELS
from src.input_config import _build_parcel_up_map, _load_parcel_selection
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
        _load_parcel_selection({}, pd.DataFrame(columns=["pid"]), DummyLogger())


def test_load_parcel_selection_rejects_duplicate_selection_rows(tmp_path) -> None:
    parcel_p = tmp_path / "parcel_p.csv"
    pd.DataFrame({"pid": ["p1", "p1"], "probability": [0.4, 0.6]}).to_csv(parcel_p, index=False)
    cfg = {CFG_PARCEL_P: str(parcel_p)}
    parcels = pd.DataFrame({"pid": ["p1", "p2"]})

    with pytest.raises(ValueError, match="must contain one row per parcel"):
        _load_parcel_selection(cfg, parcels, DummyLogger())



def test_load_parcel_selection_supports_wildcard_default_and_exact_override(tmp_path) -> None:
    parcel_p = tmp_path / "parcel_p.csv"
    pd.DataFrame(
        {
            "pid": ["*", "p2"],
            "probability": [1.0, 3.0],
        }
    ).to_csv(parcel_p, index=False)
    cfg = {CFG_PARCEL_P: str(parcel_p)}
    parcels = pd.DataFrame({"pid": ["p1", "p2", "p3"]})

    loaded = _load_parcel_selection(cfg, parcels, DummyLogger())
    probs = dict(zip(loaded["pid"].astype(str), loaded["probability"]))

    assert probs == pytest.approx({"p1": 0.2, "p2": 0.6, "p3": 0.2})


def test_build_parcel_up_map_supports_blank_wildcard_default() -> None:
    upstream_rows = pd.DataFrame(
        {
            "pid": ["*", "9"],
            "pid_up": [None, "4,5"],
        }
    )

    parcel_up_map = _build_parcel_up_map(
        upstream_rows,
        parcel_ids=["4", "5", "9", "10"],
    )

    assert parcel_up_map == {
        "4": [],
        "5": [],
        "9": ["4", "5"],
        "10": [],
    }


def test_build_parcel_up_map_rejects_nonblank_wildcard_relationship() -> None:
    upstream_rows = pd.DataFrame(
        {
            "pid": ["*"],
            "pid_up": ["4"],
        }
    )

    with pytest.raises(ValueError, match="wildcard may only declare the default of no upstream parcels"):
        _build_parcel_up_map(upstream_rows, parcel_ids=["4", "9"])

def test_build_parcel_up_map_splits_trims_and_deduplicates_ids() -> None:
    upstream_rows = pd.DataFrame(
        {
            "pid": ["9", "9", "10", "11"],
            "pid_up": ["4, 5", "5,4", None, "  "],
        }
    )
    parcel_up_map = _build_parcel_up_map(
        upstream_rows,
        parcel_ids=["4", "5", "9", "10", "11"],
    )

    assert parcel_up_map == {
        "4": [],
        "5": [],
        "9": ["4", "5"],
        "10": [],
        "11": [],
    }


def test_build_parcel_up_map_rejects_unknown_upstream_ids() -> None:
    upstream_rows = pd.DataFrame(
        {
            "pid": ["9"],
            "pid_up": ["4,missing"],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        _build_parcel_up_map(upstream_rows, parcel_ids=["4", "9"])


def test_build_parcel_up_map_handles_numeric_cells_with_missing_values() -> None:
    upstream_rows = pd.DataFrame(
        {
            "pid": [1, 2],
            "pid_up": [2, None],
        }
    )

    parcel_up_map = _build_parcel_up_map(
        upstream_rows,
        parcel_ids=["1", "2"],
    )

    assert parcel_up_map == {"1": ["2"], "2": []}

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
