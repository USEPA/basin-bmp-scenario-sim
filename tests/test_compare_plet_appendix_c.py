from pathlib import Path

import pandas as pd
import pytest

from utils.compare_plet_appendix_c import (
    ROOT,
    _runtime_concentrations,
    _select_pid,
    _single_cps_config,
)


def test_script_root_is_repository_root() -> None:
    assert ROOT == Path(__file__).resolve().parents[1]


def test_select_pid_uses_validated_runtime_selection() -> None:
    data = {
        "parcels": pd.DataFrame({"pid": [101, 202, 303]}),
        "parcel_p": pd.DataFrame(
            {
                "pid": [101, 202, 303],
                "probability": [0.0, 0.75, 0.25],
            }
        ),
    }

    assert _select_pid(data, None) == "202"
    assert _select_pid(data, "101") == "101"


def test_select_pid_rejects_parcels_outside_validated_input() -> None:
    data = {
        "parcels": pd.DataFrame({"pid": [101]}),
        "parcel_p": pd.DataFrame({"pid": [101], "probability": [1.0]}),
    }

    with pytest.raises(ValueError, match="not present in the validated parcels input"):
        _select_pid(data, "999")


def test_runtime_concentrations_use_realized_model_values() -> None:
    row = {
        "initial_concentration_tn_mg_l": 2.5,
        "initial_concentration_tp_mg_l": 0.4,
        "initial_groundwater_concentration_tn_mg_l": 1.2,
        "initial_groundwater_concentration_tp_mg_l": 0.2,
    }

    assert _runtime_concentrations(
        row, ["TN", "TP", "TSS"], "initial_concentration_"
    ) == {"TN": 2.5, "TP": 0.4}
    assert _runtime_concentrations(
        row, ["TN", "TP", "TSS"], "initial_groundwater_concentration_"
    ) == {"TN": 1.2, "TP": 0.2}


def test_runtime_concentrations_fail_if_a_realized_value_is_missing() -> None:
    with pytest.raises(ValueError, match="Missing realized runtime concentration"):
        _runtime_concentrations({}, ["TN"], "initial_concentration_")


def test_single_cps_config_uses_supported_runtime_overrides(tmp_path: Path) -> None:
    scenario_out = tmp_path / "scenario"
    cfg = _single_cps_config(
        {"cps": [329, 340], "outlets": "outlets.gpkg"},
        329,
        scenario_out,
    )

    assert cfg["cps"] == [329]
    assert cfg["n_scenarios"] == 1
    assert cfg["bmp_limit_n"] == 1
    assert cfg["outputs"] == str(scenario_out)
    assert not {"parcel_p", "parcel_down", "bmp_sel"}.intersection(cfg)
