"""Small deterministic golden regression for the complete PLET scenario path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    CFG_BMP_COST,
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BUFFER_DEPTH_FT,
    CFG_OUTPUTS,
    CFG_VERBOSE,
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_BMP_COST,
    INCH_OVER_HA_TO_LITERS,
    TON_PER_ACRE_TO_KG_PER_HA,
)
from src.model import _run_one_scenario


def _fixed_plet_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Return fixed PLET/RUSLE and concentration tables for the golden case."""
    plet = pd.DataFrame(
        [
            {"pid": "*", "parameter": "annual_precip_in", "value": 40.0},
            {"pid": "*", "parameter": "rain_days", "value": 100.0},
            {"pid": "*", "parameter": "rain_correction_fraction", "value": 0.80},
            {"pid": "*", "parameter": "runoff_day_fraction", "value": 0.25},
            {"pid": "*", "parameter": "ia_ratio", "value": 0.20},
            {"pid": "*", "parameter": "land_cover", "value": "cropland"},
            {"pid": "*", "parameter": "hsg", "value": "B"},
        ]
    )
    hydrology = pd.DataFrame(
        [
            {
                "land_cover": "cropland",
                "hsg": "B",
                "parameter": "cn",
                "value": 80.0,
            },
            {
                "land_cover": "cropland",
                "hsg": "B",
                "parameter": "infiltration_fraction",
                "value": 0.30,
            },
        ]
    )
    rusle = pd.DataFrame(
        [
            {"pid": "*", "parameter": "r", "value": 100.0},
            {"pid": "*", "parameter": "k", "value": 0.20},
            {"pid": "*", "parameter": "ls", "value": 1.5},
            {"pid": "*", "parameter": "c", "value": 0.10},
            {"pid": "*", "parameter": "p", "value": 0.50},
            {"pid": "*", "parameter": "sdr", "value": 0.40},
            {"pid": "*", "parameter": "sediment_n_pct", "value": 1.0},
            {"pid": "*", "parameter": "sediment_p_pct", "value": 0.50},
            {"pid": "*", "parameter": "enrichment_ratio", "value": 2.0},
        ]
    )
    runoff_concentrations = pd.DataFrame(
        [
            {"pid": "*", "pollutant": "TN", "value": 2.0},
            {"pid": "*", "pollutant": "TP", "value": 0.2},
        ]
    )
    groundwater_concentrations = pd.DataFrame(
        [
            {"pid": "*", "pollutant": "TN", "value": 3.0},
            {"pid": "*", "pollutant": "TP", "value": 0.3},
        ]
    )
    return plet, hydrology, rusle, runoff_concentrations, groundwater_concentrations


def _shared_payload() -> dict:
    plet, hydrology, rusle, runoff_concentrations, groundwater_concentrations = (
        _fixed_plet_tables()
    )
    delivery = {
        ("P1", "O1"): {
            COL_SDR_F_TO_S: 1.0,
            COL_SDR_S_TO_O: 1.0,
            COL_NDR_F_TO_S: 1.0,
            COL_NDR_S_TO_O: 1.0,
        }
    }
    bmp_cost = pd.DataFrame(
        [{"cps": 340, "unit": "usd/project", "value": 50.0}]
    )
    data = {
        DATA_BMP_COST: bmp_cost,
        DATA_AVG_AREA_HA: 1.0,
        DATA_AVG_PERIM_M: 100.0,
    }
    return {
        "data": data,
        "parcel_ids": ["P1"],
        "pid_to_index": {"P1": 0},
        "pollutants": ["TN", "TP", "TSS"],
        "parcel_area_ha": np.asarray([1.0]),
        "parcel_perim_m": np.asarray([100.0]),
        "parcel_out_oids": [["O1"]],
        "parcel_up_idxs": [[]],
        "parcel_selection_ids": ["P1"],
        "parcel_selection_probs": np.asarray([1.0]),
        "selection_source_idxs": np.asarray([0]),
        "outlet_oids": ["O1"],
        "outlet_target_map": {},
        "outlet_mean_map": {},
        "delivery_coeffs": delivery,
        "bmp_efficiency_stats": {
            340: [
                {
                    "surface": {"value": 0.50},
                    "subsurface": {"value": 0.0},
                },
                {
                    "surface": {"value": 0.25},
                    "subsurface": {"value": 0.10},
                },
                {
                    "surface": {"value": 0.40},
                    "subsurface": {"value": 0.0},
                },
            ]
        },
        "pollutant_load_rate_stats": None,
        "load_generation": {
            "mode": "plet_rusle",
            "_hydrology_lookup_table": hydrology,
        },
        "load_generation_mode": "plet_rusle",
        "plet_inputs": plet,
        "rusle_inputs": rusle,
        "pollutant_concentrations": runoff_concentrations,
        "groundwater_concentrations": groundwater_concentrations,
        "pathway_names": ["surface", "subsurface"],
        "pollutant_load_rate_pathway_fractions": {},
        "pollutant_load_rate_is_aggregate": False,
        "groundwater_loads": True,
        "bmp_cps": [340],
        "bmp_selection_probs": np.asarray([1.0]),
        "avg_area_ha": 1.0,
        "avg_perim_m": 100.0,
        "random_seed": 12345,
    }


def _config(outputs: Path) -> dict:
    return {
        CFG_OUTPUTS: str(outputs),
        CFG_VERBOSE: False,
        CFG_BMP_COST: "synthetic_bmp_cost.csv",
        CFG_BMP_FAIL_RATE: 0.0,
        CFG_BMP_FAIL_REDUCTION: 0.5,
        CFG_BUFFER_DEPTH_FT: 10.0,
        "bmp_limit_n": 1,
        "bmp_limit_usd": None,
    }


def test_deterministic_plet_scenario_matches_golden_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise PLET, RUSLE, BMP, cost, routing, summary, and output logic."""
    outputs = tmp_path / "outputs"
    written: dict[tuple[str, str], pd.DataFrame] = {}

    def capture_parquet(df: pd.DataFrame, path: Path, *, logger) -> None:
        del logger
        written[(path.parent.name, path.name)] = df.copy()

    monkeypatch.setattr("src.model._write_parquet_atomic", capture_parquet)

    records = _run_one_scenario(
        _shared_payload(),
        _config(outputs),
        sidx=0,
        seed=12345,
        outputs_dir=outputs,
    )

    runoff_days = 100.0 * 0.25
    event_rainfall = 40.0 * 0.80 / runoff_days
    retention = (1000.0 / 80.0) - 10.0
    initial_abstraction = 0.20 * retention
    event_runoff = (
        (event_rainfall - initial_abstraction) ** 2
        / (event_rainfall - initial_abstraction + retention)
    )
    annual_runoff = event_runoff * runoff_days
    annual_infiltration = 40.0 * 0.80 * 0.30
    runoff_l_ha = annual_runoff * INCH_OVER_HA_TO_LITERS
    infiltration_l_ha = annual_infiltration * INCH_OVER_HA_TO_LITERS

    sediment = (
        100.0
        * 0.20
        * 1.5
        * 0.10
        * 0.50
        * 0.40
        * TON_PER_ACRE_TO_KG_PER_HA
    )
    surface = {
        "TN": 2.0 * runoff_l_ha / 1_000_000.0 + sediment * 0.01 * 2.0,
        "TP": 0.2 * runoff_l_ha / 1_000_000.0 + sediment * 0.005 * 2.0,
        "TSS": sediment,
    }
    subsurface = {
        "TN": 3.0 * infiltration_l_ha / 1_000_000.0,
        "TP": 0.3 * infiltration_l_ha / 1_000_000.0,
        "TSS": 0.0,
    }
    efficiencies = {
        "TN": (0.50, 0.0),
        "TP": (0.25, 0.10),
        "TSS": (0.40, 0.0),
    }
    baseline = {pol: surface[pol] + subsurface[pol] for pol in surface}
    removed = {
        pol: surface[pol] * efficiencies[pol][0]
        + subsurface[pol] * efficiencies[pol][1]
        for pol in surface
    }
    final = {pol: baseline[pol] - removed[pol] for pol in surface}

    bmps = written[("bmps", "s1.parquet")]
    parcels = written[("parcels", "s1.parquet")]
    parameters = written[("load_parameters", "s1.parquet")]
    metrics = written[("scenario_metrics", "s1.parquet")]

    # Deterministic management decisions and cost.
    assert bmps["cps"].astype(int).tolist() == [340]
    assert bmps["pid"].astype(str).tolist() == ["P1"]
    assert float(bmps.loc[0, "cost_usd"]) == pytest.approx(50.0)

    for pol in ("TN", "TP", "TSS"):
        assert float(bmps.loc[0, f"baseline_mass_{pol}_kg"]) == pytest.approx(
            baseline[pol]
        )
        assert float(bmps.loc[0, f"removed_mass_{pol}_kg"]) == pytest.approx(
            removed[pol]
        )
        assert float(parcels.loc[0, f"baseline_load_rate_{pol}_kg_ha_yr"]) == pytest.approx(
            baseline[pol]
        )
        assert float(parcels.loc[0, f"final_load_rate_{pol}_kg_ha_yr"]) == pytest.approx(
            final[pol]
        )

        label = pol.lower()
        assert float(
            parameters.loc[0, f"initial_surface_{label}_load_rate_kg_ha_yr"]
        ) == pytest.approx(surface[pol])
        assert float(
            parameters.loc[0, f"initial_subsurface_{label}_load_rate_kg_ha_yr"]
        ) == pytest.approx(subsurface[pol])
        assert float(
            parameters.loc[0, f"final_surface_{label}_load_rate_kg_ha_yr"]
        ) == pytest.approx(surface[pol] * (1.0 - efficiencies[pol][0]))
        assert float(
            parameters.loc[0, f"final_subsurface_{label}_load_rate_kg_ha_yr"]
        ) == pytest.approx(subsurface[pol] * (1.0 - efficiencies[pol][1]))

    assert float(parameters.loc[0, "initial_annual_runoff_in"]) == pytest.approx(
        annual_runoff
    )
    assert float(parameters.loc[0, "initial_annual_infiltration_in"]) == pytest.approx(
        annual_infiltration
    )
    assert float(parameters.loc[0, "initial_sediment_load_rate_kg_ha_yr"]) == pytest.approx(
        sediment
    )

    rollup = metrics[metrics["cps"].astype(int) == 0].iloc[0]
    assert int(rollup["bmp_count"]) == 1
    assert float(rollup["total_cost_usd"]) == pytest.approx(50.0)
    for pol in ("TN", "TP", "TSS"):
        assert float(rollup[f"removed_mass_{pol}_kg_total"]) == pytest.approx(
            removed[pol]
        )

        count_record = records[(pol, "O1", "count", "total")]
        cost_record = records[(pol, "O1", "cost", "total")]
        assert count_record == pytest.approx([(1, 1.0, removed[pol])])
        assert cost_record == pytest.approx([(1, 50.0, removed[pol])])
