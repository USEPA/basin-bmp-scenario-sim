"""Reusable synthetic model cases for tests that exercise scenario execution.

These builders intentionally bypass YAML and filesystem input discovery.  They
construct the normalized ``cfg`` and ``data`` mappings consumed by ``Model`` so
model/reproducibility tests stay independent of any shipped example dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import (
    CFG_BMP_COST,
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_BMP_SEL_PROB_VIA_COSTS,
    CFG_BUFFER_DEPTH_FT,
    CFG_OUTPUTS,
    CFG_PARALLEL,
    CFG_VERBOSE,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_BMP_COST,
    DATA_BMP_EFFICIENCY,
    DATA_CPS,
    DATA_LOAD_GENERATION,
    DATA_N_SCENARIOS,
    DATA_OUTLET_LOC,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_PARCELS,
    DATA_PARCEL_OUT_MAP,
    DATA_PARCEL_P,
    DATA_PARCEL_DOWN_MAP,
    DATA_PATHWAYS,
    DATA_PLET_INPUTS,
    DATA_POLLUTANTS,
    DATA_POLLUTANT_CONCENTRATIONS,
    DATA_POLLUTANT_LOAD_RATE,
    DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE,
    DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    DATA_RANDOM_SEED,
    DATA_RUSLE_INPUTS,
    DATA_GROUNDWATER_CONCENTRATIONS,
    LOAD_MODE_PLET_RUSLE,
    LOAD_MODE_STATISTICAL,
)


def _base_cfg(
    outputs: Path,
    *,
    n_jobs: int,
    bmp_limit_n: int,
) -> dict[str, Any]:
    return {
        CFG_OUTPUTS: str(Path(outputs).resolve()),
        CFG_VERBOSE: False,
        CFG_PARALLEL: {"n_jobs": int(n_jobs)},
        CFG_BMP_LIMIT_N: int(bmp_limit_n),
        CFG_BMP_LIMIT_USD: None,
        CFG_BMP_COST: None,
        CFG_BMP_SEL_PROB_VIA_COSTS: False,
        CFG_BMP_FAIL_RATE: 0.0,
        CFG_BMP_FAIL_REDUCTION: 0.5,
        CFG_BUFFER_DEPTH_FT: 10.0,
    }


def _base_spatial_and_routing() -> dict[str, Any]:
    parcel_ids = ["P1", "P2", "P3"]
    parcels = pd.DataFrame(
        {
            "pid": parcel_ids,
            "area_ha": [1.0, 1.5, 0.8],
            "perim_m": [100.0, 130.0, 90.0],
        }
    )
    delivery = pd.DataFrame(
        [
            {
                "pid": pid,
                "oid": "O1",
                "sdr_f_to_s": 1.0,
                "sdr_s_to_o": 1.0,
                "ndr_f_to_s": 1.0,
                "ndr_s_to_o": 1.0,
            }
            for pid in parcel_ids
        ]
    )
    return {
        DATA_PARCELS: parcels,
        DATA_PARCEL_P: pd.DataFrame(
            {
                "pid": parcel_ids,
                "probability": [0.20, 0.35, 0.45],
            }
        ),
        DATA_PARCEL_DOWN_MAP: {pid: None for pid in parcel_ids},
        DATA_PARCEL_OUT_MAP: {pid: ["O1"] for pid in parcel_ids},
        DATA_OUTLET_LOC: pd.DataFrame({"oid": ["O1"]}),
        DATA_OUTLET_TARGET: None,
        DATA_OUTLET_MEAN: None,
        "delivery_ratios": delivery,
        DATA_AVG_AREA_HA: float(parcels["area_ha"].mean()),
        DATA_AVG_PERIM_M: float(parcels["perim_m"].mean()),
        DATA_BMP_COST: None,
    }


def make_statistical_case(
    outputs: Path,
    *,
    n_jobs: int = 1,
    n_scenarios: int = 2,
    random_seed: int = 202501,
    bmp_limit_n: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a complete synthetic statistical-mode model case."""
    cfg = _base_cfg(outputs, n_jobs=n_jobs, bmp_limit_n=bmp_limit_n)
    data = _base_spatial_and_routing()

    pollutants = ["TN", "TP"]
    pathways = ["surface", "subsurface"]
    cps_codes = [340, 345]

    load_rows = []
    for pid, tn, tp in (
        ("P1", 10.0, 2.0),
        ("P2", 14.0, 2.8),
        ("P3", 8.0, 1.6),
    ):
        load_rows.extend(
            [
                {"pid": pid, "pollutant": "TN", "value": tn},
                {"pid": pid, "pollutant": "TP", "value": tp},
            ]
        )

    efficiency_rows = []
    values = {
        340: {"TN": (0.30, 0.10), "TP": (0.25, 0.05)},
        345: {"TN": (0.20, 0.15), "TP": (0.15, 0.10)},
    }
    for cps, by_pollutant in values.items():
        for pollutant, (surface, subsurface) in by_pollutant.items():
            efficiency_rows.extend(
                [
                    {
                        "cps": cps,
                        "pollutant": pollutant,
                        "pathway": "surface",
                        "value": surface,
                    },
                    {
                        "cps": cps,
                        "pollutant": pollutant,
                        "pathway": "subsurface",
                        "value": subsurface,
                    },
                ]
            )

    data.update(
        {
            DATA_RANDOM_SEED: int(random_seed),
            DATA_N_SCENARIOS: int(n_scenarios),
            DATA_POLLUTANTS: pollutants,
            DATA_CPS: cps_codes,
            DATA_BMP_EFFICIENCY: pd.DataFrame(efficiency_rows),
            DATA_POLLUTANT_LOAD_RATE: pd.DataFrame(load_rows),
            DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE: True,
            DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS: {
                "surface": 0.70,
                "subsurface": 0.30,
            },
            DATA_PATHWAYS: pathways,
            DATA_LOAD_GENERATION: {"mode": LOAD_MODE_STATISTICAL},
        }
    )
    return cfg, data


def make_plet_rusle_case(
    outputs: Path,
    *,
    n_jobs: int = 1,
    n_scenarios: int = 2,
    random_seed: int = 202503,
    bmp_limit_n: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a complete synthetic PLET/RUSLE model case.

    The hydrology lookup is an in-memory user-input manifestation.  Its values
    are intentionally different from common PLET defaults so tests verify the
    configured table is used rather than any hidden source-level fallback.
    """
    cfg = _base_cfg(outputs, n_jobs=n_jobs, bmp_limit_n=bmp_limit_n)
    data = _base_spatial_and_routing()

    pollutants = ["TN", "TP", "TSS"]
    cps_codes = [340, 345]

    plet = pd.DataFrame(
        [
            {"pid": None, "parameter": "annual_precip_in", "value": 38.0},
            {"pid": None, "parameter": "rain_days", "value": 95.0},
            {"pid": None, "parameter": "rain_correction_fraction", "value": 0.82},
            {"pid": None, "parameter": "runoff_day_fraction", "value": 0.27},
            {"pid": None, "parameter": "ia_ratio", "value": 0.20},
            {"pid": None, "parameter": "land_cover", "value": "cropland"},
            {"pid": None, "parameter": "hsg", "value": "B"},
        ]
    )
    hydrology = pd.DataFrame(
        [
            {
                "land_cover": "cropland",
                "hsg": "B",
                "parameter": "cn",
                "value": 73.0,
            },
            {
                "land_cover": "cropland",
                "hsg": "B",
                "parameter": "infiltration_fraction",
                "value": 0.37,
            },
        ]
    )
    rusle = pd.DataFrame(
        [
            {"pid": None, "parameter": "r", "value": 100.0},
            {"pid": None, "parameter": "k", "value": 0.20},
            {"pid": None, "parameter": "ls", "value": 1.2},
            {"pid": None, "parameter": "c", "value": 0.10},
            {"pid": None, "parameter": "p", "value": 0.50},
            {"pid": None, "parameter": "sdr", "value": 0.40},
            {"pid": None, "parameter": "sediment_n_pct", "value": 1.0},
            {"pid": None, "parameter": "sediment_p_pct", "value": 0.50},
            {"pid": None, "parameter": "enrichment_ratio", "value": 2.0},
        ]
    )
    surface_concentrations = pd.DataFrame(
        [
            {"pid": None, "pollutant": "TN", "value": 2.2},
            {"pid": None, "pollutant": "TP", "value": 0.22},
        ]
    )
    subsurface_concentrations = pd.DataFrame(
        [
            {"pid": None, "pollutant": "TN", "value": 3.1},
            {"pid": None, "pollutant": "TP", "value": 0.31},
        ]
    )

    efficiency_rows = []
    values = {
        340: {
            "TN": (0.35, 0.00),
            "TP": (0.25, 0.10),
            "TSS": (0.40, 0.00),
        },
        345: {
            "TN": (0.20, 0.05),
            "TP": (0.15, 0.05),
            "TSS": (0.25, 0.00),
        },
    }
    for cps, by_pollutant in values.items():
        for pollutant, (surface, subsurface) in by_pollutant.items():
            efficiency_rows.extend(
                [
                    {
                        "cps": cps,
                        "pollutant": pollutant,
                        "pathway": "surface",
                        "value": surface,
                    },
                    {
                        "cps": cps,
                        "pollutant": pollutant,
                        "pathway": "subsurface",
                        "value": subsurface,
                    },
                ]
            )

    data.update(
        {
            DATA_RANDOM_SEED: int(random_seed),
            DATA_N_SCENARIOS: int(n_scenarios),
            DATA_POLLUTANTS: pollutants,
            DATA_CPS: cps_codes,
            DATA_BMP_EFFICIENCY: pd.DataFrame(efficiency_rows),
            DATA_POLLUTANT_LOAD_RATE: None,
            DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE: False,
            DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS: {},
            DATA_PATHWAYS: ["surface", "subsurface"],
            DATA_LOAD_GENERATION: {
                "mode": LOAD_MODE_PLET_RUSLE,
                "_hydrology_lookup_table": hydrology,
            },
            DATA_PLET_INPUTS: plet,
            DATA_RUSLE_INPUTS: rusle,
            DATA_POLLUTANT_CONCENTRATIONS: surface_concentrations,
            DATA_GROUNDWATER_CONCENTRATIONS: subsurface_concentrations,
        }
    )
    return cfg, data
