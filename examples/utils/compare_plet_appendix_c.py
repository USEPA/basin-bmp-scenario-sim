#!/usr/bin/env python3
"""Compare model outputs against a raw PLET/RUSLE baseline using canonical repo equations.

This script builds one scenario per BMP (CPS) where exactly one BMP is placed on
one selected parcel, then compares:

- model parcel loads from parcels/s1.parquet
- baseline pathway loads computed with the repo's canonical
  calculate_plet_pathway_load_rates(...)
- final loads reconstructed by subtracting canonical realized BMP removed mass
  from bmps/s1.parquet after converting BMP mass to parcel load-rate basis

Why this version
----------------
The repo snapshot shows the canonical PLET/RUSLE pathway implementation in
calculate_plet_pathway_load_rates(...). That function computes:

- surface nutrient load = runoff concentration * annual runoff volume
  + sediment-bound nutrient contribution
- subsurface nutrient load = groundwater concentration * annual infiltration volume
- TSS surface load = RUSLE sediment load when RUSLE is available

Using that function avoids residual mismatch from hand-coded runoff formulas.
See the canonical implementation and tests in the snapshot.  

Usage
-----
python compare_plet_appendix_c_independent.py --base-config examples/east_fork/inputs/plet/east_fork_plet.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.input_config import load_and_validate_all
from src.logging_utils import make_logger
from src.model import Model
from src.plet_rusle import (
    calculate_plet_pathway_load_rates,
    calculate_load_diagnostics,
)

INCH_OVER_HA_TO_LITERS = 254_000.0


@dataclass
class ScenarioComparison:
    cps: int
    pid: str
    pollutants: List[str]
    area_ha: float
    mass_timestep_years: float
    model_initial: Dict[str, float]
    model_final: Dict[str, float]
    raw_initial: Dict[str, float]
    raw_final: Dict[str, float]
    raw_initial_surface: Dict[str, float]
    raw_initial_subsurface: Dict[str, float]
    model_initial_surface: Dict[str, float]
    model_initial_subsurface: Dict[str, float]
    bmp_baseline_mass_kg: Dict[str, float]
    bmp_treated_baseline_mass_kg: Dict[str, float]
    bmp_removed_mass_kg: Dict[str, float]
    bmp_baseline_load_rate: Dict[str, float]
    bmp_treated_baseline_load_rate: Dict[str, float]
    bmp_removed_load_rate: Dict[str, float]
    bmp_treatment_exposure_fraction: Dict[str, float]
    bmp_realized_efficiency: Dict[str, float]
    bmp_overall_reduction_fraction: Dict[str, float]
    abs_diff_initial: Dict[str, float]
    abs_diff_final: Dict[str, float]
    pct_diff_initial: Dict[str, float]
    pct_diff_final: Dict[str, float]
    pathway_abs_diff_surface: Dict[str, float]
    pathway_abs_diff_subsurface: Dict[str, float]
    debug_inputs: Dict[str, float]
    output_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Model-aligned raw Appendix C audit for plet_rusle using canonical repo pathway equations."
    )
    p.add_argument(
        "--base-config",
        required=True,
        help="Base YAML config to clone for one-BMP test scenarios.",
    )
    p.add_argument(
        "--pid",
        default=None,
        help="Parcel ID to force. Defaults to first real parcel ID.",
    )
    p.add_argument(
        "--out-dir",
        default="examples/east_fork/outputs_plet_appendix_c_model_aligned",
        help="Directory for generated outputs and audit summaries.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print extra per-scenario diagnostic details.",
    )
    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {str(k).lower(): v for k, v in cfg.items()}


def _resolve_path_like(value: Any, base_dir: Path) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return value
    p = Path(text)
    if p.is_absolute():
        return str(p)
    cfg_relative = (base_dir / p).resolve()
    if cfg_relative.exists():
        return str(cfg_relative)
    return str((ROOT / p).resolve())


def _resolve_config_paths(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    out = dict(cfg)
    base_dir = cfg_path.parent

    path_keys = [
        "domain",
        "parcels",
        "outlet_loc",
        "parcel_out",
        "parcel_up",
        "parcel_p",
        "bmp_efficiency",
        "bmp_cost",
        "bmp_sel",
        "outlet_target",
        "outlet_mean",
        "outputs",
        "input_distributions",
    ]
    for key in path_keys:
        if key in out:
            out[key] = _resolve_path_like(out[key], base_dir)

    lg = out.get("load_generation")
    if isinstance(lg, dict):
        lg2 = dict(lg)
        for key in [
            "plet_inputs",
            "hydrology_lookup",
            "rusle_inputs",
            "pollutant_concentrations",
            "groundwater_concentrations",
            "input_distributions",
        ]:
            if key in lg2:
                lg2[key] = _resolve_path_like(lg2[key], base_dir)
        out["load_generation"] = lg2

    return out


def _normalize_pid(value: Any) -> str:
    text = str(value).strip()
    if text == "":
        raise ValueError("PID cannot be empty")
    try:
        num = float(text)
    except ValueError:
        return text
    if np.isfinite(num) and num.is_integer():
        return str(int(num))
    return text


def _explicit_pids_from_parcel_p(base_cfg: Mapping[str, Any]) -> List[str]:
    df = pd.read_csv(Path(str(base_cfg["parcel_p"])))
    if "pid" not in df.columns or df.empty:
        raise ValueError("parcel_p must contain a non-empty pid column")
    pids = [_normalize_pid(v) for v in df["pid"].tolist()]
    pids = [p for p in pids if p != "*"]
    return list(dict.fromkeys(pids))


def _parcel_ids_from_parcels_layer(base_cfg: Mapping[str, Any]) -> List[str]:
    gdf = gpd.read_file(Path(str(base_cfg["parcels"])))
    if "pid" not in gdf.columns or gdf.empty:
        raise ValueError("parcels layer must contain a non-empty pid column")
    pids = [_normalize_pid(v) for v in gdf["pid"].tolist()]
    pids = [p for p in pids if p != "*"]
    return list(dict.fromkeys(pids))


def _select_pid(base_cfg: Mapping[str, Any], requested_pid: Optional[str]) -> str:
    if requested_pid is not None:
        pid = _normalize_pid(requested_pid)
        if pid == "*":
            raise ValueError("--pid may not be '*'")
        return pid
    pids = _explicit_pids_from_parcel_p(base_cfg)
    if not pids:
        pids = _parcel_ids_from_parcels_layer(base_cfg)
    if not pids:
        raise ValueError("No parcel IDs available")
    return pids[0]


def _safe_token(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _write_single_pid_probability(path: Path, pid: str) -> None:
    pd.DataFrame([{"pid": pid, "probability": 1.0}]).to_csv(path, index=False)


def _write_single_cps_selection(path: Path, cps: int, cps_values: Iterable[int]) -> None:
    rows = [{"cps": int(c), "probability": 1.0 if int(c) == int(cps) else 0.0} for c in cps_values]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_empty_parcel_up(path: Path) -> None:
    pd.DataFrame([{"pid": "*", "pid_up": None}]).to_csv(path, index=False)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        x = float(value)
    except Exception:
        return default
    if math.isnan(x):
        return default
    return x


def _first_present_float(row: Mapping[str, Any], candidates: Sequence[str], default: float = 0.0) -> float:
    for c in candidates:
        if c in row:
            val = row.get(c)
            try:
                x = float(val)
            except Exception:
                continue
            if not math.isnan(x):
                return x
    return default


def _load_distribution_catalog(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if "distribution_id" not in df.columns:
        return {}
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        did = str(row.get("distribution_id", "")).strip()
        if not did:
            continue
        if "value" in row and pd.notna(row["value"]):
            try:
                out[did] = float(row["value"])
                continue
            except Exception:
                pass
        if "mean" in row and pd.notna(row["mean"]):
            try:
                out[did] = float(row["mean"])
            except Exception:
                pass
    return out


def _resolve_numeric_row(row: Mapping[str, Any], dist_map: Mapping[str, float]) -> float:
    value = row.get("value")
    if pd.notna(value):
        text = str(value).strip()
        try:
            return float(text)
        except Exception:
            if text in dist_map:
                return float(dist_map[text])
    did = str(row.get("distribution_id", "")).strip()
    if did and did in dist_map:
        return float(dist_map[did])
    if pd.notna(row.get("mean")):
        return float(row.get("mean"))
    return 0.0


def _lookup_concentration(
    table_path: str,
    pid: str,
    pollutant: str,
    dist_map: Mapping[str, float],
) -> float:
    df = pd.read_csv(table_path)
    df["pid"] = df["pid"].astype(str).str.strip()
    df["pollutant"] = df["pollutant"].astype(str).str.upper().str.strip()

    exact = df[(df["pid"] == str(pid)) & (df["pollutant"] == pollutant)]
    if not exact.empty:
        return _resolve_numeric_row(exact.iloc[0], dist_map)

    wildcard = df[(df["pid"] == "*") & (df["pollutant"] == pollutant)]
    if not wildcard.empty:
        return _resolve_numeric_row(wildcard.iloc[0], dist_map)

    return 0.0


def _extract_realized_parameters(
    row: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    params: Dict[str, float] = {}

    params["annual_precip_in"] = _first_present_float(row, ["initial_annual_precip_in"], 0.0)
    params["rain_days"] = _first_present_float(row, ["initial_rain_days"], 0.0)
    params["rain_correction_fraction"] = _first_present_float(row, ["initial_rain_correction_fraction"], 1.0)
    params["runoff_day_fraction"] = _first_present_float(row, ["initial_runoff_day_fraction"], 0.0)
    params["ia_ratio"] = _first_present_float(row, ["initial_ia_ratio"], 0.0)
    params["cn"] = _first_present_float(row, ["initial_cn"], 0.0)

    # Canonical resolved infiltration_fraction may or may not be retained, but
    # annual_infiltration_in definitely appears in diagnostics/tests. Keep both.
    params["infiltration_fraction"] = _first_present_float(
        row,
        ["initial_infiltration_fraction", "initial_infiltration_frac"],
        float("nan"),
    )
    params["annual_infiltration_in"] = _first_present_float(
        row,
        ["initial_annual_infiltration_in"],
        float("nan"),
    )

    # Optional multipliers/defaulted parameters used by canonical function.
    params["runoff_multiplier"] = _first_present_float(row, ["initial_runoff_multiplier"], 1.0)
    params["groundwater_multiplier"] = _first_present_float(row, ["initial_groundwater_multiplier"], 1.0)
    params["sediment_multiplier"] = _first_present_float(row, ["initial_sediment_multiplier"], 1.0)
    params["sediment_delivery_multiplier"] = _first_present_float(row, ["initial_sediment_delivery_multiplier"], 1.0)

    params["r"] = _first_present_float(row, ["initial_r"], 0.0)
    params["k"] = _first_present_float(row, ["initial_k"], 0.0)
    params["ls"] = _first_present_float(row, ["initial_ls"], 0.0)
    params["c"] = _first_present_float(row, ["initial_c"], 0.0)
    params["p"] = _first_present_float(row, ["initial_p"], 0.0)
    params["sdr"] = _first_present_float(row, ["initial_sdr"], 1.0)
    params["sediment_n_pct"] = _first_present_float(row, ["initial_sediment_n_pct"], 0.0)
    params["sediment_p_pct"] = _first_present_float(row, ["initial_sediment_p_pct"], 0.0)
    params["enrichment_ratio"] = _first_present_float(row, ["initial_enrichment_ratio"], 1.0)

    debug_inputs = {
        "initial_annual_precip_in": params["annual_precip_in"],
        "initial_rain_days": params["rain_days"],
        "initial_rain_correction_fraction": params["rain_correction_fraction"],
        "initial_runoff_day_fraction": params["runoff_day_fraction"],
        "initial_ia_ratio": params["ia_ratio"],
        "initial_cn": params["cn"],
        "initial_infiltration_fraction": params["infiltration_fraction"],
        "initial_annual_infiltration_in": params["annual_infiltration_in"],
    }

    return params, debug_inputs


def _extract_model_initial_pathways(
    row: Mapping[str, Any],
    pollutants: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    surface: Dict[str, float] = {}
    subsurface: Dict[str, float] = {}

    for pol in pollutants:
        key_lower = pol.lower()
        surface[pol] = _as_float(row.get(f"initial_surface_{key_lower}_load_rate_kg_ha_yr"), 0.0)
        subsurface[pol] = _as_float(row.get(f"initial_subsurface_{key_lower}_load_rate_kg_ha_yr"), 0.0)

    return surface, subsurface


def _parcel_area_ha_from_data(data: Mapping[str, Any], pid: str) -> float:
    parcels_src = data.get("parcels")
    if parcels_src is None:
        raise KeyError("data does not contain 'parcels'")

    match = parcels_src[parcels_src["pid"].astype(str) == str(pid)]
    if match.empty:
        raise RuntimeError(f"Could not find parcel area for pid={pid}")

    area_ha = _as_float(match.iloc[0].get("area_ha"), 0.0)
    if area_ha <= 0.0:
        raise RuntimeError(f"Non-positive area_ha for pid={pid}: {area_ha}")
    return area_ha


def _pct_diff(a: float, b: float) -> float:
    denom = max(abs(a), 1.0e-12)
    return abs(b - a) / denom * 100.0


def _canonical_raw_pathways(
    params: Mapping[str, float],
    runoff_conc: Mapping[str, float],
    groundwater_conc: Mapping[str, float],
    pollutants: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    loads = calculate_plet_pathway_load_rates(
        params,
        runoff_conc,
        groundwater_conc,
        list(pollutants),
    )

    diagnostics = calculate_load_diagnostics(params)

    surface: Dict[str, float] = {}
    subsurface: Dict[str, float] = {}
    for idx, pol in enumerate(pollutants):
        surface[pol] = float(loads[idx, 0])
        subsurface[pol] = float(loads[idx, 1])

    debug = {
        "annual_runoff_in": diagnostics.get("annual_runoff_in", float("nan")),
        "annual_infiltration_in": diagnostics.get("annual_infiltration_in", float("nan")),
        "runoff_l_ha": diagnostics.get("annual_runoff_in", 0.0) * INCH_OVER_HA_TO_LITERS,
        "infiltration_l_ha": diagnostics.get("annual_infiltration_in", 0.0) * INCH_OVER_HA_TO_LITERS,
        "tss_kg_ha": diagnostics.get("sediment_load_rate_kg_ha_yr", float("nan")),
    }

    for pol in pollutants:
        if pol == "TSS":
            continue
        debug[f"{pol}_runoff_conc_mg_l"] = _as_float(runoff_conc.get(pol), 0.0)
        debug[f"{pol}_groundwater_conc_mg_l"] = _as_float(groundwater_conc.get(pol), 0.0)
        debug[f"{pol}_surface_kg_ha"] = surface[pol]
        debug[f"{pol}_subsurface_kg_ha"] = subsurface[pol]

    return surface, subsurface, debug


def _run_one_cps(
    base_cfg: Dict[str, Any],
    cps: int,
    pid: str,
    root_out: Path,
    debug: bool = False,
) -> ScenarioComparison:
    scenario_out = root_out / f"pid_{_safe_token(pid)}" / f"cps_{int(cps)}"
    scenario_out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"cps_{int(cps)}_", dir=str(scenario_out)) as td:
        tdir = Path(td)
        parcel_p_single = tdir / "parcel_p_single.csv"
        bmp_sel_single = tdir / "bmp_sel_single.csv"
        parcel_up_empty = tdir / "parcel_up_empty.csv"

        _write_single_pid_probability(parcel_p_single, pid)
        _write_single_cps_selection(bmp_sel_single, cps, base_cfg.get("cps", []))
        _write_empty_parcel_up(parcel_up_empty)

        cfg = dict(base_cfg)
        cfg["n_scenarios"] = 1
        cfg["bmp_limit_n"] = 1
        cfg.pop("bmp_limit_usd", None)
        cfg["bmp_fail_rate"] = 0.0
        cfg["random_seed"] = 42
        cfg["verbose"] = False
        cfg["parallel"] = {"n_jobs": 1}
        cfg["outputs"] = str(scenario_out)
        cfg["parcel_p"] = str(parcel_p_single)
        cfg["parcel_up"] = str(parcel_up_empty)
        cfg["bmp_sel"] = str(bmp_sel_single)

        lg = cfg.get("load_generation")
        lg2 = dict(lg) if isinstance(lg, dict) else {}
        lg2["process_parameter_mode"] = False
        cfg["load_generation"] = lg2

        logger, _ = make_logger(scenario_out, verbose=False, console=False)
        data = load_and_validate_all(cfg, logger)

        if str(data.get("load_generation", {}).get("mode", "")).strip().lower() != "plet_rusle":
            raise ValueError("This comparison requires load_generation.mode = 'plet_rusle'.")

        model = Model(cfg, data, logger)
        model.run_all_scenarios()

        bmps_path = scenario_out / "bmps" / "s1.parquet"
        parcels_path = scenario_out / "parcels" / "s1.parquet"
        lp_path = scenario_out / "load_parameters" / "s1.parquet"

        if not bmps_path.exists() or not parcels_path.exists() or not lp_path.exists():
            raise RuntimeError(f"Expected output files not found for cps={cps} in {scenario_out}")

        bmps = pd.read_parquet(bmps_path)
        parcels = pd.read_parquet(parcels_path)
        lp = pd.read_parquet(lp_path)

        if bmps.empty:
            raise RuntimeError(f"No BMP records found for cps={cps}")
        if len(bmps) != 1:
            raise RuntimeError(f"Expected exactly one BMP record for cps={cps}, found {len(bmps)}")

        bmp_row = bmps.iloc[0]
        pid_str = str(pid)

        parcel_row = parcels[parcels["pid"].astype(str) == pid_str]
        load_row = lp[lp["pid"].astype(str) == pid_str]

        if parcel_row.empty:
            raise RuntimeError(f"Parcel pid={pid_str} not found in parcels output for cps={cps}")
        if load_row.empty:
            raise RuntimeError(f"Parcel pid={pid_str} not found in load_parameters output for cps={cps}")

        parcel_row = parcel_row.iloc[0]
        load_row = load_row.iloc[0]

        pollutants = [str(p).upper() for p in data["pollutants"]]

        params, extracted_debug = _extract_realized_parameters(load_row)

        lg_cfg = base_cfg["load_generation"]
        dist_map = _load_distribution_catalog(base_cfg.get("input_distributions") or lg_cfg.get("input_distributions"))
        runoff_conc = {
            pol: _lookup_concentration(str(lg_cfg["pollutant_concentrations"]), pid_str, pol, dist_map)
            for pol in pollutants
            if pol != "TSS"
        }
        groundwater_conc = {
            pol: _lookup_concentration(str(lg_cfg["groundwater_concentrations"]), pid_str, pol, dist_map)
            for pol in pollutants
            if pol != "TSS"
        }

        raw_initial_surface, raw_initial_subsurface, baseline_debug = _canonical_raw_pathways(
            params,
            runoff_conc,
            groundwater_conc,
            pollutants,
        )
        raw_initial = {
            pol: raw_initial_surface[pol] + raw_initial_subsurface[pol]
            for pol in pollutants
        }

        model_initial_surface, model_initial_subsurface = _extract_model_initial_pathways(load_row, pollutants)

        area_ha = _parcel_area_ha_from_data(data, pid_str)
        mass_timestep_years = _as_float(bmp_row.get("mass_timestep_years"), 1.0)
        denom = area_ha * mass_timestep_years
        if denom <= 0.0:
            raise RuntimeError(
                f"Non-positive normalization denominator for pid={pid_str}: "
                f"area_ha={area_ha}, mass_timestep_years={mass_timestep_years}"
            )

        bmp_baseline_mass_kg: Dict[str, float] = {}
        bmp_treated_baseline_mass_kg: Dict[str, float] = {}
        bmp_removed_mass_kg: Dict[str, float] = {}
        bmp_baseline_load_rate: Dict[str, float] = {}
        bmp_treated_baseline_load_rate: Dict[str, float] = {}
        bmp_removed_load_rate: Dict[str, float] = {}
        bmp_treatment_exposure_fraction: Dict[str, float] = {}
        bmp_realized_efficiency: Dict[str, float] = {}
        bmp_overall_reduction_fraction: Dict[str, float] = {}
        raw_final: Dict[str, float] = {}

        for pol in pollutants:
            bmp_baseline_mass_kg[pol] = _as_float(bmp_row.get(f"baseline_mass_{pol}_kg"))
            bmp_treated_baseline_mass_kg[pol] = _as_float(bmp_row.get(f"treated_baseline_mass_{pol}_kg"))
            bmp_removed_mass_kg[pol] = _as_float(bmp_row.get(f"removed_mass_{pol}_kg"))

            bmp_baseline_load_rate[pol] = bmp_baseline_mass_kg[pol] / denom
            bmp_treated_baseline_load_rate[pol] = bmp_treated_baseline_mass_kg[pol] / denom
            bmp_removed_load_rate[pol] = bmp_removed_mass_kg[pol] / denom

            bmp_treatment_exposure_fraction[pol] = _as_float(
                bmp_row.get(f"treatment_exposure_fraction_{pol}"), default=float("nan")
            )
            bmp_realized_efficiency[pol] = _as_float(
                bmp_row.get(f"realized_efficiency_{pol}"), default=float("nan")
            )
            bmp_overall_reduction_fraction[pol] = _as_float(
                bmp_row.get(f"overall_reduction_fraction_{pol}"), default=float("nan")
            )

            raw_final[pol] = raw_initial[pol] - bmp_removed_load_rate[pol]

        model_initial = {
            pol: _as_float(parcel_row.get(f"baseline_load_rate_{pol}_kg_ha_yr"))
            for pol in pollutants
        }
        model_final = {
            pol: _as_float(parcel_row.get(f"final_load_rate_{pol}_kg_ha_yr"))
            for pol in pollutants
        }

        abs_diff_initial = {pol: abs(model_initial[pol] - raw_initial[pol]) for pol in pollutants}
        abs_diff_final = {pol: abs(model_final[pol] - raw_final[pol]) for pol in pollutants}
        pct_diff_initial = {pol: _pct_diff(model_initial[pol], raw_initial[pol]) for pol in pollutants}
        pct_diff_final = {pol: _pct_diff(model_final[pol], raw_final[pol]) for pol in pollutants}

        pathway_abs_diff_surface = {
            pol: abs(model_initial_surface[pol] - raw_initial_surface[pol])
            for pol in pollutants
        }
        pathway_abs_diff_subsurface = {
            pol: abs(model_initial_subsurface[pol] - raw_initial_subsurface[pol])
            for pol in pollutants
        }

        debug_inputs = dict(extracted_debug)
        debug_inputs.update(baseline_debug)
        debug_inputs["area_ha"] = area_ha
        debug_inputs["mass_timestep_years"] = mass_timestep_years

        if debug:
            print(f"\n[DEBUG] CPS={cps} PID={pid_str}")
            for k, v in debug_inputs.items():
                print(f"[DEBUG] {k} = {v}")
            for pol in pollutants:
                print(
                    f"[DEBUG] {pol}: "
                    f"model_surface={model_initial_surface[pol]:.12g}, "
                    f"raw_surface={raw_initial_surface[pol]:.12g}, "
                    f"model_subsurface={model_initial_subsurface[pol]:.12g}, "
                    f"raw_subsurface={raw_initial_subsurface[pol]:.12g}, "
                    f"bmp_removed_lr={bmp_removed_load_rate[pol]:.12g}"
                )

        return ScenarioComparison(
            cps=int(cps),
            pid=pid_str,
            pollutants=pollutants,
            area_ha=area_ha,
            mass_timestep_years=mass_timestep_years,
            model_initial=model_initial,
            model_final=model_final,
            raw_initial=raw_initial,
            raw_final=raw_final,
            raw_initial_surface=raw_initial_surface,
            raw_initial_subsurface=raw_initial_subsurface,
            model_initial_surface=model_initial_surface,
            model_initial_subsurface=model_initial_subsurface,
            bmp_baseline_mass_kg=bmp_baseline_mass_kg,
            bmp_treated_baseline_mass_kg=bmp_treated_baseline_mass_kg,
            bmp_removed_mass_kg=bmp_removed_mass_kg,
            bmp_baseline_load_rate=bmp_baseline_load_rate,
            bmp_treated_baseline_load_rate=bmp_treated_baseline_load_rate,
            bmp_removed_load_rate=bmp_removed_load_rate,
            bmp_treatment_exposure_fraction=bmp_treatment_exposure_fraction,
            bmp_realized_efficiency=bmp_realized_efficiency,
            bmp_overall_reduction_fraction=bmp_overall_reduction_fraction,
            abs_diff_initial=abs_diff_initial,
            abs_diff_final=abs_diff_final,
            pct_diff_initial=pct_diff_initial,
            pct_diff_final=pct_diff_final,
            pathway_abs_diff_surface=pathway_abs_diff_surface,
            pathway_abs_diff_subsurface=pathway_abs_diff_subsurface,
            debug_inputs=debug_inputs,
            output_dir=scenario_out,
        )


def main() -> None:
    args = parse_args()

    base_cfg_path = Path(args.base_config).resolve()
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    base_cfg = _resolve_config_paths(_load_yaml(base_cfg_path), base_cfg_path)
    pid = _select_pid(base_cfg, args.pid)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cps_values = [int(x) for x in base_cfg.get("cps", [])]
    if not cps_values:
        raise ValueError("No cps values found in base config")

    print(f"Running model-aligned raw Appendix C audit with pid={pid} across cps={cps_values}")

    all_rows: List[Dict[str, Any]] = []
    all_path_rows: List[Dict[str, Any]] = []

    for cps in cps_values:
        comp = _run_one_cps(
            base_cfg=base_cfg,
            cps=cps,
            pid=pid,
            root_out=out_dir,
            debug=args.debug,
        )

        print(f"\n=== CPS {comp.cps} | PID {comp.pid} ===")
        print(
            "pollutant | model_baseline | raw_baseline | model_final | raw_final | "
            "bmp_base_lr | bmp_treated_lr | bmp_removed_lr | exposure | realized_eff | overall_reduction"
        )
        for pol in comp.pollutants:
            print(
                f"{pol:9s} | "
                f"{comp.model_initial[pol]:13.6f} | {comp.raw_initial[pol]:12.6f} | "
                f"{comp.model_final[pol]:10.6f} | {comp.raw_final[pol]:9.6f} | "
                f"{comp.bmp_baseline_load_rate[pol]:11.6f} | {comp.bmp_treated_baseline_load_rate[pol]:14.6f} | "
                f"{comp.bmp_removed_load_rate[pol]:14.6f} | "
                f"{comp.bmp_treatment_exposure_fraction[pol]:8.6f} | "
                f"{comp.bmp_realized_efficiency[pol]:12.6f} | "
                f"{comp.bmp_overall_reduction_fraction[pol]:17.6f}"
            )
            all_rows.append(
                {
                    "cps": comp.cps,
                    "pid": comp.pid,
                    "area_ha": comp.area_ha,
                    "mass_timestep_years": comp.mass_timestep_years,
                    "pollutant": pol,
                    "model_baseline_kg_ha_yr": comp.model_initial[pol],
                    "raw_baseline_kg_ha_yr": comp.raw_initial[pol],
                    "baseline_abs_diff": comp.abs_diff_initial[pol],
                    "baseline_pct_diff": comp.pct_diff_initial[pol],
                    "model_final_kg_ha_yr": comp.model_final[pol],
                    "raw_final_kg_ha_yr": comp.raw_final[pol],
                    "final_abs_diff": comp.abs_diff_final[pol],
                    "final_pct_diff": comp.pct_diff_final[pol],
                    "raw_initial_surface_kg_ha_yr": comp.raw_initial_surface[pol],
                    "raw_initial_subsurface_kg_ha_yr": comp.raw_initial_subsurface[pol],
                    "model_initial_surface_kg_ha_yr": comp.model_initial_surface[pol],
                    "model_initial_subsurface_kg_ha_yr": comp.model_initial_subsurface[pol],
                    "surface_path_abs_diff": comp.pathway_abs_diff_surface[pol],
                    "subsurface_path_abs_diff": comp.pathway_abs_diff_subsurface[pol],
                    "bmp_baseline_mass_kg": comp.bmp_baseline_mass_kg[pol],
                    "bmp_treated_baseline_mass_kg": comp.bmp_treated_baseline_mass_kg[pol],
                    "bmp_removed_mass_kg": comp.bmp_removed_mass_kg[pol],
                    "bmp_baseline_load_rate_kg_ha_yr": comp.bmp_baseline_load_rate[pol],
                    "bmp_treated_baseline_load_rate_kg_ha_yr": comp.bmp_treated_baseline_load_rate[pol],
                    "bmp_removed_load_rate_kg_ha_yr": comp.bmp_removed_load_rate[pol],
                    "bmp_treatment_exposure_fraction": comp.bmp_treatment_exposure_fraction[pol],
                    "bmp_realized_efficiency": comp.bmp_realized_efficiency[pol],
                    "bmp_overall_reduction_fraction": comp.bmp_overall_reduction_fraction[pol],
                    "output_dir": str(comp.output_dir),
                }
            )

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "appendix_c_model_aligned_audit_summary.csv"
    summary.to_csv(summary_path, index=False)

    path_summary = pd.DataFrame(all_path_rows)
    path_summary_path = out_dir / "appendix_c_model_aligned_pathway_diagnostics.csv"
    path_summary.to_csv(path_summary_path, index=False)

    manifest = {
        "base_config": str(base_cfg_path),
        "pid": pid,
        "cps": cps_values,
        "summary_csv": str(summary_path),
        "pathway_diagnostics_csv": str(path_summary_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote pathway diagnostics: {path_summary_path}")
    print("\nNotes:")
    print("1. Raw baseline uses the repo's canonical calculate_plet_pathway_load_rates(...) implementation.")
    print("2. Raw final is computed as raw_baseline minus normalized bmp removed load rate.")
    print("3. If remaining mismatch persists, it is likely due to parameter extraction rather than duplicated science formulas.")


if __name__ == "__main__":
    main()