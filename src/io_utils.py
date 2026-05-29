"""
I/O helpers and input validation.

This module centralizes filesystem interactions:
- Reading geospatial/tabular inputs
- Validations and normalizations
- Writing cross-scenario consolidated outputs (see consolidate_transposed_summaries)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from .constants import (
    CFG_BMP_COST,
    CFG_BMP_EFFICIENCY,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_CPS,
    CFG_DELIVERY_RATIOS,
    CFG_DOMAIN,
    CFG_N_SCENARIOS,
    CFG_OUTLET_LOC,
    CFG_OUTLET_MEAN,
    CFG_OUTLET_TARGET,
    CFG_PARALLEL,
    CFG_PARCEL_OUT,
    CFG_PARCEL_P,
    CFG_PARCEL_UP,
    CFG_PARCELS,
    CFG_POLLUTANT_YIELD,
    CFG_POLLUTANTS,
    CFG_RANDOM_SEED,
    COL_AREA_HA,
    COL_AREA_M2,
    COL_CPS,
    COL_MEAN,
    COL_MAX,
    COL_MIN,
    COL_OID,
    COL_OIDS,
    COL_PERIM_M,
    COL_PID,
    COL_PID_UP,
    COL_POLLUTANT,
    COL_PROBABILITY,
    COL_SD,
    COL_TARGET,
    COL_UNIT,
)
from .utils import ci_get, normalize_columns, normalize_pollutant_label


def _require_cols(df: pd.DataFrame, required: Sequence[str], label: str, logger: Any) -> None:
    """Raise if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _merge_csvs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one or multiple CSVs, normalize columns, validate and concat."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames: List[pd.DataFrame] = []
    for p in paths:
        logger.debug(f"Reading {label} from {p}")
        df = pd.read_csv(p)
        df = normalize_columns(df)
        _require_cols(df, required_cols, f"{label} ({p})", logger)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(subset=required_cols, keep=False)
    if dup.any():
        logger.warning(f"Duplicate rows detected in {label}; keeping first occurrence")
        out = out.drop_duplicates(subset=required_cols, keep="first")
    return out


def _ensure_projected(gdf: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is in a projected CRS; reproject if needed."""
    if gdf.crs is None or not gdf.crs.is_projected:
        est = gdf.estimate_utm_crs()
        logger.info(f"Reprojecting to projected CRS: {est}")
        return gdf.to_crs(est)
    return gdf


def _normalize_pollutant_column(df: pd.DataFrame, label_col: str, label_name: str, logger: Any) -> pd.DataFrame:
    """Normalize a pollutant label column using canonical labels."""
    if label_col not in df.columns:
        raise ValueError(f"{label_name} must include '{label_col}'")
    df[label_col] = df[label_col].astype(str).apply(normalize_pollutant_label)
    return df


def _load_domain(cfg: Dict[str, Any], logger: Any) -> gpd.GeoDataFrame:
    """Load and project model domain geometry."""
    domain_path = Path(ci_get(cfg, CFG_DOMAIN))
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    domain = gpd.read_file(domain_path)
    return _ensure_projected(domain, logger)


def _load_parcels(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load parcel geometries, project to domain CRS, and clip to domain."""
    parcels_path = Path(ci_get(cfg, CFG_PARCELS))
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path}")
    parcels = gpd.read_file(parcels_path)
    parcels = _ensure_projected(parcels, logger).to_crs(domain.crs)
    parcels = parcels.clip(domain.unary_union).reset_index(drop=True)
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if COL_PID not in parcels.columns:
        raise ValueError(f"Parcels must include column '{COL_PID}'")
    parcels[COL_PID] = parcels[COL_PID].astype(str)
    parcels[COL_AREA_M2] = parcels.geometry.area
    parcels[COL_AREA_HA] = parcels[COL_AREA_M2] / 10_000.0
    parcels[COL_PERIM_M] = parcels.geometry.length
    return parcels


def _load_parcel_out(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load parcel -> outlet OID mapping."""
    return _merge_csvs(ci_get(cfg, CFG_PARCEL_OUT), [COL_PID, COL_OIDS], CFG_PARCEL_OUT, logger)


def _load_parcel_up(cfg: Dict[str, Any], logger: Any) -> Optional[pd.DataFrame]:
    """Load optional parcel upstream neighbors table."""
    if ci_get(cfg, CFG_PARCEL_UP) is None:
        return None
    return _merge_csvs(ci_get(cfg, CFG_PARCEL_UP), [COL_PID, COL_PID_UP], CFG_PARCEL_UP, logger)


def _load_parcel_p(cfg: Dict[str, Any], parcels: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Load or synthesize parcel selection probabilities, normalized to 1.0."""
    if ci_get(cfg, CFG_PARCEL_P) is not None:
        parcel_p = _merge_csvs(ci_get(cfg, CFG_PARCEL_P), [COL_PID, COL_PROBABILITY], CFG_PARCEL_P, logger)
        parcel_p[COL_PID] = parcel_p[COL_PID].astype(str)
        before = len(parcel_p)
        parcel_p = parcel_p[parcel_p[COL_PID].isin(parcels[COL_PID])].copy()
        dropped = before - len(parcel_p)
        if dropped:
            logger.warning(
                f"{CFG_PARCEL_P} contained {dropped} {COL_PID}(s) not present in parcels after clipping; they were removed"
            )
        if parcel_p.empty:
            raise ValueError(f"{CFG_PARCEL_P} has no {COL_PID}s that exist in parcels after clipping")
        total_prob = parcel_p[COL_PROBABILITY].sum()
        if total_prob <= 0:
            raise ValueError(f"{CFG_PARCEL_P} probabilities sum to zero or negative")
        parcel_p[COL_PROBABILITY] /= total_prob
    else:
        parcel_p = pd.DataFrame(
            {COL_PID: parcels[COL_PID].values, COL_PROBABILITY: np.full(len(parcels), 1 / len(parcels))}
        )
    return parcel_p


def _load_outlet_loc(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load outlet points and project to domain CRS."""
    outlet_path = Path(ci_get(cfg, CFG_OUTLET_LOC))
    if not outlet_path.exists():
        raise FileNotFoundError(f"Outlet location not found: {outlet_path}")
    outlet_loc = gpd.read_file(outlet_path).to_crs(domain.crs)
    return outlet_loc.rename(columns={c: c.lower() for c in outlet_loc.columns})


def _load_optional_outlet_stats(
    cfg: Dict[str, Any],
    key: str,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Optionally load per-outlet stats (target or mean), normalizing pollutant labels."""
    if ci_get(cfg, key) is None:
        logger.debug(f"Optional key {key} not provided; skipping {label}")
        return None
    df = _merge_csvs(ci_get(cfg, key), required_cols, label, logger)
    return _normalize_pollutant_column(df, COL_POLLUTANT, label, logger)


def _load_delivery_ratios(cfg: Dict[str, Any], logger: Any) -> Optional[pd.DataFrame]:
    """Load optional parcel->outlet delivery ratio table."""
    dr_cfg = ci_get(cfg, CFG_DELIVERY_RATIOS)
    if dr_cfg is None:
        logger.debug("No delivery ratios configured; using default delivery coefficients")
        return None
    dr_path = Path(dr_cfg)
    if not dr_path.exists():
        logger.warning(f"{CFG_DELIVERY_RATIOS} specified but file not found: {dr_cfg}; skipping delivery ratios")
        return None
    return _merge_csvs(
        dr_cfg,
        [COL_PID, COL_OID, "sdr_f_to_s", "sdr_s_to_o", "ndr_f_to_s", "ndr_s_to_o"],
        CFG_DELIVERY_RATIOS,
        logger,
    )


def _load_pollutants(cfg: Dict[str, Any]) -> List[str]:
    """Normalize and return the list of pollutants from config."""
    pols = ci_get(cfg, CFG_POLLUTANTS)
    if isinstance(pols, str):
        pols = [pols]
    if not pols:
        raise ValueError(f"At least one {CFG_POLLUTANTS} value must be specified")
    return [normalize_pollutant_label(p) for p in pols]


def _load_cps(cfg: Dict[str, Any]) -> List[int]:
    """Return CPS list from config as ints."""
    cps = ci_get(cfg, CFG_CPS)
    if isinstance(cps, int):
        cps = [cps]
    if not cps:
        raise ValueError("At least one cps code must be specified")
    return [int(c) for c in cps]


def _validate_stats_table(df: pd.DataFrame, label: str) -> None:
    """Validate that a stats table provides mean/sd, min/max or percentile columns."""
    cols = set(df.columns)
    ok = (
        ({"mean", "sd"} <= cols)
        or ({"min", "max"} <= cols)
        or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    )
    if not ok:
        raise ValueError(f"{label} must provide mean/sd or min/max or percentiles")


def _load_bmp_efficiency(cfg: Dict[str, Any], cps: List[int], pollutants: List[str], logger: Any) -> pd.DataFrame:
    """Load BMP efficiency stats filtered to requested CPS and pollutants."""
    df = _merge_csvs(ci_get(cfg, CFG_BMP_EFFICIENCY), [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    _validate_stats_table(df, CFG_BMP_EFFICIENCY)
    df = df[df[COL_CPS].astype(int).isin(cps) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")
    return df


def _load_pollutant_yield(
    cfg: Dict[str, Any],
    parcels: pd.DataFrame,
    pollutants: List[str],
    logger: Any,
) -> pd.DataFrame:
    """Load parcel pollutant yield stats for all parcels and pollutants."""
    df = _merge_csvs(ci_get(cfg, CFG_POLLUTANT_YIELD), [COL_PID, COL_POLLUTANT], CFG_POLLUTANT_YIELD, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_POLLUTANT_YIELD, logger)
    _validate_stats_table(df, CFG_POLLUTANT_YIELD)
    _validate_pollutant_yield_coverage(parcels, pollutants, df)
    return df


def _validate_pollutant_yield_coverage(parcels: pd.DataFrame, pollutants: List[str], pol_y: pd.DataFrame) -> None:
    """Ensure pollutant_yield covers every parcel x pollutant combination."""
    req = pd.MultiIndex.from_product([parcels[COL_PID].astype(str).values, pollutants], names=[COL_PID, COL_POLLUTANT])
    got = pd.MultiIndex.from_frame(pol_y[[COL_PID, COL_POLLUTANT]].astype(str))
    missing = req.difference(got)
    if len(missing) > 0:
        examples = list(missing)[:5]
        fmt = ", ".join([f"{p}-{pol}" for p, pol in examples])
        raise ValueError(f"pollutant_yield missing parcel+pollutant rows, e.g.: {fmt}")


def _load_bmp_cost(cfg: Dict[str, Any], logger: Any) -> Optional[pd.DataFrame]:
    """Load optional BMP cost table with stats."""
    if ci_get(cfg, CFG_BMP_COST) is None:
        return None
    df = _merge_csvs(ci_get(cfg, CFG_BMP_COST), [COL_CPS, COL_UNIT], CFG_BMP_COST, logger)
    cols = set(df.columns)
    ok = (
        ({"mean", "sd"} <= cols)
        or ({"min", "max"} <= cols)
        or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    )
    if not ok:
        raise ValueError("bmp_cost must provide mean/sd or min/max or percentiles")
    return df


def _build_parcel_up_map(parcel_up: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    """Convert parcel_up rows to a mapping {pid: [up_pid, ...]}."""
    m: Dict[str, List[str]] = {}
    if parcel_up is None:
        return m
    for _, row in parcel_up.iterrows():
        ups: List[str] = []
        if isinstance(row[COL_PID_UP], str) and row[COL_PID_UP].strip():
            ups = [x.strip() for x in row[COL_PID_UP].split(",") if x.strip()]
        m[str(row[COL_PID])] = ups
    return m


def _validate_parcel_out(parcel_out: pd.DataFrame, outlet_loc: pd.DataFrame) -> None:
    """Ensure parcel_out maps only to known outlet oids."""
    outlet_oids = set(outlet_loc[COL_OID].astype(str).tolist())
    po = parcel_out.copy()
    po[COL_PID] = po[COL_PID].astype(str)
    po[COL_OIDS] = po[COL_OIDS].astype(str)
    po["oids_list"] = po[COL_OIDS].apply(lambda s: [x.strip() for x in s.split(",") if str(x).strip()])
    unknown = sorted({oid for olist in po["oids_list"] for oid in olist if oid not in outlet_oids})
    if unknown:
        raise ValueError(f"parcel_out references unknown outlet oid(s): {unknown}")


def _validate_parcel_up(parcel_up: Optional[pd.DataFrame], parcels: pd.DataFrame) -> None:
    """Validate parcel_up references only known PIDs."""
    if parcel_up is None:
        return
    parcel_up[COL_PID] = parcel_up[COL_PID].astype(str)
    parcel_up[COL_PID_UP] = parcel_up[COL_PID_UP].fillna("").astype(str)
    valid_pids = set(parcels[COL_PID].astype(str).tolist())
    unknown_pids = sorted({pid for pid in parcel_up[COL_PID].tolist() if pid not in valid_pids})
    unknown_up_pids = sorted({pid for pid in parcel_up[COL_PID_UP].tolist() if pid and pid not in valid_pids})
    if unknown_pids:
        raise ValueError(f"parcel_up contains pid values not present in parcels: {unknown_pids}")
    if unknown_up_pids:
        raise ValueError(f"parcel_up contains pid_up values not present in parcels: {unknown_up_pids}")


def _validate_outlet_stats(
    outlet_target: Optional[pd.DataFrame],
    outlet_mean: Optional[pd.DataFrame],
    outlet_loc: pd.DataFrame,
) -> None:
    """Validate that outlet target/mean refer only to known outlet oids."""
    outlet_oids = set(outlet_loc[COL_OID].astype(str).tolist())
    target_oids = set(outlet_target[COL_OID].astype(str).tolist()) if outlet_target is not None else set()
    mean_oids = set(outlet_mean[COL_OID].astype(str).tolist()) if outlet_mean is not None else set()
    invalid = sorted((target_oids | mean_oids) - outlet_oids)
    if invalid:
        raise ValueError(f"outlet_target/outlet_mean reference unknown outlet oid(s): {invalid}")


def _build_parcel_out_map(parcel_out: pd.DataFrame) -> Dict[str, List[str]]:
    """Convert parcel_out rows into a mapping {pid: [oid, ...]}."""
    po = parcel_out.copy()
    po[COL_PID] = po[COL_PID].astype(str)
    po[COL_OIDS] = po[COL_OIDS].astype(str)
    po["oids_list"] = po[COL_OIDS].apply(lambda s: [x.strip() for x in s.split(",") if str(x).strip()])
    return {str(row[COL_PID]): row["oids_list"] for _, row in po.iterrows()}


def load_and_validate_all(cfg: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    """Load all model inputs, validate, and return a normalized data payload.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed configuration dictionary (case-insensitive keys).
    logger : Any
        Logger for messages.

    Returns
    -------
    Dict[str, Any]
        Dictionary keyed by constants.DATA_* names, used by model.Model.
    """
    logger.info("Loading and validating input datasets")

    domain = _load_domain(cfg, logger)
    parcels = _load_parcels(cfg, domain, logger)
    parcel_out = _load_parcel_out(cfg, logger)
    parcel_up = _load_parcel_up(cfg, logger)
    parcel_p = _load_parcel_p(cfg, parcels, logger)
    outlet_loc = _load_outlet_loc(cfg, domain, logger)
    outlet_target = _load_optional_outlet_stats(cfg, CFG_OUTLET_TARGET, [COL_OID, COL_POLLUTANT, COL_TARGET], CFG_OUTLET_TARGET, logger)
    outlet_mean = _load_optional_outlet_stats(cfg, CFG_OUTLET_MEAN, [COL_OID, COL_POLLUTANT, COL_MEAN], CFG_OUTLET_MEAN, logger)
    delivery_ratios = _load_delivery_ratios(cfg, logger)
    pollutants = _load_pollutants(cfg)
    cps = _load_cps(cfg)
    bmp_eff = _load_bmp_efficiency(cfg, cps, pollutants, logger)
    pol_y = _load_pollutant_yield(cfg, parcels, pollutants, logger)
    bmp_cost = _load_bmp_cost(cfg, logger)

    n_scenarios = int(ci_get(cfg, CFG_N_SCENARIOS))
    limit_n = ci_get(cfg, CFG_BMP_LIMIT_N)
    limit_usd = ci_get(cfg, CFG_BMP_LIMIT_USD)
    if limit_n is None and limit_usd is None:
        raise ValueError("Specify bmp_limit_n or bmp_limit_usd")

    parallel = dict(ci_get(cfg, CFG_PARALLEL) or {})
    random_seed = ci_get(cfg, CFG_RANDOM_SEED)

    _validate_parcel_out(parcel_out, outlet_loc)
    _validate_parcel_up(parcel_up, parcels)
    _validate_outlet_stats(outlet_target, outlet_mean, outlet_loc)

    parcel_out_map = _build_parcel_out_map(parcel_out)
    parcel_up_map = _build_parcel_up_map(parcel_up)

    avg_area_ha = float(parcels[COL_AREA_HA].mean())
    avg_perim_m = float(parcels[COL_PERIM_M].mean())

    logger.info("Input validation complete; assembling data payload")
    data = dict(
        domain=domain,
        parcels=parcels,
        parcel_out_map=parcel_out_map,
        parcel_up_map=parcel_up_map,
        parcel_p=parcel_p,
        outlet_loc=outlet_loc,
        outlet_target=outlet_target,
        outlet_mean=outlet_mean,
        delivery_ratios=delivery_ratios,
        pollutants=pollutants,
        cps=cps,
        bmp_eff=bmp_eff,
        pollutant_yield=pol_y,
        bmp_cost=bmp_cost,
        n_scenarios=int(n_scenarios),
        bmp_limit_n=int(limit_n) if limit_n is not None else None,
        bmp_limit_usd=float(limit_usd) if limit_usd is not None else None,
        parallel=parallel,
        random_seed=int(random_seed) if random_seed is not None else None,
        avg_area_ha=avg_area_ha,
        avg_perim_m=avg_perim_m,
    )
    return data


def consolidate_transposed_summaries(outputs_dir: Path, logger) -> Path:
    """Consolidate all per-scenario transposed summaries into one CSV.

    Parameters
    ----------
    outputs_dir : Path
        Root outputs directory (contains 'summaries' subfolder).
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    Path
        Path to outputs/summaries/all_scenarios.csv.

    Notes
    -----
    - Reads: outputs/summaries/s*.csv (each with a 'field' column)
    - Writes: outputs/summaries/all_scenarios.csv
    - Outer-joins on 'field'. Columns sorted by scenario id, with "All CPS"
      first within each scenario.
    """
    outputs_dir = Path(outputs_dir)
    summaries_dir = outputs_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    out_path = summaries_dir / "all_scenarios.csv"

    files = sorted(p for p in summaries_dir.glob("s*.csv") if p.name != out_path.name)
    if not files:
        logger.info("No per-scenario summaries found to consolidate.")
        pd.DataFrame({"field": []}).to_csv(out_path, index=False)
        return out_path

    logger.info(f"Consolidating {len(files)} per-scenario summaries into {out_path}")

    combined = None
    for p in files:
        df = pd.read_csv(p)
        if "field" not in df.columns:
            logger.warning(f"Skipping {p} (no 'field' column)")
            continue
        df = df.set_index("field")
        combined = df if combined is None else combined.join(df, how="outer")

    if combined is None or combined.empty:
        logger.warning("No valid per-scenario summary data found; writing empty file.")
        pd.DataFrame({"field": []}).to_csv(out_path, index=False)
        return out_path

    def col_key(cname: str):
        m = re.match(r"s(\d+)-(.*)", str(cname))
        if not m:
            return (10**9, 1, str(cname))
        sid = int(m.group(1))
        tail = m.group(2)
        is_all = 0 if tail.strip() == "All CPS" else 1
        return (sid, is_all, tail)

    ordered_cols = sorted([c for c in combined.columns], key=col_key)
    combined = combined[ordered_cols].reset_index()
    combined.to_csv(out_path, index=False)
    logger.info(f"Wrote consolidated transposed summaries: {out_path}")
    return out_path