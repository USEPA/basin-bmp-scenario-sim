import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

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
    CFG_PARCELS,
    CFG_PARCEL_OUT,
    CFG_PARCEL_P,
    CFG_PARCEL_UP,
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
    CFG_PARALLEL,
)
from .utils import normalize_columns, ci_get, normalize_pollutant_label


def _require_cols(df: Any, required: Sequence[str], label: str, logger: Any) -> None:
    """Ensure a dataframe contains required columns and raise if any are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _merge_csvs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one or more CSV inputs, normalize their columns, and merge them.

    Duplicate rows are detected using the required column subset and de-duplicated
    so later validation and joins operate on a consistent dataset.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df = normalize_columns(df)
        _require_cols(df, required_cols, f"{label} ({p})", logger)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # detect duplicates on the required key columns
    dup = out.duplicated(subset=required_cols, keep=False)
    if dup.any():
        logger.warning(f"Duplicate rows detected in {label}, keeping first occurrence")
        out = out.drop_duplicates(subset=required_cols, keep="first").copy()
    return out


def _ensure_projected(gdf: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Ensure a GeoDataFrame is in a projected CRS for area/length calculations.

    If the input is not projected, estimate a suitable UTM CRS and reproject it.
    """
    if gdf.crs is None or not gdf.crs.is_projected:
        est = gdf.estimate_utm_crs()
        logger.info(f"Reprojecting to projected CRS: {est}")
        gdf = gdf.to_crs(est)
    return gdf


def _normalize_pollutant_column(df: pd.DataFrame, label_col: str, label_name: str, logger: Any) -> pd.DataFrame:
    """Normalize pollutant values in a DataFrame to canonical labels."""
    if label_col not in df.columns:
        raise ValueError(f"{label_name} must include '{label_col}'")
    df[label_col] = df[label_col].astype(str).apply(normalize_pollutant_label)
    return df


def load_and_validate_all(cfg: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    """Load all configured inputs, validate them, and build the shared model dataset.

    This function performs schema validation, projection normalization, ID
    normalization, and default handling for optional inputs.
    """
    # domain
    domain_path = Path(ci_get(cfg, CFG_DOMAIN))
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    domain = gpd.read_file(domain_path)
    domain = _ensure_projected(domain, logger)

    # parcels
    parcels_path = Path(ci_get(cfg, CFG_PARCELS))
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path}")
    parcels = gpd.read_file(parcels_path)
    parcels = _ensure_projected(parcels, logger)
    parcels = parcels.to_crs(domain.crs)
    parcels = parcels.clip(domain.unary_union)
    parcels = parcels.reset_index(drop=True)
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if COL_PID not in parcels.columns:
        raise ValueError(f"Parcels must include column '{COL_PID}'")
    parcels[COL_PID] = parcels[COL_PID].astype(str)  # normalize PID type
    parcels[COL_AREA_M2] = parcels.geometry.area
    parcels[COL_AREA_HA] = parcels[COL_AREA_M2] / 10000.0
    parcels[COL_PERIM_M] = parcels.geometry.length

    # parcel_out (required)
    parcel_out = _merge_csvs(ci_get(cfg, CFG_PARCEL_OUT), [COL_PID, COL_OIDS], CFG_PARCEL_OUT, logger)

    # parcel_up (optional)
    parcel_up = None
    if ci_get(cfg, CFG_PARCEL_UP) is not None:
        parcel_up = _merge_csvs(ci_get(cfg, CFG_PARCEL_UP), [COL_PID, COL_PID_UP], CFG_PARCEL_UP, logger)

    # parcel_p (optional -> default uniform)
    if ci_get(cfg, CFG_PARCEL_P) is not None:
        parcel_p = _merge_csvs(ci_get(cfg, CFG_PARCEL_P), [COL_PID, COL_PROBABILITY], CFG_PARCEL_P, logger)
        parcel_p[COL_PID] = parcel_p[COL_PID].astype(str)
        # keep only PIDs that exist in parcels after clipping
        before = len(parcel_p)
        parcel_p = parcel_p[parcel_p[COL_PID].isin(parcels[COL_PID])].copy()
        dropped = before - len(parcel_p)
        if dropped:
            logger.warning(f"{CFG_PARCEL_P} contained {dropped} {COL_PID}(s) not present in parcels after clipping; they were removed")
        if parcel_p.empty:
            raise ValueError(f"{CFG_PARCEL_P} has no {COL_PID}s that exist in parcels after clipping")
        s = parcel_p[COL_PROBABILITY].sum()
        if s <= 0:
            raise ValueError(f"{CFG_PARCEL_P} probabilities sum to zero or negative")
        parcel_p[COL_PROBABILITY] = parcel_p[COL_PROBABILITY] / s
    else:
        # uniform probabilities across all parcels present after clipping
        parcel_p = pd.DataFrame(
            {
                COL_PID: parcels[COL_PID].values,
                COL_PROBABILITY: np.full(len(parcels), 1 / len(parcels)),
            }
        )

    # outlet_loc
    outlet_path = Path(ci_get(cfg, CFG_OUTLET_LOC))
    outlet_loc = gpd.read_file(outlet_path)
    outlet_loc = outlet_loc.to_crs(domain.crs)
    outlet_loc = outlet_loc.rename(columns={c: c.lower() for c in outlet_loc.columns})
    if COL_OID not in outlet_loc.columns:
        raise ValueError(f"{CFG_OUTLET_LOC} must include '{COL_OID}'")

    # outlet_target (optional)
    outlet_target = None
    if ci_get(cfg, CFG_OUTLET_TARGET) is not None:
        outlet_target = _merge_csvs(ci_get(cfg, CFG_OUTLET_TARGET), [COL_OID, COL_POLLUTANT, COL_TARGET], CFG_OUTLET_TARGET, logger)
        outlet_target = _normalize_pollutant_column(outlet_target, COL_POLLUTANT, CFG_OUTLET_TARGET, logger)

    # outlet_mean (optional)
    outlet_mean = None
    if ci_get(cfg, CFG_OUTLET_MEAN) is not None:
        outlet_mean = _merge_csvs(ci_get(cfg, CFG_OUTLET_MEAN), [COL_OID, COL_POLLUTANT, COL_MEAN], CFG_OUTLET_MEAN, logger)
        outlet_mean = _normalize_pollutant_column(outlet_mean, COL_POLLUTANT, CFG_OUTLET_MEAN, logger)

    # delivery_ratios (optional -> default 1.0 in simulate if missing)
    delivery_ratios = None
    if ci_get(cfg, CFG_DELIVERY_RATIOS) is not None:
        delivery_ratios = _merge_csvs(
            ci_get(cfg, CFG_DELIVERY_RATIOS),
            [COL_PID, COL_OID, "sdr_f_to_s", "sdr_s_to_o", "ndr_f_to_s", "ndr_s_to_o"],
            CFG_DELIVERY_RATIOS,
            logger,
        )

    # pollutants list
    pollutants = ci_get(cfg, CFG_POLLUTANTS)
    if isinstance(pollutants, str):
        pollutants = [pollutants]
    if not pollutants:
        raise ValueError(f"At least one {CFG_POLLUTANTS} value must be specified")
    pollutants = [normalize_pollutant_label(pol) for pol in pollutants]

    # cps list
    cps = ci_get(cfg, CFG_CPS)
    if isinstance(cps, int):
        cps = [cps]
    if not cps:
        raise ValueError("At least one cps code must be specified")

    # bmp_efficiency (required)
    bmp_eff_paths = ci_get(cfg, CFG_BMP_EFFICIENCY)
    bmp_eff = _merge_csvs(bmp_eff_paths, [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    bmp_eff = _normalize_pollutant_column(bmp_eff, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    cols = set(bmp_eff.columns)
    ok = ({COL_MEAN, COL_SD} <= cols) or ({COL_MIN, COL_MAX} <= cols) or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    if not ok:
        raise ValueError("bmp_efficiency must provide mean/sd or min/max or percentiles")
    # Filter to needed cps/pollutants
    bmp_eff = bmp_eff[bmp_eff[COL_CPS].astype(int).isin(cps) & bmp_eff[COL_POLLUTANT].isin(pollutants)].copy()
    if bmp_eff.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")

    # pollutant_yield (required)
    pol_y_paths = ci_get(cfg, CFG_POLLUTANT_YIELD)
    pol_y = _merge_csvs(pol_y_paths, [COL_PID, COL_POLLUTANT], CFG_POLLUTANT_YIELD, logger)
    pol_y[COL_PID] = pol_y[COL_PID].astype(str)
    pol_y = _normalize_pollutant_column(pol_y, COL_POLLUTANT, CFG_POLLUTANT_YIELD, logger)
    cols = set(pol_y.columns)
    ok = ({COL_MEAN, COL_SD} <= cols) or ({COL_MIN, COL_MAX} <= cols) or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    if not ok:
        raise ValueError("pollutant_yield must provide mean/sd or min/max or percentiles")

    # Validate coverage for all parcels/pollutants after clipping
    req_idx = pd.MultiIndex.from_product([parcels[COL_PID].astype(str).values, pollutants], names=[COL_PID, COL_POLLUTANT])
    pol_idx = pd.MultiIndex.from_frame(pol_y[[COL_PID, COL_POLLUTANT]].astype(str))
    missing = req_idx.difference(pol_idx)
    if len(missing) > 0:
        # Show a few missing combinations to guide the user
        examples = list(missing)[:5]
        raise ValueError("pollutant_yield missing parcel+pollutant rows, e.g.: " + ", ".join([f"{p}-{pol}" for p, pol in examples]))

    # bmp_cost (optional but used for inverse-cost selection and required if bmp_limit_usd set)
    bmp_cost = None
    if ci_get(cfg, CFG_BMP_COST) is not None:
        bmp_cost = _merge_csvs(ci_get(cfg, CFG_BMP_COST), [COL_CPS, COL_UNIT], CFG_BMP_COST, logger)
        cols = set(bmp_cost.columns)
        ok = ({"mean", "sd"} <= cols) or ({"min", "max"} <= cols) or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
        if not ok:
            raise ValueError("bmp_cost must provide mean/sd or min/max or percentiles")

    # Scenarios and limits
    n_scenarios = int(ci_get(cfg, CFG_N_SCENARIOS))
    limit_n = ci_get(cfg, CFG_BMP_LIMIT_N)
    limit_usd = ci_get(cfg, CFG_BMP_LIMIT_USD)
    if limit_n is None and limit_usd is None:
        raise ValueError("Specify bmp_limit_n or bmp_limit_usd")

    # parallel (reserved for future use)
    parallel = dict(ci_get(cfg, CFG_PARALLEL) or {})
    random_seed = ci_get(cfg, CFG_RANDOM_SEED)

    # Build parcel_out map (use string PIDs)
    po = parcel_out.copy()
    po[COL_PID] = po[COL_PID].astype(str)
    po[COL_OIDS] = po[COL_OIDS].astype(str)
    po["oids_list"] = po[COL_OIDS].apply(lambda s: [x.strip() for x in s.split(",") if str(x).strip() != ""])
    parcel_out_map = {str(r[COL_PID]): r["oids_list"] for _, r in po.iterrows()}

    # parcel_up map (string PIDs)
    parcel_up_map = {}
    if parcel_up is not None:
        for _, r in parcel_up.iterrows():
            pid = str(r[COL_PID])
            ups = []
            if isinstance(r[COL_PID_UP], str) and r[COL_PID_UP].strip():
                ups = [x.strip() for x in r[COL_PID_UP].split(",") if x.strip()]
            parcel_up_map[pid] = ups

    # Averages for cost heuristics (probability estimation if bmp_sel not given)
    avg_area_ha = parcels[COL_AREA_HA].mean()
    avg_perim_m = parcels[COL_PERIM_M].mean()

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
        cps=[int(c) for c in cps],
        bmp_eff=bmp_eff,
        pollutant_yield=pol_y,
        bmp_cost=bmp_cost,
        n_scenarios=int(n_scenarios),
        bmp_limit_n=int(limit_n) if limit_n is not None else None,
        bmp_limit_usd=float(limit_usd) if limit_usd is not None else None,
        parallel=parallel,
        random_seed=int(random_seed) if random_seed is not None else None,
        avg_area_ha=float(avg_area_ha),
        avg_perim_m=float(avg_perim_m),
    )
    return data