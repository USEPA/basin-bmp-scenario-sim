import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from pathlib import Path

from .utils import normalize_columns, ci_get


def _require_cols(df, required, label, logger):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _merge_csvs(paths, required_cols, label, logger):
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames = []
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


def _ensure_projected(gdf, logger):
    if gdf.crs is None or not gdf.crs.is_projected:
        est = gdf.estimate_utm_crs()
        logger.info(f"Reprojecting to projected CRS: {est}")
        gdf = gdf.to_crs(est)
    return gdf


def load_and_validate_all(cfg: dict, logger):
    # domain
    domain_path = Path(ci_get(cfg, "domain"))
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    domain = gpd.read_file(domain_path)
    domain = _ensure_projected(domain, logger)

    # parcels
    parcels_path = Path(ci_get(cfg, "parcels"))
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path}")
    parcels = gpd.read_file(parcels_path)
    parcels = _ensure_projected(parcels, logger)
    parcels = parcels.to_crs(domain.crs)
    parcels = parcels.clip(domain.unary_union)
    parcels = parcels.reset_index(drop=True)
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if "pid" not in parcels.columns:
        raise ValueError("Parcels must include column 'pid'")
    parcels["pid"] = parcels["pid"].astype(str)  # normalize PID type
    parcels["area_m2"] = parcels.geometry.area
    parcels["area_ha"] = parcels["area_m2"] / 10000.0
    parcels["perim_m"] = parcels.geometry.length

    # parcel_out (required)
    parcel_out = _merge_csvs(ci_get(cfg, "parcel_out"), ["pid", "oids"], "parcel_out", logger)

    # parcel_up (optional)
    parcel_up = None
    if ci_get(cfg, "parcel_up") is not None:
        parcel_up = _merge_csvs(ci_get(cfg, "parcel_up"), ["pid", "pid_up"], "parcel_up", logger)

    # parcel_p (optional -> default uniform)
    if ci_get(cfg, "parcel_p") is not None:
        parcel_p = _merge_csvs(ci_get(cfg, "parcel_p"), ["pid", "probability"], "parcel_p", logger)
        parcel_p["pid"] = parcel_p["pid"].astype(str)
        # keep only PIDs that exist in parcels after clipping
        before = len(parcel_p)
        parcel_p = parcel_p[parcel_p["pid"].isin(parcels["pid"])].copy()
        dropped = before - len(parcel_p)
        if dropped:
            logger.warning(f"parcel_p contained {dropped} pid(s) not present in parcels after clipping; they were removed")
        if parcel_p.empty:
            raise ValueError("parcel_p has no PIDs that exist in parcels after clipping")
        s = parcel_p["probability"].sum()
        if s <= 0:
            raise ValueError("parcel_p probabilities sum to zero or negative")
        parcel_p["probability"] = parcel_p["probability"] / s
    else:
        # uniform probabilities across all parcels present after clipping
        parcel_p = pd.DataFrame({"pid": parcels["pid"].values, "probability": np.full(len(parcels), 1 / len(parcels))})

    # outlet_loc
    outlet_path = Path(ci_get(cfg, "outlet_loc"))
    outlet_loc = gpd.read_file(outlet_path)
    outlet_loc = outlet_loc.to_crs(domain.crs)
    outlet_loc = outlet_loc.rename(columns={c: c.lower() for c in outlet_loc.columns})
    if "oid" not in outlet_loc.columns:
        raise ValueError("outlet_loc must include 'oid'")

    # outlet_target (optional)
    outlet_target = None
    if ci_get(cfg, "outlet_target") is not None:
        outlet_target = _merge_csvs(ci_get(cfg, "outlet_target"), ["oid", "pollutant", "target"], "outlet_target", logger)

    # outlet_mean (optional)
    outlet_mean = None
    if ci_get(cfg, "outlet_mean") is not None:
        outlet_mean = _merge_csvs(ci_get(cfg, "outlet_mean"), ["oid", "pollutant", "mean"], "outlet_mean", logger)

    # delivery_ratios (optional -> default 1.0 in simulate if missing)
    delivery_ratios = None
    if ci_get(cfg, "delivery_ratios") is not None:
        delivery_ratios = _merge_csvs(
            ci_get(cfg, "delivery_ratios"),
            ["pid", "oid", "sdr_f_to_s", "sdr_s_to_o", "ndr_f_to_s", "ndr_s_to_o"],
            "delivery_ratios",
            logger,
        )

    # pollutants list
    pollutants = ci_get(cfg, "pollutants")
    if isinstance(pollutants, str):
        pollutants = [pollutants]
    if not pollutants:
        raise ValueError("At least one pollutant must be specified")

    # cps list
    cps = ci_get(cfg, "cps")
    if isinstance(cps, int):
        cps = [cps]
    if not cps:
        raise ValueError("At least one cps code must be specified")

    # bmp_efficiency (required)
    bmp_eff_paths = ci_get(cfg, "bmp_efficiency")
    bmp_eff = _merge_csvs(bmp_eff_paths, ["cps", "pollutant"], "bmp_efficiency", logger)
    cols = set(bmp_eff.columns)
    ok = ({"mean", "sd"} <= cols) or ({"min", "max"} <= cols) or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    if not ok:
        raise ValueError("bmp_efficiency must provide mean/sd or min/max or percentiles")
    # Filter to needed cps/pollutants
    bmp_eff = bmp_eff[bmp_eff["cps"].astype(int).isin(cps) & bmp_eff["pollutant"].isin(pollutants)].copy()
    if bmp_eff.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")

    # pollutant_yield (required)
    pol_y_paths = ci_get(cfg, "pollutant_yield")
    pol_y = _merge_csvs(pol_y_paths, ["pid", "pollutant"], "pollutant_yield", logger)
    pol_y["pid"] = pol_y["pid"].astype(str)
    cols = set(pol_y.columns)
    ok = ({"mean", "sd"} <= cols) or ({"min", "max"} <= cols) or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    if not ok:
        raise ValueError("pollutant_yield must provide mean/sd or min/max or percentiles")

    # Validate coverage for all parcels/pollutants after clipping
    req_idx = pd.MultiIndex.from_product([parcels["pid"].astype(str).values, pollutants], names=["pid", "pollutant"])
    pol_idx = pd.MultiIndex.from_frame(pol_y[["pid", "pollutant"]].astype(str))
    missing = req_idx.difference(pol_idx)
    if len(missing) > 0:
        # Show a few missing combinations to guide the user
        examples = list(missing)[:5]
        raise ValueError("pollutant_yield missing parcel+pollutant rows, e.g.: " + ", ".join([f"{p}-{pol}" for p, pol in examples]))

    # bmp_cost (optional but used for inverse-cost selection and required if bmp_limit_usd set)
    bmp_cost = None
    if ci_get(cfg, "bmp_cost") is not None:
        bmp_cost = _merge_csvs(ci_get(cfg, "bmp_cost"), ["cps", "unit"], "bmp_cost", logger)
        cols = set(bmp_cost.columns)
        ok = ({"mean", "sd"} <= cols) or ({"min", "max"} <= cols) or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
        if not ok:
            raise ValueError("bmp_cost must provide mean/sd or min/max or percentiles")

    # Scenarios and limits
    n_scenarios = int(ci_get(cfg, "n_scenarios"))
    limit_n = ci_get(cfg, "bmp_limit_n")
    limit_usd = ci_get(cfg, "bmp_limit_usd")
    if limit_n is None and limit_usd is None:
        raise ValueError("Specify bmp_limit_n or bmp_limit_usd")

    # parallel (reserved for future use)
    parallel = dict(ci_get(cfg, "parallel") or {})
    random_seed = ci_get(cfg, "random_seed")

    # Build parcel_out map (use string PIDs)
    po = parcel_out.copy()
    po["pid"] = po["pid"].astype(str)
    po["oids"] = po["oids"].astype(str)
    po["oids_list"] = po["oids"].apply(lambda s: [x.strip() for x in s.split(",") if str(x).strip() != ""])
    parcel_out_map = {str(r["pid"]): r["oids_list"] for _, r in po.iterrows()}

    # parcel_up map (string PIDs)
    parcel_up_map = {}
    if parcel_up is not None:
        for _, r in parcel_up.iterrows():
            pid = str(r["pid"])
            ups = []
            if isinstance(r["pid_up"], str) and r["pid_up"].strip():
                ups = [x.strip() for x in r["pid_up"].split(",") if x.strip()]
            parcel_up_map[pid] = ups

    # Averages for cost heuristics (probability estimation if bmp_sel not given)
    avg_area_ha = parcels["area_ha"].mean()
    avg_perim_m = parcels["perim_m"].mean()

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