import pandas as pd
from numpy.random import Generator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .constants import (
    CFG_BUFFER_DEPTH_FT,
    COL_AREA_HA,
    COL_CPS,
    COL_PERIM_M,
    COL_PID,
    COL_POLLUTANT,
    COL_MEAN,
    COL_SD,
    COL_MIN,
    COL_MAX,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_PORTION_TREATED,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
    OUTPUT_WETLAND_AREA,
    PERCENTILE_PREFIX,
)
from .costs import compute_bmp_cost_usd
from .sampling import sample_from_stats

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

FT_TO_M = 0.3048  # meters per foot


def sample_efficiency(
    rng: Generator,
    bmp_eff_df: pd.DataFrame,
    cps: Union[int, str],
    pollutant: str,
    logger: Any,
) -> float:
    """Sample BMP efficiency for a specific CPS code and pollutant."""
    sub = bmp_eff_df[(bmp_eff_df[COL_CPS].astype(int) == int(cps)) & (bmp_eff_df[COL_POLLUTANT] == pollutant)]
    row = sub.iloc[0]
    stats = {
        k: row[k]
        for k in row.index
        if k in (COL_MEAN, COL_SD, COL_MIN, COL_MAX) or (str(k).startswith(PERCENTILE_PREFIX) and str(k)[1:].isdigit())
    }
    return sample_from_stats(rng, stats, kind="efficiency", verbose_logger=logger, ctx=f"cps={cps},pollutant={pollutant}")


def sample_yield(
    rng: Generator,
    pollutant_yield_df: pd.DataFrame,
    pid: Union[int, str],
    pollutant: str,
    logger: Any,
) -> float:
    """Sample baseline pollutant yield for a parcel and pollutant."""
    sub = pollutant_yield_df[(pollutant_yield_df[COL_PID].astype(str) == str(pid)) & (pollutant_yield_df[COL_POLLUTANT] == pollutant)]
    row = sub.iloc[0]
    stats = {
        k: row[k]
        for k in row.index
        if k in (COL_MEAN, COL_SD, COL_MIN, COL_MAX) or (str(k).startswith(PERCENTILE_PREFIX) and str(k)[1:].isdigit())
    }
    return sample_from_stats(rng, stats, kind="yield", verbose_logger=logger, ctx=f"pid={pid},pollutant={pollutant}")


def compute_bmp_cost(
    rng: Generator,
    bmp_cost_df: Optional[pd.DataFrame],
    cps: Union[int, str],
    quantity: float,
    logger: Any,
) -> float:
    """Estimate BMP cost in USD from configured cost statistics."""
    if bmp_cost_df is None:
        return 0.0
    sub = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
    if sub.empty:
        return 0.0
    unit_row = sub.iloc[0]
    return compute_bmp_cost_usd(rng, cps, unit_row, quantity, logger)


def simulate_wetland(
    rng: Generator,
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
    parcel_area_ha: Sequence[float],
    parcel_up_idxs: Sequence[List[int]],
    parcel_ids: Sequence[str],
    pollutants: Sequence[str],
    logger: Optional[Any] = None,
) -> None:
    """Simulate wetland BMP behavior and reduce yields across impacted parcels."""
    area_field_ha = float(parcel_area_ha[parcel_idx])

    wet_area_stats = {"min": 0.1, "max": 10.0, "mean": 0.4}
    wet_area = sample_from_stats(rng, wet_area_stats, kind=None, verbose_logger=logger, ctx=f"pid={parcel_ids[parcel_idx]}")
    wet_area = min(wet_area, area_field_ha)

    ratio_stats = {"min": 2.0, "max": 100.0, "mean": 5.0}
    cat_ratio = sample_from_stats(rng, ratio_stats, kind=None, verbose_logger=logger, ctx=f"pid={parcel_ids[parcel_idx]}")
    catchment_area_ha = cat_ratio * wet_area
    impacted_area_ha = wet_area + catchment_area_ha

    up_list = parcel_up_idxs[parcel_idx]
    impacted_idxs = [parcel_idx]
    total_available_ha = area_field_ha
    if impacted_area_ha > area_field_ha and len(up_list):
        for up_idx in up_list:
            impacted_idxs.append(up_idx)
            total_available_ha += float(parcel_area_ha[up_idx])
            if total_available_ha >= impacted_area_ha:
                break

    if impacted_area_ha > total_available_ha:
        impacted_area_ha = total_available_ha
        cat_ratio = max(0.0, (impacted_area_ha - wet_area) / max(wet_area, 1e-9))

    bmp_rec[OUTPUT_WETLAND_AREA] = wet_area
    bmp_rec[OUTPUT_CATCHMENT_RATIO] = cat_ratio
    bmp_rec[OUTPUT_IMPACTED_PIDS] = ",".join(
        [parcel_ids[idx] for idx in impacted_idxs] if len(impacted_idxs) > 1 else []
    )

    remaining = impacted_area_ha
    for p_idx in impacted_idxs:
        A = float(parcel_area_ha[p_idx])
        if remaining <= 0:
            frac = 0.0
        elif remaining < A:
            frac = remaining / A
        else:
            frac = 1.0

        for pol_idx, pollutant in enumerate(pollutants):
            y = float(yields[p_idx, pol_idx])
            reduction = y * (A * frac) * eff[pol_idx]
            bmp_outputs[OUTPUT_TREATED][pol_idx] += y * (A * frac)
            bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
            y_new = y - reduction / A
            yields[p_idx, pol_idx] = max(0.0, y_new)

        remaining -= A


def simulate_grassed(
    rng: Generator,
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
    parcel_area_ha: Sequence[float],
    parcel_perim_m: Sequence[float],
    cfg: Dict[str, Any],
    pollutants: Sequence[str],
    logger: Optional[Any] = None,
 ) -> None:
    """Simulate a grassed waterway or buffer BMP and update yield reductions."""
    perim_m = float(parcel_perim_m[parcel_idx])

    frac_stats = {"min": 0.1, "max": 0.5, "mean": 0.25}
    frac = sample_from_stats(rng, frac_stats, kind=None, verbose_logger=logger, ctx=f"pid={parcel_idx}")
    length_m = perim_m * frac
    depth_m = float(cfg.get(CFG_BUFFER_DEPTH_FT, 35.0)) * FT_TO_M
    area_ha = (length_m * depth_m) / 10000.0
    bmp_rec[OUTPUT_LINEAR_LENGTH] = length_m
    bmp_rec[OUTPUT_BUFFER_AREA] = area_ha
    bmp_rec[OUTPUT_PORTION_TREATED] = frac

    A = float(parcel_area_ha[parcel_idx])
    for pol_idx, pollutant in enumerate(pollutants):
        y = float(yields[parcel_idx, pol_idx])
        reduction = y * (A * frac) * eff[pol_idx]
        bmp_outputs[OUTPUT_TREATED][pol_idx] += y * (A * frac)
        bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
        y_new = y - reduction / A
        yields[parcel_idx, pol_idx] = max(0.0, y_new)


def simulate_infield(
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
    parcel_area_ha: Sequence[float],
    pollutants: Sequence[str],
) -> None:
    """Simulate an in-field BMP and update the parcel yield state."""
    A = float(parcel_area_ha[parcel_idx])
    for pol_idx, pollutant in enumerate(pollutants):
        y = float(yields[parcel_idx, pol_idx])
        reduction = y * A * eff[pol_idx]
        bmp_outputs[OUTPUT_TREATED][pol_idx] += y * A
        bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
        y_new = y - reduction / A
        yields[parcel_idx, pol_idx] = max(0.0, y_new)
