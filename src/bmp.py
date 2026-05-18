import pandas as pd
from numpy.random import Generator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    return sample_from_stats(rng, stats, kind="efficiency", verbose_logger=logger)


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
    return sample_from_stats(rng, stats, kind="yield", verbose_logger=logger)


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
    pid: Union[int, str],
    eff: Dict[str, float],
    yields_map: Dict[Tuple[str, str], float],
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, Dict[str, float]],
    parcel_record: ParcelRecordFn,
    parcel_up_list: ParcelUpListFn,
    pollutants: List[str],
) -> None:
    """Simulate wetland BMP behavior and reduce yields across impacted parcels."""
    row = parcel_record(pid)
    area_field_ha = float(row[COL_AREA_HA])

    wet_area_stats = {"min": 0.1, "max": 10.0, "mean": 0.4}
    wet_area = sample_from_stats(rng, wet_area_stats, kind=None, verbose_logger=None)
    wet_area = min(wet_area, area_field_ha)

    ratio_stats = {"min": 2.0, "max": 100.0, "mean": 5.0}
    cat_ratio = sample_from_stats(rng, ratio_stats, kind=None, verbose_logger=None)
    catchment_area_ha = cat_ratio * wet_area
    impacted_area_ha = wet_area + catchment_area_ha

    up_list = parcel_up_list(pid)
    impacted_pids = [str(pid)]
    total_available_ha = area_field_ha
    if impacted_area_ha > area_field_ha and len(up_list):
        for up_pid in up_list:
            r = parcel_record(up_pid)
            impacted_pids.append(str(up_pid))
            total_available_ha += float(r[COL_AREA_HA])
            if total_available_ha >= impacted_area_ha:
                break

    if impacted_area_ha > total_available_ha:
        impacted_area_ha = total_available_ha
        cat_ratio = max(0.0, (impacted_area_ha - wet_area) / max(wet_area, 1e-9))

    bmp_rec[OUTPUT_WETLAND_AREA] = wet_area
    bmp_rec[OUTPUT_CATCHMENT_RATIO] = cat_ratio
    bmp_rec[OUTPUT_IMPACTED_PIDS] = ",".join(impacted_pids if len(impacted_pids) > 1 else [])

    remaining = impacted_area_ha
    for p in impacted_pids:
        r = parcel_record(p)
        A = float(r[COL_AREA_HA])
        if remaining <= 0:
            frac = 0.0
        elif remaining < A:
            frac = remaining / A
        else:
            frac = 1.0

        for pollutant in pollutants:
            y = yields_map[(p, pollutant)]
            reduction = y * (A * frac) * eff[pollutant]
            bmp_outputs[OUTPUT_TREATED][pollutant] += y * (A * frac)
            bmp_outputs[OUTPUT_REMOVED][pollutant] += reduction
            y_new = y - reduction / A
            yields_map[(p, pollutant)] = max(0.0, y_new)

        remaining -= A


def simulate_grassed(
    rng: Generator,
    pid: Union[int, str],
    eff: Dict[str, float],
    yields_map: Dict[Tuple[str, str], float],
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, Dict[str, float]],
    parcel_record: ParcelRecordFn,
    cfg: Dict[str, Any],
    pollutants: List[str],
) -> None:
    """Simulate a grassed waterway or buffer BMP and update yield reductions."""
    row = parcel_record(pid)
    perim_m = float(row[COL_PERIM_M])

    frac_stats = {"min": 0.1, "max": 0.5, "mean": 0.25}
    frac = sample_from_stats(rng, frac_stats, kind=None, verbose_logger=None)
    length_m = perim_m * frac
    depth_m = float(cfg.get(CFG_BUFFER_DEPTH_FT, 35.0)) * FT_TO_M
    area_ha = (length_m * depth_m) / 10000.0
    bmp_rec[OUTPUT_LINEAR_LENGTH] = length_m
    bmp_rec[OUTPUT_BUFFER_AREA] = area_ha
    bmp_rec[OUTPUT_PORTION_TREATED] = frac

    A = float(row[COL_AREA_HA])
    for pollutant in pollutants:
        y = yields_map[(str(pid), pollutant)]
        reduction = y * (A * frac) * eff[pollutant]
        bmp_outputs[OUTPUT_TREATED][pollutant] += y * (A * frac)
        bmp_outputs[OUTPUT_REMOVED][pollutant] += reduction
        y_new = y - reduction / A
        yields_map[(str(pid), pollutant)] = max(0.0, y_new)


def simulate_infield(
    pid: Union[int, str],
    eff: Dict[str, float],
    yields_map: Dict[Tuple[str, str], float],
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, Dict[str, float]],
    parcel_record: ParcelRecordFn,
    pollutants: List[str],
) -> None:
    """Simulate an in-field BMP and update the parcel yield state."""
    row = parcel_record(pid)
    A = float(row[COL_AREA_HA])
    for pollutant in pollutants:
        y = yields_map[(str(pid), pollutant)]
        reduction = y * A * eff[pollutant]
        bmp_outputs[OUTPUT_TREATED][pollutant] += y * A
        bmp_outputs[OUTPUT_REMOVED][pollutant] += reduction
        y_new = y - reduction / A
        yields_map[(str(pid), pollutant)] = max(0.0, y_new)
