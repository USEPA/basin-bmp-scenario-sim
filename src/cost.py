from __future__ import annotations # Allows using class names as hints before they are defined
import pandas as pd
import numpy as np
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from scenario import Model


from .constants import (
    COL_CPS,
    DATA_AVG_PERIM_M,
    DATA_AVG_AREA_HA,
    DATA_BMP_COST,
    DATA_CPS,
    DATA_DELIVERY_RATIOS,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_POLLUTANT_YIELD,
    DATA_POLLUTANTS,
    DATA_RANDOM_SEED,
    DATA_BMP_EFFICIENCY,
    COL_PROBABILITY,
    COL_UNIT,
)


def compute_bmp_cost(
    self: Model,
    bmp_cost_df: Optional[pd.DataFrame],
    cps: Union[int, str],
    quantity: float,
    logger: logging.Logger,
) -> float:
    """Estimate BMP cost in USD from configured cost statistics."""
    logger.debug(f" calling compute_bmp_cost, cps={cps}, quantity={quantity:.4f}")
    if bmp_cost_df is None:
        logger.warning(f"  no BMP cost table provided for cps={cps}; returning cost=0.0")
        return 0.0
    sub = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
    if sub.empty:
        logger.warning(f"  no cost entry found for cps={cps}; returning cost=0.0")
        return 0.0
    unit_row = sub.iloc[0]
    cost = self._compute_bmp_cost_usd(cps, unit_row, quantity, logger)
    logger.debug(f"  computed cost for cps={cps}, quantity={quantity:.4f} => cost={cost:.2f}")
    return cost

def _compute_bmp_cost(
    self: Model,
    cps: Union[int, str],
    unit_row: Optional[pd.Series],
    quantity: float,
    logger: logging.Logger,
) -> float:
    """Compute total BMP cost in USD from sampled unit cost and quantity.

    The function samples a cost rate based on the provided statistics, validates
    that the sampled rate and resulting total are non-negative, and returns the
    USD cost for the given BMP quantity.
    """
    logger.debug(f'calling compute_bmp_cost_usd, cps = {cps}, quantity={quantity}')
    
    df = self.data[DATA_BMP_COST][self.data[DATA_BMP_COST][COL_CPS].astype(int) == int(cps)]
    if df is None:
        logger.debug(f" no cost row available for cps={cps}; returning cost = $0")
        return 0.0
    stats = {k: df.iloc[0][k] for k in df.columns if k in ("mean","sd","min","max") or (str(k).startswith("p") and str(k)[1:].isdigit())}
    logger.debug(f" sampling cost rate for cps={cps} using stats={stats}")
    rate = self._sample_from_stats(stats, kind=None)
    logger.debug(f" sampled cost rate = {rate:.4f}")
    if rate < 0:
        raise ValueError("Negative cost-rate sampled")
    total = rate * quantity
    if total < 0:
        raise ValueError("Negative total cost computed")
    logger.debug(f" total cost = {total:.2f}")
    return float(total)

def _select_cost_rate_median(
    self: Model,
    row: pd.Series, 
    logger: Optional[logging.Logger],
    cps: Optional[Union[int, str]] = None) -> float:
    """Select a representative BMP cost rate for probability estimation.
    If a median percentile exists, use it directly; otherwise sample from stats.
    """
    logger.debug(f" calling _select_cost_rate_median cps={cps} row={row.to_dict()}")
    stats: Dict[str, float] = {
        k: row[k]
        for k in row.index
        if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
    }
    if "p50" in {k.lower():v for k,v in stats.items()}:
        selected = float(stats.get("p50") or stats.get("P50"))
        return selected

    import logging
    logging.getLogger(__name__).debug(f"Sampling cost rate for probability estimate cps={cps} stats={stats}")
    return self._sample_from_stats(stats, kind=None)

def _estimate_costs_for_probabilities(
    self: Model,
    bmp_cost_df: pd.DataFrame,
    cps_list: Sequence[Union[int, str]],
    avg_area_ha: float,
    avg_perim_m: float,
    overrides: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
    """Estimate BMP selection probabilities using inverse expected cost.

    This heuristic transforms cost estimates into selection probability weights,
    favoring lower-cost BMP types when explicit probabilities are not provided.
    """
    
    overrides = overrides or {}
    rows: list[Dict[str, float]] = []
    for cps in cps_list:
        sub = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        unit = str(r[COL_UNIT]).lower().strip()
        rate = self._select_cost_rate_median(r, logging.getLogger(__name__), cps)
        if rate < 0:
            raise ValueError(f"Negative cost-rate for cps {cps}")

        if unit in ("usd/ha","usd per ha","usd_per_ha","usd per unit area"):
            if cps in (656,657):
                area_ha = float(overrides.get("wetland_area_ha_for_prob", min(0.8, avg_area_ha)))
            else:
                area_ha = float(overrides.get("field_area_ha_for_prob", avg_area_ha))
            total = rate * area_ha
        elif unit in ("usd/m","usd per m","usd_per_m","usd per unit length"):
            length_m = float(overrides.get("buffer_length_m_for_prob", 0.2 * avg_perim_m))
            total = rate * length_m
        elif unit in ("usd/project","usd per project","usd_per_project"):
            count = float(overrides.get("project_count_for_prob", 1.0))
            total = rate * count
        else:
            total = rate
        if total < 0:
            raise ValueError(f"Estimated total cost < 0 for cps {cps}")
        rows.append({"cps": int(cps), "est_total_cost": float(total)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Could not estimate costs for probability computation")
    inv = 1.0 / df["est_total_cost"].values
    probs = inv / inv.sum()
    df[COL_PROBABILITY] = probs
    logging.getLogger(__name__).debug(f"Estimated selection probabilities from cost totals: {df.to_dict(orient='records')}")
    return df[[COL_CPS, COL_PROBABILITY]]


def _compute_bmp_cost_usd(
    self: Model,
    cps: Union[int, str],
    unit_row: Optional[pd.Series],
    quantity: float,
    logger: logging.Logger,
) -> float:
    """Compatibility wrapper: compute USD cost for a BMP using existing helper."""
    # Delegate to the internal implementation which samples a rate and multiplies
    return _compute_bmp_cost(self, cps, unit_row, quantity, logger)