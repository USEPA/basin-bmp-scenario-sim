#from __future__ import annotations # Allows using class names as hints before they are defined
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, TYPE_CHECKING
if TYPE_CHECKING: from model import Model


from .constants import (
    COL_CPS,
    DATA_AVG_PERIM_M,
    DATA_AVG_AREA_HA,
    DATA_BMP_COST,
    DATA_CPS,
    COL_PROBABILITY,
    COL_UNIT,
)


def _get_bmp_cost(
    self: Model,
    cps: Union[int, str],
    quantity: float,
) -> float:
    """Estimate BMP cost in USD from configured cost statistics."""
    self.logger.debug(f"calling compute_bmp_cost")
    
    bmp_cost_df = self.data[DATA_BMP_COST]
    bmp_cost_df = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
    if bmp_cost_df.empty:
        self.logger.debug(f"no cost entry found for cps={cps}; returning cost=$0.0")
        return 0.0
    row = bmp_cost_df.iloc[0] # TODO - handle multiple rows per CPS with different stats? or validate that there's only one row per CPS in the input data
    unit = str(row[COL_UNIT]).lower().strip()
    stats: Dict[str, float] = {
        k: row[k]
        for k in row.index
        if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
    }
    rate_value = self._sample_from_stats(stats, kind=None)
    self.logger.debug(f"sampled cost rate {rate_value:.4f}")
    if unit in ("usd/ha","usd per ha","usd_per_ha","usd per unit area"):
        if cps in (656,657):
            area_ha = float(min(0.8, self.data[DATA_AVG_AREA_HA]))
            cost_total = rate_value * area_ha
        else:
            area_ha = float(self.data[DATA_AVG_AREA_HA])
            cost_total = rate_value * area_ha
    elif unit in ("usd/m","usd per m","usd_per_m","usd per unit length"):
        length_m = float(0.2 * self.data[DATA_AVG_PERIM_M])
        cost_total = rate_value * length_m
    elif unit in ("usd/project","usd per project","usd_per_project"):
        count = 1
        cost_total = rate_value * count
    else:
        cost_total = rate_value
    self.logger.debug(f"computed cost using rate{rate_value:.4f}, quantity={quantity:.4f} => cost={cost_total:.2f}")
    return cost_total


def _select_cost_rate_median(
    self: Model,
    row: pd.Series, 
    cps: Optional[Union[int, str]] = None) -> float:
    """Select a representative BMP cost rate for probability estimation.
    If a median percentile exists, use it directly; otherwise sample from stats.
    """
    self.logger.debug(f"calling _select_cost_rate_median")
    stats: Dict[str, float] = {
        k: row[k]
        for k in row.index
        if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
    }
    rate_value = None
    stats = {str(k).lower(): v for k, v in stats.items()}
    if "p50" in stats or "median" in stats:
        rate_value = float(stats.get("p50") or stats.get("median"))
    elif "mean" in stats or "average" in stats or "avg" in stats:
        rate_value = float(
            stats.get("mean") or stats.get("average") or stats.get("avg")
        )
    else:
        rate_min = float(
            stats.get("min") or stats.get("minimum") or stats.get("p0")
        )
        rate_max = float(
            stats.get("max") or stats.get("maximum") or stats.get("p100")
        )
        rate_value = (rate_min + rate_max) / 2.0
    self.logger.debug(f"selected cost rate {rate_value:.4f}")
    if rate_value is None:
        raise ValueError(f"Could not determine cost rate for cps={cps} from stats={stats}")
    return rate_value


def _estimate_costs_for_probabilities(
    self: Model,
    ) -> pd.DataFrame:
    """Estimate BMP selection probabilities using inverse expected cost.

    This heuristic transforms cost estimates into selection probability weights,
    favoring lower-cost BMP types when explicit probabilities are not provided.
    """
    self.logger.debug(f"calling _estimate_costs_for_probabilities")
    rows: list[Dict[str, float]] = []
    for cps in list(set(self.data[DATA_CPS])):
        bmp_cost_df = self.data[DATA_BMP_COST]
        bmp_cost_df = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
        if bmp_cost_df.empty:
            print(f"  no cost entry found for cps={cps}; skipping probability estimation for this CPS")
            rows.append({"cps": int(cps), "est_total_cost": float(0.0)})
            continue
        row = bmp_cost_df.iloc[0] # TODO - handle multiple rows per CPS with different stats? or validate that there's only one row per CPS in the input data
        unit = str(row[COL_UNIT]).lower().strip()
        stats: Dict[str, float] = {
            k: row[k]
            for k in row.index
            if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
        }
        rate_value = None
        if "p50" in {k.lower():v for k,v in stats.items()} or 'median' in {k.lower():v for k,v in stats.items()}:
            median = float(stats.get("p50") or stats.get("P50") or stats.get("median"))
            rate_value = median
        elif "mean" in stats in stats:
            mean = float(stats.get("mean") or stats.get("average") or stats.get("avg"))
            rate_value = mean
        else:
            rate_min = float(stats.get("min") or stats.get("minimum") or stats.get("p0"))
            rate_max = float(stats.get("max") or stats.get("maximum") or stats.get("p100"))
            rate_value = (rate_min + rate_max) / 2.0
        self.logger.debug(f"selected cost rate {rate_value:.4f}")
        if rate_value is None:
            raise ValueError(f"Could not determine cost rate for cps={cps} from stats={stats}")
        if unit in ("usd/ha","usd per ha","usd_per_ha","usd per unit area"):
            if cps in (656,657):
                area_ha = min(0.8, self.data[DATA_AVG_AREA_HA])
            else:
                area_ha = float(self.data[DATA_AVG_AREA_HA])
            total = rate_value * area_ha
        elif unit in ("usd/m","usd per m","usd_per_m","usd per unit length"):
            length_m = float(0.2 * self.data[DATA_AVG_PERIM_M])
            total = rate_value * length_m
        elif unit in ("usd/project","usd per project","usd_per_project"):
            count = 1.0
            total = rate_value * count
        else:
            total = rate_value
        if total < 0:
            raise ValueError(f"Estimated total cost < 0 for cps {cps}")
        rows.append({"cps": int(cps), "est_total_cost": float(total)})

    df = pd.DataFrame(rows)
    df["est_total_cost"] = df["est_total_cost"].clip(lower=0.01)
    df['est_total_cost'] = df['est_total_cost'].replace(np.nan, 0.01)
    if df.empty:
        raise ValueError("Could not estimate costs for probability computation")
    inv = 1.0 / df["est_total_cost"].values
    probs = inv / inv.sum()
    df[COL_PROBABILITY] = probs
    self.logger.debug(f"estimated probabilities for cps: {df[[COL_CPS, COL_PROBABILITY]].to_dict(orient='records')}")
    return df[[COL_CPS, COL_PROBABILITY]]