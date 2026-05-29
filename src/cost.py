# src/cost.py
#from __future__ import annotations  # Allows using class names as hints before they are defined
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

from .constants import (
    COL_CPS,
    DATA_AVG_PERIM_M,
    DATA_AVG_AREA_HA,
    DATA_BMP_COST,
    DATA_CPS,
    COL_PROBABILITY,
    COL_UNIT,
    CFG_BUFFER_DEPTH_FT,
    DEFAULT_BUFFER_DEPTH_FT,
)

# Code-level constants used ONLY for selection-time average-cost heuristics
# (kept hardcoded by design per user request)
PROB_EST_WETLAND_MAX_AREA_HA: float = 0.8
PROB_EST_BUFFER_PERIM_FRACTION: float = 0.2

FT_TO_M = 0.3048  # meters per foot


def _get_bmp_cost(
    self: "Model",
    cps: Union[int, str],
    quantity: float,
) -> float:
    """Estimate BMP cost (USD) for a realized BMP instance.

    Behavior:
    - USD/ha: use realized area (ha) if provided; otherwise fall back to average-area heuristic.
    - USD/m: if quantity > 0, interpret quantity as area_ha for buffers and convert to length using depth.
             otherwise fall back to average-perimeter heuristic (fraction * avg_perim_m).
    - USD/project: multiply rate by 1 project.
    - Unitless: return rate.

    Notes
    -----
    - This function is used at scenario runtime for costing reported BMPs.
    - Average-based heuristics remain in the probability estimation path (selection-time),
      not here.
    """
    self.logger.debug("calling _get_bmp_cost")
    bmp_cost_df = self.data[DATA_BMP_COST]
    bmp_cost_df = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
    if bmp_cost_df.empty:
        self.logger.debug(f"no cost entry found for cps={cps}; returning cost=$0.0")
        return 0.0

    row = bmp_cost_df.iloc[0]  # Assumes one row per CPS; validated upstream
    unit = str(row[COL_UNIT]).lower().strip()

    stats: Dict[str, float] = {
        k: row[k]
        for k in row.index
        if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
    }
    rate_value = float(self._sample_from_stats(stats, kind=None))
    self.logger.debug(f"sampled cost rate {rate_value:.4f} for cps={cps}, unit={unit}")

    cost_total: float
    if unit in ("usd/ha", "usd per ha", "usd_per_ha", "usd per unit area"):
        if quantity and quantity > 0:
            area_ha = float(quantity)
        else:
            # Fallback to average-area heuristic only when no realized quantity is given
            if int(cps) in (656, 657):
                area_ha = float(min(PROB_EST_WETLAND_MAX_AREA_HA, self.data[DATA_AVG_AREA_HA]))
            else:
                area_ha = float(self.data[DATA_AVG_AREA_HA])
        cost_total = rate_value * area_ha

    elif unit in ("usd/m", "usd per m", "usd_per_m", "usd per unit length"):
        length_m: float
        if quantity and quantity > 0:
            # quantity represents area_ha for grassed buffers; convert to length via depth
            # length_m = area_m2 / depth_m
            depth_ft = float(self.cfg.get(CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT))
            depth_m = depth_ft * FT_TO_M
            area_m2 = float(quantity) * 10000.0
            length_m = area_m2 / max(depth_m, 1e-9)
        else:
            # Fallback to average-perimeter heuristic (selection-time heuristic reused)
            length_m = float(PROB_EST_BUFFER_PERIM_FRACTION * self.data[DATA_AVG_PERIM_M])
        cost_total = rate_value * length_m

    elif unit in ("usd/project", "usd per project", "usd_per_project"):
        count = 1.0
        cost_total = rate_value * count
    else:
        cost_total = rate_value

    self.logger.debug(
        f"computed cost for cps={cps} using rate={rate_value:.4f}, unit='{unit}', "
        f"realized_quantity={quantity:.4f} => cost={cost_total:.2f}"
    )
    return float(cost_total)


def _select_cost_rate_median(
    self: "Model",
    row: pd.Series,
    cps: Optional[Union[int, str]] = None,
) -> float:
    """Select a representative BMP cost rate for probability estimation.

    If a median percentile exists, use it directly; otherwise mean; otherwise mid-point of [min, max].
    """
    self.logger.debug("calling _select_cost_rate_median")
    stats: Dict[str, float] = {
        k: row[k]
        for k in row.index
        if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
    }
    stats = {str(k).lower(): v for k, v in stats.items()}

    if "p50" in stats or "median" in stats:
        rate_value = float(stats.get("p50") or stats.get("median"))
    elif "mean" in stats or "average" in stats or "avg" in stats:
        rate_value = float(stats.get("mean") or stats.get("average") or stats.get("avg"))
    else:
        rate_min = float(stats.get("min") or stats.get("minimum") or stats.get("p0"))
        rate_max = float(stats.get("max") or stats.get("maximum") or stats.get("p100"))
        rate_value = (rate_min + rate_max) / 2.0

    if rate_value is None:
        raise ValueError(f"Could not determine cost rate for cps={cps} from stats={stats}")
    self.logger.debug(f"selected representative cost rate {rate_value:.4f} for cps={cps}")
    return float(rate_value)


def _estimate_costs_for_probabilities(self: "Model") -> pd.DataFrame:
    """Estimate BMP selection probabilities using inverse expected cost.

    - Keeps average-cost heuristics hardcoded for selection-time weighting:
      wetlands cap area at PROB_EST_WETLAND_MAX_AREA_HA; linear BMPs use
      PROB_EST_BUFFER_PERIM_FRACTION * avg_perim_m; in-field uses avg_area_ha.
    - Returns a DataFrame with [cps, probability]; probabilities sum to 1.
    """
    self.logger.debug("calling _estimate_costs_for_probabilities")
    rows: list[Dict[str, float]] = []

    for cps in sorted(set(int(x) for x in self.data[DATA_CPS])):
        bmp_cost_df = self.data[DATA_BMP_COST]
        sub = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
        if sub.empty:
            self.logger.warning(f"no cost entry found for cps={cps}; assigning small placeholder cost for probability estimation")
            rows.append({"cps": int(cps), "est_total_cost": float(0.01)})
            continue

        row = sub.iloc[0]
        unit = str(row[COL_UNIT]).lower().strip()
        rate_value = self._select_cost_rate_median(row, cps=cps)

        if unit in ("usd/ha", "usd per ha", "usd_per_ha", "usd per unit area"):
            if cps in (656, 657):
                area_ha = float(min(PROB_EST_WETLAND_MAX_AREA_HA, self.data[DATA_AVG_AREA_HA]))
            else:
                area_ha = float(self.data[DATA_AVG_AREA_HA])
            total = rate_value * area_ha

        elif unit in ("usd/m", "usd per m", "usd_per_m", "usd per unit length"):
            length_m = float(PROB_EST_BUFFER_PERIM_FRACTION * self.data[DATA_AVG_PERIM_M])
            total = rate_value * length_m

        elif unit in ("usd/project", "usd per project", "usd_per_project"):
            total = rate_value * 1.0
        else:
            total = rate_value

        rows.append({"cps": int(cps), "est_total_cost": float(max(total, 0.01))})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Could not estimate costs for probability computation")

    inv = 1.0 / df["est_total_cost"].values
    probs = inv / inv.sum()
    df[COL_PROBABILITY] = probs

    self.logger.debug(
        "Probability estimation constants: "
        f"PROB_EST_WETLAND_MAX_AREA_HA={PROB_EST_WETLAND_MAX_AREA_HA}, "
        f"PROB_EST_BUFFER_PERIM_FRACTION={PROB_EST_BUFFER_PERIM_FRACTION}"
    )
    self.logger.debug(f"estimated probabilities: {df[[COL_CPS, COL_PROBABILITY]].to_dict(orient='records')}")

    return df[[COL_CPS, COL_PROBABILITY]]