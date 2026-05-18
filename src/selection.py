import numpy as np
import pandas as pd
from numpy.random import Generator
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .constants import COL_CPS, COL_UNIT, COL_PROBABILITY
from .sampling import sample_from_stats

def _select_cost_rate_median(rng: Generator, row: pd.Series) -> float:
    """Select a representative BMP cost rate for probability estimation.

    If a median percentile exists, use it directly; otherwise sample from stats.
    """
    stats: Dict[str, float] = {
        k: row[k]
        for k in row.index
        if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
    }
    if "p50" in {k.lower():v for k,v in stats.items()}:
        return float(stats.get("p50") or stats.get("P50"))
    return sample_from_stats(rng, stats, kind=None)

def estimate_costs_for_probabilities(
    rng: Generator,
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
        rate = _select_cost_rate_median(rng, r)
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
    return df[[COL_CPS, COL_PROBABILITY]]
