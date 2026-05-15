import numpy as np
import pandas as pd
from .sampling import sample_from_stats

def _select_cost_rate_median(rng, row):
    stats = {k: row[k] for k in row.index if k in ("mean","sd","min","max") or (str(k).startswith("p") and str(k)[1:].isdigit())}
    if "p50" in {k.lower():v for k,v in stats.items()}:
        return float(stats.get("p50") or stats.get("P50"))
    return sample_from_stats(rng, stats, kind=None)

def estimate_costs_for_probabilities(rng, bmp_cost_df, cps_list, avg_area_ha, avg_perim_m, overrides=None):
    overrides = overrides or {}
    rows = []
    for cps in cps_list:
        sub = bmp_cost_df[bmp_cost_df["cps"].astype(int) == int(cps)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        unit = str(r["unit"]).lower().strip()
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
    df["probability"] = probs
    return df[["cps","probability"]]
