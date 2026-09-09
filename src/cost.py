"""BMP costing and cost-based selection helpers.

This module estimates BMP placement costs from cost tables and derives
selection probabilities that favor lower-cost BMPs when configured to do so.
"""

from __future__ import annotations

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
    FT_TO_M,
)
from .logging_utils import log_scope
from .input_distributions import row_unit, stats_from_row

# Code-level constants used ONLY for selection-time average-cost heuristics
PROB_EST_WETLAND_MAX_AREA_HA: float = 0.8
PROB_EST_BUFFER_PERIM_FRACTION: float = 0.2


def _finite_numeric_row_values(row: pd.Series) -> Dict[str, float]:
    """Return finite numeric row values keyed by normalized column name.

    Blank cells in standardized input tables are commonly represented as NaN.
    Those cells must behave as absent statistics rather than overriding other
    valid distribution parameters.

    Parameters
    ----------
    row : pd.Series
        Input table row.

    Returns
    -------
    Dict[str, float]
        Finite numeric row values keyed by normalized column name.
    """
    values: Dict[str, float] = {}
    for key, raw_value in row.items():
        if pd.isna(raw_value):
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        values[str(key).lower()] = numeric_value
    return values


def _canonical_cost_unit(unit: object) -> str:
    """Return the canonical cost unit label for a cost-table row.

    Parameters
    ----------
    unit : object
        Unit metadata value from a BMP cost row.

    Returns
    -------
    str
        Canonical unit label when recognized, otherwise a normalized string.

    Notes
    -----
    Cost rows should already be normalized by ``row_unit`` where possible.
    This helper exists to map supported same-scale aliases onto a small set of
    unit families used by costing logic.
    """
    canonical = row_unit({"unit": unit})
    if canonical is None:
        return str(unit).strip().lower()

    label = str(canonical).strip().lower()

    if label in {
        "usd/ha",
        "usd per ha",
        "usd_per_ha",
        "usd per unit area",
    }:
        return "usd/ha"

    if label in {
        "usd/m",
        "usd per m",
        "usd_per_m",
        "usd per unit length",
    }:
        return "usd/m"

    if label in {
        "usd/project",
        "usd per project",
        "usd_per_project",
    }:
        return "usd/project"

    return label


def _sample_cost_rate(self: "Model", row: pd.Series, cps: Union[int, str]) -> float:
    """Sample one unit-aware BMP cost rate from a cost row.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    row : pd.Series
        Cost table row.
    cps : int or str
        BMP CPS code used for logging.

    Returns
    -------
    float
        Sampled cost rate in canonical internal units.

    Raises
    ------
    ValueError
        If no valid numeric statistics are available or sampling returns a
        non-finite value.
    """
    stats = stats_from_row(row, exclude=(COL_CPS, "cps_name"))
    if not stats:
        raise ValueError(f"No finite BMP cost value or distribution statistics for cps={cps}")

    rate_value = float(self._sample_from_stats(stats, kind=None))
    if not np.isfinite(rate_value):
        raise ValueError(f"Sampled non-finite BMP cost rate for cps={cps}: {rate_value}")

    return rate_value


def _representative_cost_rate(self: "Model", row: pd.Series, cps: Optional[Union[int, str]] = None) -> float:
    """Select a representative unit-aware cost rate from one cost row.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    row : pd.Series
        Cost table row.
    cps : int or str, optional
        BMP CPS code used for logging.

    Returns
    -------
    float
        Representative cost rate in canonical internal units.

    Raises
    ------
    ValueError
        If no representative finite rate can be inferred from the row.
    """
    with log_scope(label=f"representative_cost_rate cps={cps}", logger=self.logger):
        self.logger.verbose("calling _representative_cost_rate")

        cols = stats_from_row(row, exclude=(COL_CPS, "cps_name"))
        if not cols:
            raise ValueError(f"Could not determine finite cost rate for cps={cps}")

        if "value" in cols:
            rate_value = cols["value"]
        elif "p50" in cols:
            rate_value = cols["p50"]
        elif "mean" in cols:
            rate_value = cols["mean"]
        else:
            rate_min = cols.get("min")
            rate_max = cols.get("max")
            if rate_min is None or rate_max is None:
                raise ValueError(f"Could not determine finite cost rate for cps={cps}")
            rate_value = (rate_min + rate_max) / 2.0

        if rate_value is None or not np.isfinite(float(rate_value)):
            raise ValueError(f"Could not determine finite cost rate for cps={cps}")

        self.logger.verbose(f"selected representative cost rate {rate_value:.4f} for cps={cps}")
        return float(rate_value)


def _scale_cost_rate_to_total(
    self: "Model",
    cps: Union[int, str],
    rate_value: float,
    unit: str,
    quantity: float = 0.0,
    use_selection_heuristics: bool = False,
) -> float:
    """Convert a unit cost rate into a total BMP cost.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    cps : int or str
        BMP CPS code.
    rate_value : float
        Cost rate in canonical internal units.
    unit : str
        Canonicalized cost unit label.
    quantity : float, optional
        Realized BMP quantity used for costing, such as area or length.
    use_selection_heuristics : bool, optional
        Whether to use representative selection-time geometry heuristics
        instead of realized placement geometry.

    Returns
    -------
    float
        Total BMP cost in USD.
    """
    cost_total: float

    if unit == "usd/ha":
        if use_selection_heuristics or not (quantity and quantity > 0):
            if int(cps) in (656, 657):
                area_ha = float(min(PROB_EST_WETLAND_MAX_AREA_HA, self.data[DATA_AVG_AREA_HA]))
            else:
                area_ha = float(self.data[DATA_AVG_AREA_HA])
        else:
            area_ha = float(quantity)
        cost_total = rate_value * area_ha

    elif unit == "usd/m":
        if use_selection_heuristics or not (quantity and quantity > 0):
            length_m = float(PROB_EST_BUFFER_PERIM_FRACTION * self.data[DATA_AVG_PERIM_M])
        else:
            depth_ft = float(self.cfg.get(CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT))
            depth_m = depth_ft * FT_TO_M
            area_m2 = float(quantity) * 10000.0
            length_m = area_m2 / max(depth_m, 1e-9)
        cost_total = rate_value * length_m

    elif unit == "usd/project":
        cost_total = rate_value * 1.0

    else:
        cost_total = rate_value

    if not np.isfinite(cost_total):
        raise ValueError(f"Computed non-finite BMP cost for cps={cps}: {cost_total}")

    return float(cost_total)


def _get_bmp_cost(
    self: "Model",
    cps: Union[int, str],
    quantity: float,
) -> float:
    """Estimate the cost of one BMP placement.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    cps : int or str
        BMP CPS code.
    quantity : float
        Realized BMP quantity used for costing, such as area or length.

    Returns
    -------
    float
        Estimated BMP placement cost in USD.
    """
    with log_scope(label=f"get_bmp_cost cps={cps}", logger=self.logger):
        self.logger.verbose("calling _get_bmp_cost")
        bmp_cost_df = self.data.get(DATA_BMP_COST)
        if bmp_cost_df is None or bmp_cost_df.empty:
            self.logger.verbose("no BMP cost table configured; returning cost=$0.0")
            return 0.0

        bmp_cost_df = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
        if bmp_cost_df.empty:
            self.logger.verbose(f"no cost entry found for cps={cps}; returning cost=$0.0")
            return 0.0

        row = bmp_cost_df.iloc[0]  # Assumes one row per CPS; validated upstream
        unit = _canonical_cost_unit(row.get(COL_UNIT))
        rate_value = _sample_cost_rate(self, row, cps=cps)

        self.logger.verbose(f"sampled cost rate {rate_value:.4f} for cps={cps}, unit={unit}")

        cost_total = _scale_cost_rate_to_total(
            self=self,
            cps=cps,
            rate_value=rate_value,
            unit=unit,
            quantity=float(quantity),
            use_selection_heuristics=False,
        )

        self.logger.verbose(
            f"computed cost for cps={cps} using rate={rate_value:.4f}, unit='{unit}', "
            f"realized_quantity={quantity:.4f} => cost={cost_total:.2f}"
        )
        return float(cost_total)


def _select_cost_rate_median(
    self: "Model",
    row: pd.Series,
    cps: Optional[Union[int, str]] = None,
) -> float:
    """Select a representative cost rate from one cost row.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    row : pd.Series
        Cost table row.
    cps : int or str, optional
        BMP CPS code used for logging.

    Returns
    -------
    float
        Representative cost rate.
    """
    return _representative_cost_rate(self, row=row, cps=cps)


def _estimate_costs_for_probabilities(self: "Model") -> pd.DataFrame:
    """Estimate BMP selection probabilities from cost heuristics.

    The function computes a representative total cost for every configured BMP
    type and assigns probabilities inversely proportional to those costs. Cost-
    based selection requires complete cost coverage: a missing cost table or a
    missing CPS row is an input error rather than a signal to invent a small
    placeholder cost.

    Parameters
    ----------
    self : Model
        Active simulation model instance.

    Returns
    -------
    pandas.DataFrame
        Two-column dataframe containing CPS codes and normalized selection
        probabilities.

    Raises
    ------
    ValueError
        If cost-based selection lacks a cost table, any configured CPS lacks a
        cost row, or a representative finite cost cannot be computed.
    """
    with log_scope(label="estimate_costs_for_probabilities", logger=self.logger):
        self.logger.verbose("calling _estimate_costs_for_probabilities")

        bmp_cost_df = self.data.get(DATA_BMP_COST)
        if bmp_cost_df is None or bmp_cost_df.empty:
            raise ValueError(
                "Cost-based BMP selection requires BMP cost data for every configured CPS"
            )

        configured_cps = sorted(set(int(x) for x in self.data[DATA_CPS]))
        available_cps = set(
            pd.to_numeric(bmp_cost_df[COL_CPS], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        missing_cps = [cps for cps in configured_cps if cps not in available_cps]
        if missing_cps:
            raise ValueError(
                "Cost-based BMP selection requires a cost entry for every configured CPS; "
                f"missing cost entries for cps={missing_cps}"
            )

        rows: list[Dict[str, float]] = []
        for cps in configured_cps:
            sub = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
            row = sub.iloc[0]

            unit = _canonical_cost_unit(row.get(COL_UNIT))
            rate_value = _representative_cost_rate(self, row, cps=cps)

            total = _scale_cost_rate_to_total(
                self=self,
                cps=cps,
                rate_value=rate_value,
                unit=unit,
                quantity=0.0,
                use_selection_heuristics=True,
            )

            if not np.isfinite(total):
                raise ValueError(f"Computed non-finite representative BMP cost for cps={cps}: {total}")

            rows.append({"cps": int(cps), "est_total_cost": float(max(total, 0.01))})

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Could not estimate costs for probability computation")

        inv = 1.0 / df["est_total_cost"].values
        probs = inv / inv.sum()
        if not np.all(np.isfinite(probs)):
            raise ValueError("Cost-based BMP selection produced non-finite probabilities")

        df[COL_PROBABILITY] = probs

        self.logger.verbose(
            "Probability estimation constants: "
            f"PROB_EST_WETLAND_MAX_AREA_HA={PROB_EST_WETLAND_MAX_AREA_HA}, "
            f"PROB_EST_BUFFER_PERIM_FRACTION={PROB_EST_BUFFER_PERIM_FRACTION}"
        )
        self.logger.verbose(
            f"estimated probabilities: {df[[COL_CPS, COL_PROBABILITY]].to_dict(orient='records')}"
        )
        return df[[COL_CPS, COL_PROBABILITY]]