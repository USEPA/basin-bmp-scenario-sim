# src/bmp.py
#from __future__ import annotations  # Allows using class names as hints before they are defined
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

from .constants import (
    CFG_BUFFER_DEPTH_FT,
    CFG_BMP_SEL,
    BMP_CPS_NAME_MAP,
    COL_CPS,
    COL_PROBABILITY,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_PORTION_TREATED,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
    OUTPUT_WETLAND_AREA,
    DATA_BMP_COST,
    DATA_CPS,
    DEFAULT_BUFFER_DEPTH_FT,
)

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

FT_TO_M = 0.3048  # meters per foot


def _select_bmp_type(self: "Model") -> int:
    """Choose a BMP type code from the probability distribution."""
    idx = self.rng.choice(len(self.bmp_cps), p=self.bmp_selection_probs)
    cps = int(self.bmp_cps[idx])
    self.logger.debug(f"selected bmp {cps} ({self._get_bmp_name(cps)})")
    return cps


def _get_bmp_name(self: "Model", cps: Union[int, str]) -> str:
    """Return the human-readable name for the BMP CPS code."""
    key = int(cps)
    return BMP_CPS_NAME_MAP.get(key, f"CPS {key}")


def _sample_efficiency(self: "Model", cps: Union[int, str], pol_idx: int) -> float:
    """Sample BMP efficiency for a specific CPS code and pollutant."""
    stats = self.bmp_efficiency_stats[int(cps)][pol_idx]
    eff = self._sample_from_stats(stats, kind="efficiency")
    self.logger.debug(f"selected efficiency value {eff:.2f} for pollutant={self.pollutants[pol_idx]}")
    return eff


def _simulate_wetland(
    self: "Model",
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
    cps: Union[int, str] = 656,
) -> None:
    """Simulate wetland BMP behavior and reduce yields across impacted parcels."""
    self.logger.debug("calling simulate_wetland")

    # wetland area
    area_field_ha = float(self.parcel_area_ha[parcel_idx])
    wet_area_stats = {"min": 0.1, "p25": 0.4, "p50": 0.81, "p75": 2.0, "max": 4.0}  # heuristic
    wet_area = self._sample_from_stats(stats=wet_area_stats, kind=None)
    wet_area = min(wet_area, area_field_ha)
    self.logger.debug(f"selected wetland area of {wet_area:.2f} ha in parcel idx={parcel_idx} of area={area_field_ha:.2f} ha")

    # catchment area ratio
    ratio_stats = {"min": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "max": 100.0}  # heuristic
    cat_ratio = self._sample_from_stats(stats=ratio_stats, kind=None)
    cat_ratio = max(0.0, float(cat_ratio))

    # impacted parcels list (field + upgradient parcels until ratio requirement)
    impacted_idxs: List[int] = [parcel_idx]
    impacted_area_ha: float = wet_area * (1.0 + cat_ratio)
    total_available_ha = float(self.parcel_area_ha[parcel_idx])

    # Accumulate up-gradient parcels until enough area is impacted (or exhausted)
    for up_idx in self.parcel_up_idxs[parcel_idx]:
        if up_idx not in impacted_idxs:
            impacted_idxs.append(up_idx)
            total_available_ha += float(self.parcel_area_ha[up_idx])
            self.logger.debug(
                f"added upgradient parcel (pid={self.parcel_ids[up_idx]}) with area {self.parcel_area_ha[up_idx]:.2f} ha "
                "to wetland-impacted parcels"
            )
            if total_available_ha >= impacted_area_ha:
                break

    if impacted_area_ha > total_available_ha:
        self.logger.debug(
            f"total available upgradient area ({total_available_ha:.2f} ha) < impacted area (wetland+catchment) ({impacted_area_ha:.2f} ha)"
        )
        impacted_area_ha = total_available_ha
        cat_ratio = max(0.0, (impacted_area_ha - wet_area) / max(wet_area, 1e-9))
        self.logger.debug(f"reduced impacted area to {impacted_area_ha:.2f} ha and catchment ratio to {cat_ratio:.2f}")

    bmp_rec[OUTPUT_WETLAND_AREA] = float(wet_area)
    bmp_rec[OUTPUT_CATCHMENT_RATIO] = float(cat_ratio)
    bmp_rec[OUTPUT_IMPACTED_PIDS] = ",".join([self.parcel_ids[idx] for idx in impacted_idxs] if len(impacted_idxs) > 1 else [])

    remaining = impacted_area_ha
    for p_idx in impacted_idxs:
        A = float(self.parcel_area_ha[p_idx])
        if remaining <= 0:
            frac = 0.0
        elif remaining < A:
            frac = remaining / A
        else:
            frac = 1.0
        self.logger.debug(
            f"processing wetland-impacted parcel pid={self.parcel_ids[p_idx]}, area={A:.2f} ha, fraction draining={frac:.2f}"
        )

        for pol_idx, pollutant in enumerate(self.pollutants):
            y = float(yields[p_idx, pol_idx])
            reduction = y * (A * frac) * eff[pol_idx]
            treated = y * (A * frac)
            bmp_outputs[OUTPUT_TREATED][pol_idx] += treated
            bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
            y_new = y - reduction / A
            yields[p_idx, pol_idx] = max(0.0, y_new)

        remaining -= A


def _simulate_grassed(
    self: "Model",
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
) -> None:
    """Simulate a grassed waterway or buffer BMP and update yield reductions."""
    self.logger.debug("calling simulate_grassed")

    # Determine linear length as a fraction of parcel perimeter
    perim_m = float(self.parcel_perim_m[parcel_idx])
    frac_stats = {"min": 0.1, "max": 0.3, "mean": 0.2}  # heuristic
    perim_frac = self._sample_from_stats(stats=frac_stats, kind=None)
    length_m = perim_m * perim_frac
    self.logger.debug(
        f"grassed buffer length={length_m:.2f} m from fraction={perim_frac:.2f} of perimeter={perim_m:.2f} m"
    )

    # Depth and area
    depth_ft = float(self.cfg.get(CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT))
    depth_m = depth_ft * FT_TO_M
    area_ha = (length_m * depth_m) / 10000.0
    self.logger.debug(f"grassed buffer depth={depth_ft:.2f} ft ({depth_m:.2f} m), area={area_ha:.4f} ha")

    # Portion treated
    frac_stats = {"min": 0.2, "max": 0.4, "mean": 0.3}  # heuristic
    frac_treated = self._sample_from_stats(stats=frac_stats, kind=None)

    # Update record and outputs
    bmp_rec[OUTPUT_LINEAR_LENGTH] = float(length_m)
    bmp_rec[OUTPUT_BUFFER_AREA] = float(area_ha)
    bmp_rec[OUTPUT_PORTION_TREATED] = float(frac_treated)

    A = float(self.parcel_area_ha[parcel_idx])
    for pol_idx, pollutant in enumerate(self.pollutants):
        y = float(yields[parcel_idx, pol_idx])
        reduction = y * (A * frac_treated) * eff[pol_idx]
        bmp_outputs[OUTPUT_TREATED][pol_idx] += y * (A * frac_treated)
        bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
        y_new = y - reduction / A
        yields[parcel_idx, pol_idx] = max(0.0, y_new)


def _simulate_infield(
    self: "Model",
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
) -> None:
    """Simulate an in-field BMP and update the parcel yield state."""
    self.logger.debug("calling _simulate_infield")

    A = float(self.parcel_area_ha[parcel_idx])
    for pol_idx, pollutant in enumerate(self.pollutants):
        y = float(yields[parcel_idx, pol_idx])
        reduction = y * A * eff[pol_idx]
        bmp_outputs[OUTPUT_TREATED][pol_idx] += y * A
        bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
        y_new = y - reduction / A
        yields[parcel_idx, pol_idx] = max(0.0, y_new)


def _get_bmp_selection_probs(self: "Model", bmp_sel_path: Optional[str]) -> pd.DataFrame:
    """Return BMP type selection probabilities.

    If an explicit probability file is provided via cfg, use it.
    Otherwise derive weights from estimated costs so lower-cost BMPs are more likely.
    """
    if bmp_sel_path:
        df = pd.read_csv(bmp_sel_path)
        df.columns = [c.lower() for c in df.columns]
        df = df[df[COL_CPS].astype(int).isin(self.data[DATA_CPS])].copy()
        if COL_PROBABILITY not in df.columns and "pr" in df.columns:
            df[COL_PROBABILITY] = df["pr"]
        elif COL_PROBABILITY not in df.columns and "p" in df.columns:
            df[COL_PROBABILITY] = df["p"]
        s = df[COL_PROBABILITY].sum()
        if s <= 0:
            raise ValueError(f"{CFG_BMP_SEL} probabilities sum to zero or negative")
        df[COL_PROBABILITY] = df[COL_PROBABILITY] / s
        self.logger.debug(
            f"Loaded explicit BMP selection probabilities from {bmp_sel_path}: "
            f"{df[[COL_CPS, COL_PROBABILITY]].to_dict(orient='records')}"
        )
        return df[[COL_CPS, COL_PROBABILITY]]
    else:
        if self.data[DATA_BMP_COST] is None:
            probs = np.full(len(self.data[DATA_CPS]), 1.0 / len(self.data[DATA_CPS]))
            return pd.DataFrame({COL_CPS: self.data[DATA_CPS], COL_PROBABILITY: probs})
        else:
            df = self._estimate_costs_for_probabilities()
            return df