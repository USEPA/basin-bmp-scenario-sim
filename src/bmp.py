from __future__ import annotations # Allows using class names as hints before they are defined
import pandas as pd
import numpy as np
import logging
from numpy.random import Generator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scenario import Model


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
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_BMP_COST,
    DATA_CPS,
)


ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

FT_TO_M = 0.3048  # meters per foot


def _select_bmp_type(
    self: Model
    ) -> int:
    """Choose a BMP type code from the probability distribution."""
    idx = self.rng.choice(len(self.bmp_cps), p=self.bmp_selection_probs)
    cps = int(self.bmp_cps[idx])
    self.logger.debug(f" selected bmp {cps} ({self._get_bmp_name(cps)})")
    return cps


def _get_bmp_name(
    self: Model, 
    cps: Union[int, str]
    ) -> str:
    """Return the human-readable name for the BMP CPS code."""
    key = int(cps)
    return BMP_CPS_NAME_MAP.get(key, f"CPS {key}")


def _sample_efficiency(
    self: Model,
    cps: Union[int, str],
    pol_idx: int,
    ) -> float:
    """Sample BMP efficiency for a specific CPS code and pollutant."""
    stats = self.bmp_efficiency_stats[int(cps)][pol_idx]
    eff = self._sample_from_stats(stats, kind="efficiency")
    self.logger.debug(f" selected efficiency value {eff:.2f} for pollutant={self.pollutants[pol_idx]}")
    return eff


def _simulate_wetland(
    self: Model,
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
    cps: Union[int, str] = 656,
    ) -> None:
    """Simulate wetland BMP behavior and reduce yields across impacted parcels."""
    self.logger.debug(" calling simulate_wetland")

    # wetland area
    area_field_ha = float(self.parcel_area_ha[parcel_idx])
    wet_area_stats = {"min": 0.1, "p25": 0.4, "p50": 0.81, "p75": 2.0, "max": 4.0} # TODO - make these configurable or based on field area 
    wet_area = self._sample_from_stats(stats=wet_area_stats, kind=None)
    wet_area = min(wet_area, area_field_ha)
    self.logger.debug(f" selected wetland area of {wet_area:.2f} ha in parcel idx = {parcel_idx} of area = {area_field_ha:.2f} ha")

    # catchment area ratio
    ratio_stats = {"min": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "max": 100.0} # TODO - make these configurable
    cat_ratio = self._sample_from_stats(stats=ratio_stats, kind=None)
    catchment_area_ha = cat_ratio * wet_area
    impacted_area_ha = wet_area + catchment_area_ha
    self.logger.debug(f" selected catchment:wetland area ratio = {cat_ratio:.2f}")
    self.logger.debug(f" computed catchment area = {catchment_area_ha:.2f} ha, total wetland-impacted area (wetland + catchment) = {impacted_area_ha:.2f} ha")

    up_list = self.parcel_up_idxs[parcel_idx]
    impacted_idxs = [parcel_idx]
    total_available_ha = area_field_ha

    # if the impacted area exceeds the field area, add upgradient parcels until we meet the impacted area or run out of parcels
    if impacted_area_ha > total_available_ha:
        self.logger.debug(f" impacted area (wetland + catchment) ({impacted_area_ha:.2f} ha) > field area ({total_available_ha:.2f} ha)") # here, total_available_ha = area_field_ha
        if len(up_list) == 0:
            self.logger.debug(f" parcel (pid={self.parcel_ids[parcel_idx]}) has no upgradient parcels")
        else:
            self.logger.debug(f" parcel (pid={self.parcel_ids[parcel_idx]}) has {len(up_list)} upgradient parcels "
                              f"with pid(s) = {", ".join(str(pid_up) for pid_up in up_list)} "
                              f"and area(s) = {", ".join(f'{self.parcel_area_ha[up_idx]:.2f}' for up_idx in up_list)} ha")
            for up_idx in up_list:
                impacted_idxs.append(up_idx)
                total_available_ha += float(self.parcel_area_ha[up_idx])
                self.logger.debug(f" added upgradient parcel (pid={self.parcel_ids[up_idx]}) with area {self.parcel_area_ha[up_idx]:.2f} ha to wetland-impacted parcels")
                if total_available_ha >= impacted_area_ha:
                    break

    # reduce catchment area and ratio if total available area from field and upgradient parcels is less than the selected ratio's resultant impacted area (wetland + catchment)
    if impacted_area_ha > total_available_ha:
        self.logger.debug(f" total available area from field and upgradient parcels ({total_available_ha:.2f} ha) < is less than the selected ratio's resultant impacted area (wetland + catchment) ({impacted_area_ha:.2f} ha)")
        impacted_area_ha = total_available_ha
        cat_ratio = max(0.0, (impacted_area_ha - wet_area) / max(wet_area, 1e-9))
        self.logger.debug(f" reduced wetland impacted area (wetland + catchment) to {impacted_area_ha:.2f} ha and reduced catchment ratio to {cat_ratio:.2f}")

    # add wetland area, catchment ratio, and impacted pids to bmp record
    bmp_rec[OUTPUT_WETLAND_AREA] = wet_area
    bmp_rec[OUTPUT_CATCHMENT_RATIO] = cat_ratio
    # If only one parcel is impacted, OUTPUT_IMPACTED_PIDS will be an empty string.
    bmp_rec[OUTPUT_IMPACTED_PIDS] = ",".join(
        [self.parcel_ids[idx] for idx in impacted_idxs] if len(impacted_idxs) > 1 else []
    )

    # apply reductions to yields for the impacted parcels
    remaining = impacted_area_ha
    for p_idx in impacted_idxs:

        # compute the fraction of the parcel that is impacted by the wetland based on the remaining impacted area and the parcel area
        A = float(self.parcel_area_ha[p_idx])
        if remaining <= 0:
            frac = 0.0
        elif remaining < A:
            frac = remaining / A
        else:
            frac = 1.0
        self.logger.debug(f" processing wetland-impacted parcel (pid = {self.parcel_ids[p_idx]}, area = {A:.2f} ha, fraction of parcel draining to wetland = {frac:.2f})")

        # apply reductions to each pollutant yield for this parcel based on the impacted fraction and effectiveness
        for pol_idx, pollutant in enumerate(self.pollutants):
            self.logger.debug(f"   processing pollutant {pollutant}")

            # get current yield for this parcel and pollutant
            y = float(yields[p_idx, pol_idx])
            self.logger.debug(f"  current yield for parcel {self.parcel_ids[p_idx]} pollutant {pollutant} is {y:.2f} units/ha")

            # calculate reduction based on yield, impacted area, and effectiveness, then update yields and bmp outputs
            reduction = y * (A * frac) * eff[pol_idx]
            treated = y * (A * frac)
            bmp_outputs[OUTPUT_TREATED][pol_idx] += treated
            bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
            self.logger.debug(f"  calculated reduction for pollutant {pollutant}: {reduction:.2f} units of {treated:.2f} treated")

            # reduce yield based on the calculated reduction and the fraction of the parcel that is impacted by the wetland, ensuring yields don't go negative
            y_new = y - reduction / A
            yields[p_idx, pol_idx] = max(0.0, y_new)
            self.logger.debug(f"   reduced parcel {self.parcel_ids[p_idx]} yield for pollutant {pollutant} is {yields[p_idx, pol_idx]:.2f} units/ha")

        # reduce the remaining impacted area by the area of parcel that drains to the wetland
        remaining -= A


def _simulate_grassed(
    self: Model,
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
 ) -> None:
    """Simulate a grassed waterway or buffer BMP and update yield reductions."""
    self.logger.debug(f" calling simulate_grassed")
    
    # determine the length of the buffer/grass strip based on a randomly selected fraction of the parcel perimeter
    perim_m = float(self.parcel_perim_m[parcel_idx])
    frac_stats = {"min": 0.1, "max": 0.3, "mean": 0.2} # TODO - make these configurable, print to output files
    perim_frac = self._sample_from_stats(stats=frac_stats, kind=None)
    length_m = perim_m * perim_frac
    self.logger.debug(f"  length of grassed buffer/waterway is {length_m:.2f} m based on fraction {perim_frac:.2f} of parcel perimeter {perim_m:.2f} m")

    # set depth and area of grassed waterway
    depth_m = float(self.cfg.get(CFG_BUFFER_DEPTH_FT, 35.0)) * FT_TO_M # TODO: make the depth configurable or sampled from a range of values?
    area_ha = (length_m * depth_m) / 10000.0

    # set portion of parcel treated
    frac_stats = {"min": 0.2, "max": 0.4, "mean": 0.3} # TODO - make these configurable, print to output files
    frac_treated = self._sample_from_stats(stats=frac_stats, kind=None)

    # update record and outputs
    bmp_rec[OUTPUT_LINEAR_LENGTH] = length_m
    bmp_rec[OUTPUT_BUFFER_AREA] = area_ha
    bmp_rec[OUTPUT_PORTION_TREATED] = frac_treated

    A = float(self.parcel_area_ha[parcel_idx])
    for pol_idx, pollutant in enumerate(self.pollutants):
        y = float(yields[parcel_idx, pol_idx])
        reduction = y * (A * frac_treated) * eff[pol_idx]
        bmp_outputs[OUTPUT_TREATED][pol_idx] += y * (A * frac_treated)
        bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
        y_new = y - reduction / A
        yields[parcel_idx, pol_idx] = max(0.0, y_new)
        self.logger.debug(f"   reduced yield for pollutant {pollutant}: existing yield ={y:.2f}, effectiveness={eff[pol_idx]:.2f}, treated_area={A * frac_treated:.2f} ha, removal={reduction:.2f}, new yield={yields[parcel_idx, pol_idx]:.2f}")


def _simulate_infield(
    self: Model,
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
) -> None:
    """Simulate an in-field BMP and update the parcel yield state."""
    self.logger.debug(f" calling _simulate_infield")
    
    A = float(self.parcel_area_ha[parcel_idx])
    self.logger.debug(f"  cps={bmp_rec['cps']}, parcel_idx={parcel_idx}, area={A:.2f} ha")
    for pol_idx, pollutant in enumerate(self.pollutants):
        y = float(yields[parcel_idx, pol_idx])
        reduction = y * A * eff[pol_idx]
        bmp_outputs[OUTPUT_TREATED][pol_idx] += y * A
        bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
        y_new = y - reduction / A
        yields[parcel_idx, pol_idx] = max(0.0, y_new)
        self.logger.debug(f"   reduced yield for pollutant {pollutant}: existing yield ={y:.2f}, effectiveness={eff[pol_idx]:.2f}, treated_area={A:.2f} ha, removal={reduction:.2f}, new yield={yields[parcel_idx, pol_idx]:.2f}")


def _get_bmp_selection_probs(
    self: Model,
    bmp_sel_path: Optional[str],
    ) -> pd.DataFrame:
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
        self.logger.debug(f"Loaded explicit BMP selection probabilities from {bmp_sel_path}: {df.to_dict(orient='records')}")
        return df[[COL_CPS, COL_PROBABILITY]]
    else:
        if self.data[DATA_BMP_COST] is None:
            probs = np.full(len(self.data[DATA_CPS]), 1.0 / len(self.data[DATA_CPS]))
            return pd.DataFrame({COL_CPS: self.data[DATA_CPS], COL_PROBABILITY: probs})
        else:
            df = self._estimate_costs_for_probabilities()
            return df
