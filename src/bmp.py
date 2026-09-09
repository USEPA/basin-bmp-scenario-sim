"""BMP selection and load-reduction helpers.

This module contains the logic used to choose best management practices
(BMPs), sample their effectiveness, and apply their impacts to parcel-level
pollutant loads. The functions are implemented as model helpers and operate
on shared model state such as parcel geometry, pollutant load_rates, and
simulation outputs.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

from .constants import (
    CFG_BUFFER_DEPTH_FT,
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
    PATHWAY_VALUES,
)
from .logging_utils import log_scope
from .input_config import load_bmp_selection_probabilities

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

FT_TO_M = 0.3048  # meters per foot


def _active_pathways(self: "Model") -> List[str]:
    """Return the model's active pathways, using the configured pathways or the standard three-path default.

        Returns
        -------
        List[str]
            Active pathway names in model order.
        
    """
    configured = getattr(self, "pathway_names", None)
    return list(configured) if configured else list(PATHWAY_VALUES)


def _get_pathway_load_rates(self: "Model", parcel_idx: int, pol_idx: int, total_load_rate: float) -> Dict[str, float]:
    """Return current parcel areal load rate contributions by active pathway.

        Parameters
        ----------
        parcel_idx : int
            Zero-based index of the parcel.
        pol_idx : int
            Zero-based index of the pollutant.
        total_load_rate : float
            Current total pollutant load rate for the parcel.

        Returns
        -------
        Dict[str, float]
            Current load rate for each active pathway.

        Raises
        ------
        ValueError
            If multiple pathways are active but neither pathway-resolved state nor pathway fractions are available.
        
    """
    pathways = _active_pathways(self)
    pathway_load_rates = getattr(self, "current_pathway_load_rates", None)
    if pathway_load_rates is not None:
        values = pathway_load_rates[parcel_idx, pol_idx, :]
        return {pathways[i]: float(values[i]) for i in range(len(pathways))}

    fractions = dict(
        getattr(self, "pollutant_load_rate_pathway_fractions", None)
        or getattr(self, "pollutant_load_rate_pathway_fractions", {})
        or {}
    )
    if not fractions:
        if len(pathways) != 1:
            raise ValueError("Multiple active pathways require tracked pathway load_rates or pathway fractions")
        fractions = {pathways[0]: 1.0}
    return {path: float(total_load_rate) * float(fractions[path]) for path in pathways}

def _get_current_total_load_rate(
    self: "Model", parcel_idx: int, pol_idx: int, fallback_load_rate: float
) -> float:
    """Return the current total across tracked pathways.

        Protected groundwater load rate is included when present in the active scenario state.

        Parameters
        ----------
        parcel_idx : int
            Zero-based index of the parcel.
        pol_idx : int
            Zero-based index of the pollutant.
        fallback_load_rate : float
            Load rate to use when pathway-resolved state is unavailable.

        Returns
        -------
        float
            Current total pollutant load rate across tracked pathways.
        
    """
    pathway_load_rates = getattr(self, "current_pathway_load_rates", None)
    if pathway_load_rates is None:
        return float(fallback_load_rate)
    protected = getattr(self, "current_untreated_groundwater_load_rates", None)
    protected_value = 0.0 if protected is None else float(protected[parcel_idx, pol_idx])
    return float(np.sum(pathway_load_rates[parcel_idx, pol_idx, :]) + protected_value)


def _apply_pathway_reduction(
    self: "Model",
    parcel_idx: int,
    pol_idx: int,
    treatment_fraction: float,
    eff_map: Dict[str, float],
) -> float:
    """Apply pathway-specific BMP reduction to in-memory pathway load_rates.

    The reduction is applied only when the model is tracking pathway-specific
    loads. Each pathway is reduced by ``treatment_fraction * eff_map[path]``.
    Hydrologic partitioning and BMP treatability are intentionally independent.
    Each active pathway is affected only by its own pathway-specific efficiency.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel being treated.
    pol_idx : int
        Index of the pollutant being treated.
    treatment_fraction : float
        Fraction of the parcel or drainage area treated by the BMP.
    eff_map : dict[str, float]
        Pathway-specific effectiveness values keyed by pathway name.

    Returns
    -------
    float
        Total load removed across all pathways.
    """
    pathway_load_rates = getattr(self, "current_pathway_load_rates", None)
    if pathway_load_rates is None:
        return 0.0

    removed_load_rate = 0.0
    for path_idx, path in enumerate(_active_pathways(self)):
        current_load_rate = float(pathway_load_rates[parcel_idx, pol_idx, path_idx])
        eff = float(eff_map[path])
        if current_load_rate < 0.0:
            raise RuntimeError("Current pathway load rate cannot be negative")
        if not 0.0 <= float(treatment_fraction) <= 1.0:
            raise RuntimeError("Treatment fraction must remain in [0, 1]")
        if eff > 1.0:
            raise RuntimeError("BMP efficiency must not exceed 1 after input validation")
        new_value = current_load_rate * (1.0 - treatment_fraction * eff)
        removed_load_rate += current_load_rate - new_value
        pathway_load_rates[parcel_idx, pol_idx, path_idx] = new_value
    return float(removed_load_rate)



def _select_bmp_type(self: "Model") -> int:
    """Randomly choose the next BMP CPS code.

    The selection is made from the probability table produced by
    :func:`_get_bmp_selection_probs` and stored on the model as
    ``self.bmp_selection_probs``.

    Returns
    -------
    int
        Selected BMP CPS code.
    """
    idx = self.rng.choice(len(self.bmp_cps), p=self.bmp_selection_probs)
    cps = int(self.bmp_cps[idx])
    self.logger.verbose(f"selected bmp {cps} ({self._get_bmp_name(cps)})")
    return cps


def _get_bmp_name(self: "Model", cps: Union[int, str]) -> str:
    """Return a human-readable BMP name for a CPS code.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    cps : int or str
        BMP CPS identifier.

    Returns
    -------
    str
        Known BMP name when available, otherwise ``"CPS {code}"``.
    """
    key = int(cps)
    return BMP_CPS_NAME_MAP[key] if key in BMP_CPS_NAME_MAP else f"CPS {key}"


def _sample_efficiency_map(self: "Model", cps: Union[int, str], pol_idx: int) -> Dict[str, float]:
    """Sample one BMP efficiency for every active pathway.

        Input validation guarantees complete coverage. In PLET/RUSLE mode this
        means surface is explicitly required and subsurface is either explicitly
        supplied or has already been inserted as a fixed zero distribution.
        Statistical mode requires explicit coverage for every active user-defined
        pathway.

        Parameters
        ----------
        cps : Union[int, str]
            Conservation Practice Standard (CPS) code or codes.
        pol_idx : int
            Zero-based index of the pollutant.

        Returns
        -------
        Dict[str, float]
            Sampled BMP efficiencies keyed by active pathway.

        Raises
        ------
        ValueError
            If BMP efficiency coverage is incomplete for the requested CPS, pollutant, or active pathway.
        
    """
    entry = self.bmp_efficiency_stats[int(cps)][pol_idx]
    pathways = _active_pathways(self)
    missing_paths = (
        list(pathways)
        if not isinstance(entry, dict)
        else [path for path in pathways if path not in entry or not isinstance(entry[path], dict)]
    )
    if missing_paths:
        raise ValueError(
            "Incomplete bmp_efficiency coverage for "
            f"cps={int(cps)}, pollutant={self.pollutants[pol_idx]}; "
            f"missing pathways: {missing_paths}"
        )
    return {
        path: float(self._sample_from_stats(entry[path], kind="efficiency"))
        for path in pathways
    }

def _simulate_wetland(
    self: "Model",
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    load_rates: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_mass_rate_outputs: Dict[str, np.ndarray],
    cps: Union[int, str] = 656,
) -> None:
    """Apply a wetland BMP and update parcel loads.

    The wetland treatment area is sampled from heuristic area statistics and
    clipped to the parcel area. The function then allocates the wetland to a
    contributing set of upstream parcels, computes the impacted drainage area,
    and applies pathway-specific reductions to each affected parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel receiving the wetland BMP.
    eff_maps : sequence of dict[str, float]
        Effectiveness maps, one per pollutant, keyed by pathway name.
    load_rates : numpy.ndarray
        Parcel-by-pollutant areal load-rate array updated in place.
    bmp_rec : dict[str, Any]
        Record for the current BMP placement, updated with wetland metadata.
    bmp_mass_rate_outputs : dict[str, numpy.ndarray]
        Accumulators for treated and removed pollutant loads.
    cps : int or str, optional
        BMP CPS code associated with the wetland. Default is ``656``.

    Returns
    -------
    None
    """
    with log_scope(label="simulate_wetland", logger=self.logger):
        self.logger.verbose("calling simulate_wetland")

        # wetland area (ha), clipped by field area
        area_field_ha = float(self.parcel_area_ha[parcel_idx])
        wet_area_stats = {"min": 0.1, "p25": 0.4, "p50": 0.81, "p75": 2.0, "max": 4.0}  # heuristic
        wet_area = self._sample_from_stats(stats=wet_area_stats, kind=None)
        wet_area = min(wet_area, area_field_ha)
        # Restored from legacy: detailed diagnostics
        self.logger.verbose(
            f"selected wetland area of {wet_area:.2f} ha in parcel idx={parcel_idx} of area={area_field_ha:.2f} ha"
        )

        # catchment area ratio (dimensionless)
        ratio_stats = {"min": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "max": 100.0}  # heuristic
        cat_ratio = float(self._sample_from_stats(stats=ratio_stats, kind=None))

        # Impacted area to satisfy ratio
        impacted_idxs: List[int] = [parcel_idx]
        impacted_area_ha: float = wet_area * (1.0 + cat_ratio)
        total_available_ha = float(self.parcel_area_ha[parcel_idx])

        for up_idx in self.parcel_up_idxs[parcel_idx]:
            if up_idx not in impacted_idxs:
                impacted_idxs.append(up_idx)
                total_available_ha += float(self.parcel_area_ha[up_idx])
                self.logger.verbose(
                    f"added upgradient parcel (pid={self.parcel_ids[up_idx]}) with area "
                    f"{self.parcel_area_ha[up_idx]:.2f} ha to wetland-impacted parcels"
                )
                if total_available_ha >= impacted_area_ha:
                    break

        # Adjust ratio when upstream area is insufficient
        if impacted_area_ha > total_available_ha:
            self.logger.verbose(
                f"total available upgradient area ({total_available_ha:.2f} ha) < impacted area "
                f"(wetland+catchment) ({impacted_area_ha:.2f} ha)"
            )
            impacted_area_ha = total_available_ha
            # This is a derived-state constraint, not input sanitization. The
            # denominator guard protects the ratio calculation if a degenerate
            # zero-area parcel reaches this internal heuristic.
            cat_ratio = (impacted_area_ha - wet_area) / max(wet_area, 1e-9)
            self.logger.verbose(
                f"reduced impacted area to {impacted_area_ha:.2f} ha and catchment ratio to {cat_ratio:.2f}"
            )

        bmp_rec[OUTPUT_WETLAND_AREA] = float(wet_area)
        bmp_rec[OUTPUT_CATCHMENT_RATIO] = float(cat_ratio)
        bmp_rec[OUTPUT_IMPACTED_PIDS] = ",".join([self.parcel_ids[idx] for idx in impacted_idxs] if len(impacted_idxs) > 1 else [])

        # Apply reductions across impacted parcels
        remaining = impacted_area_ha
        for p_idx in impacted_idxs:
            parcel_area_ha = float(self.parcel_area_ha[p_idx])
            if remaining <= 0:
                treated_area_fraction = 0.0
            elif remaining < parcel_area_ha:
                treated_area_fraction = remaining / parcel_area_ha
            else:
                treated_area_fraction = 1.0
            self.logger.verbose(
                f"processing wetland-impacted parcel pid={self.parcel_ids[p_idx]}, "
                f"area={parcel_area_ha:.2f} ha, fraction draining={treated_area_fraction:.2f}"
            )

            for pol_idx, pollutant in enumerate(self.pollutants):
                load_rate = float(load_rates[p_idx, pol_idx])
                pathway_load_rates_by_name = self._get_pathway_load_rates(p_idx, pol_idx, load_rate)
                efficiency_by_pathway = eff_maps[pol_idx]

                treated_baseline_mass_rate_kg_per_yr = sum(pathway_load_rates_by_name.values()) * (parcel_area_ha * treated_area_fraction)
                removed_mass_rate_kg_per_yr = (parcel_area_ha * treated_area_fraction) * sum(
                    pathway_load_rates_by_name[path] * efficiency_by_pathway[path]
                    for path in _active_pathways(self)
                )
                self._apply_pathway_reduction(p_idx, pol_idx, treated_area_fraction, efficiency_by_pathway)

                bmp_mass_rate_outputs[OUTPUT_TREATED][pol_idx] += treated_baseline_mass_rate_kg_per_yr
                bmp_mass_rate_outputs[OUTPUT_REMOVED][pol_idx] += removed_mass_rate_kg_per_yr
                updated_load_rate = (
                    self._get_current_total_load_rate(p_idx, pol_idx, load_rate)
                    if getattr(self, "current_pathway_load_rates", None) is not None
                    else (load_rate - removed_mass_rate_kg_per_yr / parcel_area_ha)
                )
                load_rates[p_idx, pol_idx] = updated_load_rate

            remaining -= parcel_area_ha


def _simulate_grassed(
    self: "Model",
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    load_rates: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_mass_rate_outputs: Dict[str, np.ndarray],
) -> None:
    """Apply a grassed buffer BMP to the selected parcel.

    The buffer length is sampled as a fraction of parcel perimeter, converted
    to area using the configured buffer depth, and then used to compute the
    treated and removed pollutant loads for the parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel receiving the BMP.
    eff_maps : sequence of dict[str, float]
        Effectiveness maps, one per pollutant, keyed by pathway name.
    load_rates : numpy.ndarray
        Parcel-by-pollutant areal load-rate array updated in place.
    bmp_rec : dict[str, Any]
        Record for the current BMP placement, updated with buffer metadata.
    bmp_mass_rate_outputs : dict[str, numpy.ndarray]
        Accumulators for treated and removed pollutant loads.

    Returns
    -------
    None
    """
    with log_scope(label="simulate_grassed", logger=self.logger):
        self.logger.verbose("calling simulate_grassed")

        # Determine linear length as a fraction of parcel perimeter
        perim_m = float(self.parcel_perim_m[parcel_idx])
        frac_stats = {"min": 0.1, "max": 0.3, "mean": 0.2}  # heuristic
        perim_frac = self._sample_from_stats(stats=frac_stats, kind=None)
        length_m = perim_m * perim_frac
        self.logger.verbose(
            f"grassed buffer length={length_m:.2f} m from fraction={perim_frac:.2f} of perimeter={perim_m:.2f} m"
        )

        # Depth and area (length * depth -> m^2 -> ha)
        depth_ft = float(self.cfg[CFG_BUFFER_DEPTH_FT])
        depth_m = depth_ft * FT_TO_M
        area_ha = (length_m * depth_m) / 10000.0
        self.logger.verbose(
            f"grassed buffer depth={depth_ft:.2f} ft ({depth_m:.2f} m), area={area_ha:.4f} ha"
        )

        # Portion treated
        frac_stats = {"min": 0.2, "max": 0.4, "mean": 0.3}  # heuristic
        treated_area_fraction = self._sample_from_stats(stats=frac_stats, kind=None)

        # Update record and outputs
        bmp_rec[OUTPUT_LINEAR_LENGTH] = float(length_m)
        bmp_rec[OUTPUT_BUFFER_AREA] = float(area_ha)
        bmp_rec[OUTPUT_PORTION_TREATED] = float(treated_area_fraction)

        parcel_area_ha = float(self.parcel_area_ha[parcel_idx])
        for pol_idx, pollutant in enumerate(self.pollutants):
            load_rate = float(load_rates[parcel_idx, pol_idx])
            pathway_load_rates_by_name = self._get_pathway_load_rates(parcel_idx, pol_idx, load_rate)
            efficiency_by_pathway = eff_maps[pol_idx]

            treated_baseline_mass_rate_kg_per_yr = sum(pathway_load_rates_by_name.values()) * (parcel_area_ha * treated_area_fraction)
            removed_mass_rate_kg_per_yr = (parcel_area_ha * treated_area_fraction) * sum(
                pathway_load_rates_by_name[path] * efficiency_by_pathway[path]
                for path in _active_pathways(self)
            )
            self._apply_pathway_reduction(parcel_idx, pol_idx, treated_area_fraction, efficiency_by_pathway)

            bmp_mass_rate_outputs[OUTPUT_TREATED][pol_idx] += treated_baseline_mass_rate_kg_per_yr
            bmp_mass_rate_outputs[OUTPUT_REMOVED][pol_idx] += removed_mass_rate_kg_per_yr
            updated_load_rate = self._get_current_total_load_rate(parcel_idx, pol_idx, load_rate) if getattr(self, "current_pathway_load_rates", None) is not None else (load_rate - removed_mass_rate_kg_per_yr / parcel_area_ha)
            load_rates[parcel_idx, pol_idx] = updated_load_rate


def _simulate_infield(
    self: "Model",
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    load_rates: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_mass_rate_outputs: Dict[str, np.ndarray],
) -> None:
    """Apply an in-field BMP to the selected parcel.

    In-field BMPs are treated as covering the whole parcel. The function
    therefore applies each pollutant's pathway-specific effectiveness across
    the full parcel area and updates both the output accumulators and the
    areal load-rate array in place.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel receiving the BMP.
    eff_maps : sequence of dict[str, float]
        Effectiveness maps, one per pollutant, keyed by pathway name.
    load_rates : numpy.ndarray
        Parcel-by-pollutant areal load-rate array updated in place.
    bmp_rec : dict[str, Any]
        Record for the current BMP placement.
    bmp_mass_rate_outputs : dict[str, numpy.ndarray]
        Accumulators for treated and removed pollutant loads.

    Returns
    -------
    None
    """
    with log_scope(label="simulate_infield", logger=self.logger):
        self.logger.verbose("calling _simulate_infield")

        parcel_area_ha = float(self.parcel_area_ha[parcel_idx])
        for pol_idx, pollutant in enumerate(self.pollutants):
            load_rate = float(load_rates[parcel_idx, pol_idx])
            pathway_load_rates_by_name = self._get_pathway_load_rates(parcel_idx, pol_idx, load_rate)
            efficiency_by_pathway = eff_maps[pol_idx]

            treated_baseline_mass_rate_kg_per_yr = sum(pathway_load_rates_by_name.values()) * parcel_area_ha
            removed_mass_rate_kg_per_yr = parcel_area_ha * sum(
                pathway_load_rates_by_name[path] * efficiency_by_pathway[path]
                for path in _active_pathways(self)
            )
            self._apply_pathway_reduction(parcel_idx, pol_idx, 1.0, efficiency_by_pathway)

            bmp_mass_rate_outputs[OUTPUT_TREATED][pol_idx] += treated_baseline_mass_rate_kg_per_yr
            bmp_mass_rate_outputs[OUTPUT_REMOVED][pol_idx] += removed_mass_rate_kg_per_yr
            updated_load_rate = self._get_current_total_load_rate(parcel_idx, pol_idx, load_rate) if getattr(self, "current_pathway_load_rates", None) is not None else (load_rate - removed_mass_rate_kg_per_yr / parcel_area_ha)
            load_rates[parcel_idx, pol_idx] = updated_load_rate


def _get_bmp_selection_probs(self: "Model", bmp_sel_path: Optional[str]) -> pd.DataFrame:
    """Build the BMP selection probability table.

    The function first attempts to read an explicit probability CSV. If no
    file is provided, it may estimate probabilities from cost heuristics when
    enabled in the configuration. Otherwise, all BMP types receive equal
    probability.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    bmp_sel_path : str or None
        Optional path to a CSV file containing BMP selection probabilities.

    Returns
    -------
    pandas.DataFrame
        Two-column dataframe with ``COL_CPS`` and ``COL_PROBABILITY``.

    Raises
    ------
    ValueError
        If the probability file is malformed, incomplete, or contains invalid
        probability values.
    """
    if bmp_sel_path:
        return load_bmp_selection_probabilities(
            bmp_sel_path,
            self.data[DATA_CPS],
            self.logger,
        )
    else:
        est_via_costs = self.cfg["bmp_sel_prob_via_costs"]
        if est_via_costs and self.data[DATA_BMP_COST] is not None:
            self.logger.info("estimating BMP selection probabilities via cost heuristics")
            df = self._estimate_costs_for_probabilities()
            return df[[COL_CPS, COL_PROBABILITY]]
        else:
            probs = np.full(len(self.data[DATA_CPS]), 1.0 / len(self.data[DATA_CPS]))
            return pd.DataFrame({COL_CPS: self.data[DATA_CPS], COL_PROBABILITY: probs})