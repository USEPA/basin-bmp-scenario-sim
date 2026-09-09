"""Run the simulation workflow and write scenario output files.

This module coordinates scenario execution, applies BMPs, collects summary
outputs, and writes the per-scenario and cross-scenario result tables used by
the rest of the application.
"""

from __future__ import annotations

import logging
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import SeedSequence, default_rng

from src.bmp import (
    _apply_pathway_reduction,
    _get_bmp_name,
    _get_current_total_load_rate,
    _get_pathway_load_rates,
    _get_bmp_selection_probs,
    _sample_efficiency_map,      # per-pathway sampler
    _select_bmp_type,
    _simulate_grassed,
    _simulate_infield,
    _simulate_wetland,
)
from src.cost import _estimate_costs_for_probabilities, _get_bmp_cost, _select_cost_rate_median
from src.io_utils import _write_parquet_atomic, _flatten_plot_records
from src.logging_utils import make_worker_logger, log_scope
from src.plet_rusle import (
    calculate_load_diagnostics,
    initialize_plet_rusle_state,
)
from src.parcel import (
    _get_delivery_coeffs,
    _get_parcel_metadata,
    _get_parcel_out_oids,
    _get_parcel_up_list,
    _sample_parcel_index,
    _sample_load_rate,
)
from src.sampling import _piecewise_quantile_sample, _sample_from_stats, _trunc_normal
from src.summaries import BMPSummaryCollector
from src.constants import (
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_OUTPUTS,
    CFG_PARALLEL,
    CFG_VERBOSE,  # pass verbose to worker loggers
    # Failure config keys
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    DEFAULT_BMP_FAIL_REDUCTION,
    # Outputs
    OUTPUT_PORTION_TREATED,
    OUTPUT_BMP_FAILED,
    COL_POLLUTANT,
    COL_PATHWAY,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_BMP_COST,
    DATA_BMP_EFFICIENCY,
    DATA_BMP_LIMIT_N,
    DATA_BMP_LIMIT_USD,
    DATA_CPS,
    DATA_DELIVERY_RATIOS,
    DATA_N_SCENARIOS,
    DATA_RANDOM_SEED,
    DATA_OUTLET_LOC,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_PARCEL_OUT_MAP,
    DATA_PARCEL_P,
    DATA_PARCEL_UP_MAP,
    DATA_PARCELS,
    DATA_POLLUTANT_LOAD_RATE,
    DATA_LOAD_GENERATION,
    DATA_PLET_INPUTS,
    DATA_RUSLE_INPUTS,
    DATA_POLLUTANT_CONCENTRATIONS,
    DATA_GROUNDWATER_CONCENTRATIONS,
    DATA_PATHWAYS,
    DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE,
    LOAD_GROUNDWATER_LOADS,
    LOAD_MODE_PLET_RUSLE,
    LOAD_MODE_STATISTICAL,
    DATA_POLLUTANTS,
    DIR_OUTLET_TRAJECTORIES,
    DIR_SCENARIO_METRICS,
    FILE_ALL_SCENARIOS_PARQUET,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_COST_USD,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
    OUTPUT_WETLAND_AREA,
    XAXIS_COST,
    XAXIS_COUNT,
    YAXIS_MEAN,
    YAXIS_TARGET,
    YAXIS_TOTAL,
)


# Current production runs use an annual timestep. Multiplying annual load rates
# (kg/yr) by one year converts them to mass (kg) without changing the numeric
# value. Future dynamic implementations should replace this constant with the
# actual timestep duration in years before aggregating mass.
_CURRENT_TIMESTEP_YEARS = 1.0


def _safe_mass_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return a dimensionless mass ratio, or ``None`` when undefined.

        Parameters
        ----------
        numerator : float
            Numerator of the dimensionless ratio.
        denominator : float
            Denominator of the dimensionless ratio.

        Returns
        -------
        Optional[float]
            Dimensionless ratio, or ``None`` when the denominator is not positive or finite.
        
    """
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return numerator / denominator


def _bmp_impacted_parcel_indices(ctx: Any, parcel_idx: int, bmp_rec: Dict[str, Any]) -> List[int]:
    """Return full-hydrologic-universe parcel indices represented by one BMP.

        For wetlands, ``impacted_pids`` may include upstream parcels in addition to
        the placement parcel. Other BMP types operate on the placement parcel only.
        The placement parcel is always included defensively even when the wetland
        output contains an empty or partial impacted-PID string.

        Parameters
        ----------
        ctx : Any
            Active scenario or model context.
        parcel_idx : int
            Zero-based index of the parcel.
        bmp_rec : Dict[str, Any]
            Mutable record for the current BMP placement.

        Returns
        -------
        List[int]
            Indices of parcels represented by the BMP placement.
        
    """
    pids: List[str] = [str(ctx.parcel_ids[parcel_idx])]
    raw = str(bmp_rec.get(OUTPUT_IMPACTED_PIDS) or "")
    for token in raw.split(","):
        pid = token.strip()
        if pid and pid not in pids:
            pids.append(pid)
    pid_to_index = getattr(ctx, "pid_to_index", None) or {
        str(pid): idx for idx, pid in enumerate(ctx.parcel_ids)
    }
    return [int(pid_to_index[pid]) for pid in pids if pid in pid_to_index]


def _baseline_mass_before_bmp_kg(
    ctx: Any,
    pre_bmp_load_rates: np.ndarray,
    parcel_idx: int,
    bmp_rec: Dict[str, Any],
) -> np.ndarray:
    """Calculate pre-BMP pollutant mass for the parcels represented by a BMP.

        ``pre_bmp_load_rates`` contains the current load rate immediately before this
        BMP is applied. For the present annual model, rate × area × 1 year gives kg.
        Wetland denominators include all hydrologically impacted parcels identified
        by the wetland routine, including parcels that are not BMP-selectable.

        Parameters
        ----------
        ctx : Any
            Active scenario or model context.
        pre_bmp_load_rates : np.ndarray
            Parcel pollutant load rates immediately before BMP application.
        parcel_idx : int
            Zero-based index of the parcel.
        bmp_rec : Dict[str, Any]
            Mutable record for the current BMP placement.

        Returns
        -------
        np.ndarray
            Pre-BMP pollutant masses for the represented parcels, in kilograms.
        
    """
    n_pol = int(pre_bmp_load_rates.shape[1])
    mass = np.zeros(n_pol, dtype=float)
    for idx in _bmp_impacted_parcel_indices(ctx, parcel_idx, bmp_rec):
        area_ha = float(ctx.parcel_area_ha[idx])
        mass += np.asarray(pre_bmp_load_rates[idx, :], dtype=float) * area_ha * _CURRENT_TIMESTEP_YEARS
    return mass


def _add_mass_metrics_to_bmp_record(
    bmp_rec: Dict[str, Any],
    pollutants: List[str],
    baseline_mass_kg: np.ndarray,
    treated_mass_kg: np.ndarray,
    removed_mass_kg: np.ndarray,
) -> None:
    """Write explicit mass accounting and dimensionless BMP metrics to a record.

        Parameters
        ----------
        bmp_rec : Dict[str, Any]
            Mutable record for the current BMP placement.
        pollutants : List[str]
            Pollutant names in model order.
        baseline_mass_kg : np.ndarray
            Baseline pollutant masses, in kilograms.
        treated_mass_kg : np.ndarray
            Pollutant masses subjected to BMP treatment, in kilograms.
        removed_mass_kg : np.ndarray
            Pollutant masses removed by the BMP, in kilograms.
        
    """
    bmp_rec["mass_timestep_years"] = float(_CURRENT_TIMESTEP_YEARS)
    for pol_idx, pol in enumerate(pollutants):
        baseline_mass = float(baseline_mass_kg[pol_idx])
        treated_mass = float(treated_mass_kg[pol_idx])
        removed_mass = float(removed_mass_kg[pol_idx])

        bmp_rec[f"baseline_mass_{pol}_kg"] = baseline_mass
        bmp_rec[f"treated_baseline_mass_{pol}_kg"] = treated_mass
        bmp_rec[f"removed_mass_{pol}_kg"] = removed_mass

        exposure = _safe_mass_ratio(treated_mass, baseline_mass)
        realized = _safe_mass_ratio(removed_mass, treated_mass)
        overall = _safe_mass_ratio(removed_mass, baseline_mass)
        bmp_rec[f"treatment_exposure_fraction_{pol}"] = exposure
        bmp_rec[f"realized_efficiency_{pol}"] = realized
        bmp_rec[f"overall_reduction_fraction_{pol}"] = overall




class Model:
    """Main controller for the simulation.

    The model prepares data, binds helper methods, runs one or more scenarios,
    and writes the output files consumed by summary and plotting workflows.
    """

    def __init__(self, cfg: Dict[str, Any], data: Dict[str, Any], logger: logging.Logger) -> None:
        """Initialize the model and bind helper methods.

        Parameters
        ----------
        cfg : dict[str, Any]
            Scenario configuration mapping.
        data : dict[str, Any]
            Validated input data bundle.
        logger : logging.Logger
            Logger used for model-level messages.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If configured pathway fractions are invalid.
        """
        self.cfg = cfg
        self.data = data
        self.logger = logger
        seed = data[DATA_RANDOM_SEED]
        self.rng = np.random.default_rng(seed)
        self.outputs_dir: Optional[Path] = None

        # Bind helper functions
        self._sample_from_stats = types.MethodType(_sample_from_stats, self)
        self._piecewise_quantile_sample = types.MethodType(_piecewise_quantile_sample, self)
        self._trunc_normal = types.MethodType(_trunc_normal, self)

        self._select_bmp_type = types.MethodType(_select_bmp_type, self)
        self._get_bmp_name = types.MethodType(_get_bmp_name, self)
        self._get_pathway_load_rates = types.MethodType(_get_pathway_load_rates, self)
        self._get_current_total_load_rate = types.MethodType(_get_current_total_load_rate, self)
        self._apply_pathway_reduction = types.MethodType(_apply_pathway_reduction, self)
        self._sample_efficiency_map = types.MethodType(_sample_efficiency_map, self)  # pathway-aware
        self._simulate_wetland = types.MethodType(_simulate_wetland, self)
        self._simulate_grassed = types.MethodType(_simulate_grassed, self)
        self._simulate_infield = types.MethodType(_simulate_infield, self)
        self._get_bmp_selection_probs = types.MethodType(_get_bmp_selection_probs, self)
        self._get_bmp_cost = types.MethodType(_get_bmp_cost, self)

        self._sample_parcel_index = types.MethodType(_sample_parcel_index, self)
        self._sample_load_rate = types.MethodType(_sample_load_rate, self)
        self._get_parcel_metadata = types.MethodType(_get_parcel_metadata, self)
        self._get_parcel_up_list = types.MethodType(_get_parcel_up_list, self)
        self._get_parcel_out_oids = types.MethodType(_get_parcel_out_oids, self)
        self._delivery_coeffs = types.MethodType(_get_delivery_coeffs, self)

        self._estimate_costs_for_probabilities = types.MethodType(_estimate_costs_for_probabilities, self)
        self._select_cost_rate_median = types.MethodType(_select_cost_rate_median, self)

        # Pathway definitions and aggregate-load-rate fractions are validated in
        # input_config because statistical mode may use arbitrary pathway names.

        self._prepare_lookup_tables()

    def _prepare_lookup_tables(self) -> None:
        """Build lookup tables used during scenario execution.

        The method derives parcel indices, outlet mappings, selection
        probabilities, BMP effectiveness structures, and load-generation
        configuration values from the validated input data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If parcel IDs are duplicated or required input tables are missing.
        """
        with log_scope(label="prepare_lookup_tables", logger=self.logger):
            parcels = self.data[DATA_PARCELS]
            self.parcel_ids = parcels["pid"].astype(str).tolist()
            if len(set(self.parcel_ids)) != len(self.parcel_ids):
                pid_series = pd.Series(self.parcel_ids)
                dup_pids = sorted(pid_series[pid_series.duplicated()].unique().tolist())
                raise ValueError(f"Duplicate parcel IDs found in loaded parcels: {dup_pids}")
            self.pid_to_index = {pid: idx for idx, pid in enumerate(self.parcel_ids)}
            self.pollutants = list(self.data[DATA_POLLUTANTS])
            self.pollutant_to_index = {p: i for i, p in enumerate(self.pollutants)}
            self.parcel_area_ha = parcels["area_ha"].astype(float).tolist()
            self.parcel_perim_m = parcels["perim_m"].astype(float).tolist()

            # Parcel outlet and up-gradient mappings
            po_map = self.data[DATA_PARCEL_OUT_MAP]
            self.parcel_out_oids = [[str(x) for x in po_map[pid]] for pid in self.parcel_ids]
            pu_map = self.data[DATA_PARCEL_UP_MAP]
            self.parcel_up_idxs = [[self.pid_to_index[u] for u in pu_map[pid] if u in self.pid_to_index] for pid in self.parcel_ids]

            # Parcel selection probabilities
            sel = self.data[DATA_PARCEL_P]
            self.parcel_selection_ids = sel["pid"].astype(str).tolist()
            self.parcel_selection_probs = sel["probability"].astype(float).values
            self.selection_source_idxs = [self.pid_to_index[pid] for pid in self.parcel_selection_ids]
            # ``parcel_p`` defines only the BMP-placement universe.  Hydrologic
            # state remains indexed to every modeled parcel so upstream parcels
            # continue to contribute loads to downstream structural practices
            # even when those upstream parcels are not themselves BMP-eligible.

            # Outlet IDs and optional targets/means
            self.outlet_oids = list(self.data[DATA_OUTLET_LOC]["oid"].astype(str).tolist())
            self.outlet_target_map = {}
            if self.data.get(DATA_OUTLET_TARGET) is not None:
                for _, r in self.data[DATA_OUTLET_TARGET].iterrows():
                    self.outlet_target_map[(str(r["oid"]), str(r[COL_POLLUTANT]))] = float(r["target"])
            self.outlet_mean_map = {}
            if self.data.get(DATA_OUTLET_MEAN) is not None:
                for _, r in self.data[DATA_OUTLET_MEAN].iterrows():
                    self.outlet_mean_map[(str(r["oid"]), str(r[COL_POLLUTANT]))] = float(r["mean"])

            # Delivery coeffs
            self.delivery_coeffs = {}
            if self.data.get("delivery_ratios") is not None:
                for _, r in self.data["delivery_ratios"].iterrows():
                    self.delivery_coeffs[(str(r["pid"]), str(r["oid"]))] = dict(
                        sdr_f_to_s=float(r["sdr_f_to_s"]),
                        sdr_s_to_o=float(r["sdr_s_to_o"]),
                        ndr_f_to_s=float(r["ndr_f_to_s"]),
                        ndr_s_to_o=float(r["ndr_s_to_o"]),
                    )

            # Active pathways are mode-specific and were validated during input loading.
            self.pathway_names = list(self.data.get(DATA_PATHWAYS) or ["surface"])
            self.pollutant_load_rate_pathway_fractions = dict(
                self.data.get(DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS) or {}
            )
            self.pollutant_load_rate_is_aggregate = bool(
                self.data[DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE]
            )

            # Efficiency stats by CPS x pollutant x active pathway.
            self.bmp_cps = sorted(int(c) for c in self.data[DATA_CPS])
            self.bmp_efficiency_stats = {int(c): [None] * len(self.pollutants) for c in self.bmp_cps}
            eff = self.data[DATA_BMP_EFFICIENCY]
            for _, row in eff.iterrows():
                cps_key = int(row["cps"])
                pol_key = self.pollutant_to_index[str(row[COL_POLLUTANT])]
                path = str(row[COL_PATHWAY]).strip().lower()
                stats = {
                    k: row[k]
                    for k in row.index
                    if k not in ("cps", COL_POLLUTANT, COL_PATHWAY) and not pd.isna(row[k])
                }
                entry = self.bmp_efficiency_stats[cps_key][pol_key]
                if entry is None:
                    entry = {}
                    self.bmp_efficiency_stats[cps_key][pol_key] = entry
                entry[path] = stats  # type: ignore[index]

            # Load-generation mode and optional PLET/RUSLE inputs
            self.load_generation = dict(self.data.get(DATA_LOAD_GENERATION) or {})
            self.load_generation_mode = str(
                self.load_generation["mode"]
            ).strip().lower()
            self.plet_inputs = self.data.get(DATA_PLET_INPUTS)
            self.rusle_inputs = self.data.get(DATA_RUSLE_INPUTS)
            self.pollutant_concentrations = self.data.get(DATA_POLLUTANT_CONCENTRATIONS)
            self.groundwater_concentrations = self.data.get(DATA_GROUNDWATER_CONCENTRATIONS)
            self.groundwater_loads = self.load_generation_mode == LOAD_MODE_PLET_RUSLE

            # Statistical mode supports either explicit pathway-specific load rates
            # distributions or one aggregate_load_rate distribution split by validated
            # user-defined pathway fractions.
            self.pollutant_load_rate_stats = [[None] * len(self.pollutants) for _ in range(len(self.parcel_ids))]
            pollutant_load_rate_table = self.data.get(DATA_POLLUTANT_LOAD_RATE)
            if pollutant_load_rate_table is not None:
                for _, row in pollutant_load_rate_table.iterrows():
                    i = self.pid_to_index[str(row["pid"])]
                    j = self.pollutant_to_index[str(row[COL_POLLUTANT])]
                    stats = {
                        k: row[k]
                        for k in row.index
                        if k not in ("pid", COL_POLLUTANT, COL_PATHWAY) and not pd.isna(row[k])
                    }
                    if self.pollutant_load_rate_is_aggregate:
                        self.pollutant_load_rate_stats[i][j] = stats
                    else:
                        path = str(row[COL_PATHWAY]).strip().lower()
                        entry = self.pollutant_load_rate_stats[i][j]
                        if entry is None:
                            entry = {}
                            self.pollutant_load_rate_stats[i][j] = entry
                        entry[path] = stats  # type: ignore[index]
            elif self.load_generation_mode != LOAD_MODE_PLET_RUSLE:
                raise ValueError("pollutant_load_rate is required in statistical load-generation mode")


            # BMP selection probabilities (possibly via costs or explicit table)
            bmp_probs = self._get_bmp_selection_probs(self.cfg.get(CFG_BMP_SEL))
            self.bmp_cps = bmp_probs["cps"].astype(int).tolist()
            self.bmp_selection_probs = bmp_probs["probability"].astype(float).to_numpy()

            # One-line summary (now VERBOSE)
            self.logger.verbose(
                f"Prepared lookup tables: parcels={len(self.parcel_ids)}, "
                f"pollutants={len(self.pollutants)}, bmp_types={len(self.bmp_cps)}"
            )

    def _shared_payload(self) -> Dict[str, Any]:
        """Package shared data for worker processes.

        Returns
        -------
        dict[str, Any]
            Serializable data bundle containing the model state needed by
            worker processes.
        """
        pollutant_load_rate_stats = self.pollutant_load_rate_stats
        pollutant_load_rate_pathway_fractions = dict(self.pollutant_load_rate_pathway_fractions or {})
        pollutant_load_rate_is_aggregate = bool(self.pollutant_load_rate_is_aggregate)

        return dict(
            cfg=self.cfg,
            data=self.data,
            # Full hydrologic parcel universe.  Scenario load arrays, routing,
            # and upstream relationships use these indices.
            parcel_ids=self.parcel_ids,
            pid_to_index=self.pid_to_index,
            pollutants=self.pollutants,
            parcel_area_ha=np.asarray(self.parcel_area_ha, dtype=float),
            parcel_perim_m=np.asarray(self.parcel_perim_m, dtype=float),
            parcel_out_oids=self.parcel_out_oids,
            parcel_up_idxs=self.parcel_up_idxs,
            # BMP placement remains restricted to the parcel_p universe.  The
            # selection position is translated to a full parcel index through
            # selection_source_idxs inside the scenario loop.
            parcel_selection_ids=self.parcel_selection_ids,
            parcel_selection_probs=np.asarray(self.parcel_selection_probs, dtype=float),
            selection_source_idxs=np.asarray(self.selection_source_idxs, dtype=int),
            outlet_oids=self.outlet_oids,
            outlet_target_map=self.outlet_target_map,
            outlet_mean_map=self.outlet_mean_map,
            delivery_coeffs=self.delivery_coeffs,
            bmp_efficiency_stats=self.bmp_efficiency_stats,
            pollutant_load_rate_stats=pollutant_load_rate_stats,
            load_generation=self.load_generation,
            load_generation_mode=self.load_generation_mode,
            plet_inputs=self.plet_inputs,
            rusle_inputs=self.rusle_inputs,
            pollutant_concentrations=self.pollutant_concentrations,
            groundwater_concentrations=self.groundwater_concentrations,
            pathway_names=self.pathway_names,
            pollutant_load_rate_pathway_fractions=pollutant_load_rate_pathway_fractions,
            pollutant_load_rate_is_aggregate=pollutant_load_rate_is_aggregate,
            groundwater_loads=self.groundwater_loads,
            bmp_cps=self.bmp_cps,
            bmp_selection_probs=self.bmp_selection_probs,
            avg_area_ha=self.data[DATA_AVG_AREA_HA],
            avg_perim_m=self.data[DATA_AVG_PERIM_M],
            random_seed=self.data.get("random_seed"),
        )

    def run_all_scenarios(self) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
        """Run all scenarios and return cross-scenario trajectory data.

        The method spawns workers when configured, executes each scenario,
        merges their plot records, and writes the canonical outlet trajectory
        parquet file.

        Returns
        -------
        dict[tuple[str, str, str, str], list[tuple[int, float, float]]]
            Plot records keyed by pollutant, outlet, x-axis, and y-axis.
        """
        outputs_dir = Path(self.cfg[CFG_OUTPUTS])
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir

        n_scenarios = int(self.data[DATA_N_SCENARIOS])
        parallel = dict(self.cfg.get(CFG_PARALLEL) or {})
        n_jobs = int(parallel["n_jobs"])

        shared = self._shared_payload()
        base_seed = self.data.get("random_seed")
        ss = SeedSequence(base_seed if base_seed is not None else None)
        child_seeds = ss.spawn(n_scenarios)

        self.logger.info(f"Running {n_scenarios} scenario(s) with n_jobs={n_jobs}")
        func = delayed(_run_one_scenario)
        results = Parallel(n_jobs=n_jobs)(
            func(shared, self.cfg, sidx, int(child_seeds[sidx].generate_state(1)[0]), outputs_dir)
            for sidx in range(n_scenarios)
        )

        # Merge plotting records
        merged: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
        for recs in results:
            for k, v in recs.items():
                merged[k].extend(v)

        traj_dir = outputs_dir / DIR_OUTLET_TRAJECTORIES
        traj_parquet = traj_dir / FILE_ALL_SCENARIOS_PARQUET
        traj_df = _flatten_plot_records(merged)
        _write_parquet_atomic(traj_df, traj_parquet, logger=self.logger)
        self.logger.info(f"Wrote canonical outlet trajectories parquet: {traj_parquet}")
        return merged


class _ScenarioContext:
    """Lightweight container for one worker's scenario state.

    The context mirrors the methods and attributes expected by the helper
    functions so each worker can run a scenario without needing the full
    ``Model`` instance.
    """

    def __init__(self, cfg: Dict[str, Any], shared: Dict[str, Any], logger, seed: int) -> None:
        """Create a scenario context with its own random number generator.

        Parameters
        ----------
        cfg : dict[str, Any]
            Scenario configuration mapping.
        shared : dict[str, Any]
            Shared model payload produced by :meth:`Model._shared_payload`.
        logger : Any
            Worker logger used for scenario output.
        seed : int
            Seed for the worker-specific random number generator.

        Returns
        -------
        None
        """
        self.cfg = cfg
        self.logger = logger
        self.rng = default_rng(seed)

        # Unpack shared, then alias for getattr to work like the Model instance
        for k, v in shared.items():
            setattr(self, k, v)

        # Keep the full hydrologic PID->parcel-index map normalized for repeated lookups.
        self.pid_to_index = {str(pid): int(idx) for pid, idx in self.pid_to_index.items()}

        # Bind helpers with self as first arg
        self._sample_from_stats = types.MethodType(_sample_from_stats, self)
        self._piecewise_quantile_sample = types.MethodType(_piecewise_quantile_sample, self)
        self._trunc_normal = types.MethodType(_trunc_normal, self)

        self._select_bmp_type = types.MethodType(_select_bmp_type, self)
        self._get_bmp_name = types.MethodType(_get_bmp_name, self)
        self._get_pathway_load_rates = types.MethodType(_get_pathway_load_rates, self)
        self._get_current_total_load_rate = types.MethodType(_get_current_total_load_rate, self)
        self._apply_pathway_reduction = types.MethodType(_apply_pathway_reduction, self)
        self._sample_efficiency_map = types.MethodType(_sample_efficiency_map, self)    # pathway-aware
        self._simulate_wetland = types.MethodType(_simulate_wetland, self)
        self._simulate_grassed = types.MethodType(_simulate_grassed, self)
        self._simulate_infield = types.MethodType(_simulate_infield, self)
        self._get_bmp_selection_probs = types.MethodType(_get_bmp_selection_probs, self)
        self._get_bmp_cost = types.MethodType(_get_bmp_cost, self)

        self._sample_parcel_index = types.MethodType(_sample_parcel_index, self)
        self._sample_load_rate = types.MethodType(_sample_load_rate, self)
        self._get_parcel_metadata = types.MethodType(_get_parcel_metadata, self)
        self._get_parcel_up_list = types.MethodType(_get_parcel_up_list, self)
        self._get_parcel_out_oids = types.MethodType(_get_parcel_out_oids, self)
        self._delivery_coeffs = types.MethodType(_get_delivery_coeffs, self)


def _run_one_scenario(
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
    sidx: int,
    seed: int,
    outputs_dir: Path,
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Run one scenario and write its output files.

    Parameters
    ----------
    shared : dict[str, Any]
        Shared model payload used to construct the worker context.
    cfg : dict[str, Any]
        Scenario configuration mapping.
    sidx : int
        Zero-based scenario index.
    seed : int
        Seed for the worker-specific random number generator.
    outputs_dir : pathlib.Path
        Directory where scenario outputs should be written.

    Returns
    -------
    dict[tuple[str, str, str, str], list[tuple[int, float, float]]]
        Plot records for the scenario.
    """
    sid = sidx + 1
    logger = make_worker_logger(
        outputs_dir,
        scenario_id=sid,
        verbose=bool(cfg[CFG_VERBOSE]),
    )
    ctx = _ScenarioContext(cfg, shared, logger, seed)

    with log_scope(label=f"scenario {sid}", logger=logger, level=logging.INFO):
        logger.info(f"=== scenario {sid} start ===")

        n_pol = len(ctx.pollutants)
        load_state = None

        if ctx.load_generation_mode == LOAD_MODE_PLET_RUSLE:
            baseline_load_rates, load_state = initialize_plet_rusle_state(ctx)
            load_rates = baseline_load_rates.copy()
            ctx.current_pathway_load_rates = load_state.pathway_load_rates
            ctx.current_untreated_groundwater_load_rates = load_state.untreated_groundwater_load_rates
            ctx.pathway_names = ["surface", "subsurface"]
            logger.info("Generated baseline_load_rates PLET/RUSLE load_rates for surface and subsurface pathways")
        else:
            baseline_load_rates = np.zeros((len(ctx.parcel_ids), n_pol), dtype=float)
            load_rates = np.zeros_like(baseline_load_rates)
            pathway_load_rates = np.zeros(
                (len(ctx.parcel_ids), n_pol, len(ctx.pathway_names)), dtype=float
            )
            for i, _pid in enumerate(ctx.parcel_ids):
                for pol_idx in range(n_pol):
                    entry = ctx.pollutant_load_rate_stats[i][pol_idx]
                    if ctx.pollutant_load_rate_is_aggregate:
                        aggregate_load_rate = float(ctx._sample_from_stats(entry, kind="load_rate"))
                        for path_idx, path in enumerate(ctx.pathway_names):
                            pathway_load_rates[i, pol_idx, path_idx] = (
                                aggregate_load_rate * float(ctx.pollutant_load_rate_pathway_fractions[path])
                            )
                    else:
                        for path_idx, path in enumerate(ctx.pathway_names):
                            pathway_load_rates[i, pol_idx, path_idx] = float(
                                ctx._sample_from_stats(entry[path], kind="load_rate")
                            )
                    baseline_load_rates[i, pol_idx] = float(np.sum(pathway_load_rates[i, pol_idx, :]))
                    load_rates[i, pol_idx] = baseline_load_rates[i, pol_idx]
            ctx.current_pathway_load_rates = pathway_load_rates
            logger.info(
                "Generated statistical baseline_load_rates parcel load_rates across pathways: "
                + ", ".join(ctx.pathway_names)
            )

        # Limits
        limit_n = cfg.get("bmp_limit_n",None)
        limit_usd = cfg.get("bmp_limit_usd",None)
        if limit_n is None and limit_usd is None:
            logger.warning("No BMP limits specified; setting bmp limit to 1 to prevent indefinite run")
            limit_n = 1
        total_cost = 0.0
        total_bmp = 0

        # Axes and record buffers
        x_axes: List[str] = [XAXIS_COUNT]
        if cfg.get(CFG_BMP_COST):
            x_axes.append(XAXIS_COST)
        y_axes: List[str] = [YAXIS_TOTAL]
        if ctx.outlet_target_map:
            y_axes.append(YAXIS_TARGET)
        if ctx.outlet_mean_map:
            y_axes.append(YAXIS_MEAN)

        records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
        scenario_bmps: List[Dict[str, Any]] = []
        scenario_parcels: List[Dict[str, Any]] = []
        scenario_load_parameters: List[Dict[str, Any]] = []
        cumul: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Initialize summary collector once per scenario
        collector = BMPSummaryCollector(ctx.pollutants, scenario_id=sid)

        # Track CPS applied per parcel ID to prevent duplicates
        applied_by_pid: Dict[str, set] = defaultdict(set)

        # Main loop
        idle_tries = 0
        max_idle_tries = max(100, len(ctx.parcel_selection_ids) * len(ctx.bmp_cps))
        while True:
            if limit_usd is not None and total_cost >= limit_usd:
                break
            if limit_n is not None and total_bmp >= limit_n:
                break

            selection_idx = ctx._sample_parcel_index()
            parcel_idx = int(ctx.selection_source_idxs[selection_idx])
            pid = ctx.parcel_ids[parcel_idx]

            # Filter CPS by what has already been applied to this parcel
            already = applied_by_pid[str(pid)]
            allowed_idx = [i for i, c in enumerate(ctx.bmp_cps) if int(c) not in already]

            # If no CPS remain for this parcel, try another parcel; if globally exhausted, stop
            if not allowed_idx:
                idle_tries += 1
                if idle_tries >= max_idle_tries:
                    logger.info("No remaining CPS options across parcels; stopping early to avoid infinite loop.")
                    break
                continue
            idle_tries = 0  # reset on a viable placement

            # Renormalize probabilities over the allowed CPS subset
            probs_sub = ctx.bmp_selection_probs[allowed_idx]
            probs_sum = float(np.sum(probs_sub))
            if not np.isfinite(probs_sum) or probs_sum <= 0.0:
                idle_tries += 1
                if idle_tries >= max_idle_tries:
                    logger.info("No selectable CPS probabilities remain across parcels; stopping early.")
                    break
                logger.warning(
                    f"Skipping parcel pid={pid}: remaining BMP probabilities are non-finite or sum to zero."
                )
                continue
            probs_sub = probs_sub / probs_sum
            sel = ctx.rng.choice(len(allowed_idx), p=probs_sub)
            cps = int(ctx.bmp_cps[allowed_idx[sel]])

            # Scope each BMP placement for deeper indentation
            with log_scope(label=f"apply bmp cps={cps} pid={pid}", logger=logger):
                eff_maps = [
                    ctx._sample_efficiency_map(cps, pol_idx) for pol_idx in range(n_pol)
                ]

                # Optional BMP failure draw scales the sampled efficiency effects.
                failed_flag = False
                fr_cfg = ctx.cfg[CFG_BMP_FAIL_RATE]
                fail_rate = float(fr_cfg if fr_cfg is not None else 0.0)
                if fail_rate > 0.0:
                    failed = int(ctx.rng.choice([0, 1], p=[1.0 - fail_rate, fail_rate]))
                    if failed == 1:
                        red_cfg = ctx.cfg[CFG_BMP_FAIL_REDUCTION]
                        reduction = float(red_cfg if red_cfg is not None else DEFAULT_BMP_FAIL_REDUCTION)
                        eff_maps = [{k: float(v) * reduction for k, v in emap.items()} for emap in eff_maps]
                        failed_flag = True
                        ctx.logger.verbose(f"BMP failure triggered for cps={cps}; scaling efficiencies by {reduction:.2f}")

                # Per-BMP record
                bmp_rec: Dict[str, Any] = dict(
                    scenario=sid,
                    cps=cps,
                    cps_name=ctx._get_bmp_name(cps),
                    pid=str(pid),
                    **{
                        OUTPUT_IMPACTED_PIDS: "",
                        OUTPUT_LINEAR_LENGTH: None,
                        OUTPUT_BUFFER_AREA: None,
                        OUTPUT_PORTION_TREATED: None,
                        OUTPUT_WETLAND_AREA: None,
                        OUTPUT_CATCHMENT_RATIO: None,
                    },
                )
                bmp_rec[OUTPUT_BMP_FAILED] = bool(failed_flag)
                bmp_rec["load_generation_mode"] = str(ctx.load_generation_mode)

                bmp_mass_rate_outputs = {OUTPUT_TREATED: np.zeros(n_pol, dtype=float), OUTPUT_REMOVED: np.zeros(n_pol, dtype=float)}

                # Preserve the current pre-BMP load state long enough to derive a
                # true mass denominator after the BMP routine identifies its
                # impacted parcels. This is especially important for wetlands.
                pre_bmp_load_rates = load_rates.copy()

                # Apply BMP using sampled efficiency by flow pathway.
                if cps in (656, 657):
                    ctx._simulate_wetland(parcel_idx, eff_maps, load_rates, bmp_rec, bmp_mass_rate_outputs)
                    quantity = float(bmp_rec[OUTPUT_WETLAND_AREA])
                elif cps in (412,):
                    ctx._simulate_grassed(parcel_idx, eff_maps, load_rates, bmp_rec, bmp_mass_rate_outputs)
                    quantity = float(bmp_rec[OUTPUT_BUFFER_AREA]) if bmp_rec[OUTPUT_BUFFER_AREA] else 0.0
                else:
                    ctx._simulate_infield(parcel_idx, eff_maps, load_rates, bmp_rec, bmp_mass_rate_outputs)
                    quantity = float(ctx.parcel_area_ha[parcel_idx])

                # Costing and totals
                cost_this = ctx._get_bmp_cost(cps, quantity)
                total_cost += cost_this
                total_bmp += 1

                # Mark CPS as applied for this parcel
                applied_by_pid[str(pid)].add(int(cps))

                # Finalize the BMP record using explicit mass fields. The current
                # one-year timestep converts annual mass rates to timestep mass.
                bmp_rec[OUTPUT_COST_USD] = cost_this
                treated_mass_kg = np.asarray(bmp_mass_rate_outputs[OUTPUT_TREATED], dtype=float) * _CURRENT_TIMESTEP_YEARS
                removed_mass_kg = np.asarray(bmp_mass_rate_outputs[OUTPUT_REMOVED], dtype=float) * _CURRENT_TIMESTEP_YEARS
                baseline_mass_kg = _baseline_mass_before_bmp_kg(
                    ctx, pre_bmp_load_rates, parcel_idx, bmp_rec
                )
                _add_mass_metrics_to_bmp_record(
                    bmp_rec,
                    list(ctx.pollutants),
                    baseline_mass_kg,
                    treated_mass_kg,
                    removed_mass_kg,
                )
                scenario_bmps.append(bmp_rec)

                # Summary metrics are calculated from accumulated masses rather
                # than from parcel areal-load-rate denominators or averaged efficiencies.
                collector.add_bmp_record(bmp_rec)

                # Delivered reductions for plots
                oids = ctx._get_parcel_out_oids(parcel_idx)
                for pol_idx, pol in enumerate(ctx.pollutants):
                    removed_mass_rate_kg_per_yr = float(bmp_mass_rate_outputs[OUTPUT_REMOVED][pol_idx])
                    for oid in oids:
                        dr = ctx._delivery_coeffs(pid, oid)
                        deliver = (
                            removed_mass_rate_kg_per_yr * dr[COL_SDR_F_TO_S] * dr[COL_SDR_S_TO_O]
                            if pol == "TSS"
                            else removed_mass_rate_kg_per_yr * dr[COL_NDR_F_TO_S] * dr[COL_NDR_S_TO_O]
                        )
                        cumul[pol][oid] += deliver

                # Record current cumulative for each axis choice
                for pol in ctx.pollutants:
                    for oid in ctx.outlet_oids:
                        for xax in x_axes:
                            for yax in y_axes:
                                xval = total_bmp if xax == XAXIS_COUNT else total_cost
                                if yax == YAXIS_TOTAL:
                                    yval = cumul[pol][oid]
                                elif yax == YAXIS_TARGET:
                                    tgt = ctx.outlet_target_map.get((str(oid), pol))
                                    yval = (cumul[pol][oid] / tgt * 100.0) if tgt is not None and tgt > 0 else 0.0
                                elif yax == YAXIS_MEAN:
                                    mu = ctx.outlet_mean_map.get((str(oid), pol))
                                    yval = (cumul[pol][oid] / mu * 100.0) if mu is not None and mu > 0 else 0.0
                                else:
                                    yval = 0.0
                                records[(pol, oid, xax, yax)].append((sid, xval, yval))

        # Parcel-level before/after
        for parcel_idx, pid_i in enumerate(ctx.parcel_ids):
            row = dict(scenario=sid, pid=str(pid_i))
            for pol_idx, pol in enumerate(ctx.pollutants):
                # Explicit areal-load-rate columns are canonical.
                row[f"baseline_load_rate_{pol}_kg_ha_yr"] = float(baseline_load_rates[parcel_idx, pol_idx])
                row[f"final_load_rate_{pol}_kg_ha_yr"] = float(load_rates[parcel_idx, pol_idx])
            scenario_parcels.append(row)

        # Realized stochastic PLET/RUSLE inputs and derived diagnostics.
        if load_state is not None:
            for parcel_idx, pid_i in enumerate(ctx.parcel_ids):
                initial_params = load_state.baseline_parameters[parcel_idx]
                final_params = load_state.parameters[parcel_idx]
                initial_conc = load_state.baseline_concentrations[parcel_idx]
                final_conc = load_state.concentrations[parcel_idx]
                initial_gw_conc = load_state.baseline_groundwater_concentrations[parcel_idx]
                final_gw_conc = load_state.groundwater_concentrations[parcel_idx]
                row: Dict[str, Any] = {"scenario": sid, "pid": str(pid_i)}
                for key in sorted(set(initial_params) | set(final_params)):
                    row[f"initial_{key}"] = initial_params.get(key)
                    row[f"final_{key}"] = final_params.get(key)
                for pol in sorted(set(initial_conc) | set(final_conc)):
                    label = str(pol).lower()
                    row[f"initial_concentration_{label}_mg_l"] = initial_conc.get(pol)
                    row[f"final_concentration_{label}_mg_l"] = final_conc.get(pol)
                for pol in sorted(set(initial_gw_conc) | set(final_gw_conc)):
                    label = str(pol).lower()
                    row[f"initial_groundwater_concentration_{label}_mg_l"] = initial_gw_conc.get(pol)
                    row[f"final_groundwater_concentration_{label}_mg_l"] = final_gw_conc.get(pol)
                for pol_idx, pol in enumerate(ctx.pollutants):
                    label = str(pol).lower()
                    for path_idx, path in enumerate(ctx.pathway_names):
                        path_label = str(path).lower().replace(" ", "_")
                        initial_pathway_load_rate = float(
                            load_state.baseline_pathway_load_rates[parcel_idx, pol_idx, path_idx]
                        )
                        final_pathway_load_rate = float(
                            load_state.pathway_load_rates[parcel_idx, pol_idx, path_idx]
                        )
                        # Explicit annual areal-load-rate columns.
                        row[f"initial_{path_label}_{label}_load_rate_kg_ha_yr"] = initial_pathway_load_rate
                        row[f"final_{path_label}_{label}_load_rate_kg_ha_yr"] = final_pathway_load_rate
                for key, value in calculate_load_diagnostics(initial_params).items():
                    row[f"initial_{key}"] = value
                for key, value in calculate_load_diagnostics(final_params).items():
                    row[f"final_{key}"] = value
                scenario_load_parameters.append(row)

        # Build summary tables once, then write parquet outputs.
        summary_df = collector.generate_summary_dataframe()
        rollup = collector.generate_rollup_summary()
        summary_with_rollup = pd.concat([summary_df, pd.DataFrame([rollup])], ignore_index=True)

        # Write per-scenario parquet tables.
        bmps_dir = outputs_dir / "bmps"
        parcels_dir = outputs_dir / "parcels"
        load_parameters_dir = outputs_dir / "load_parameters"
        metrics_dir = outputs_dir / DIR_SCENARIO_METRICS
        output_dirs = [bmps_dir, parcels_dir, metrics_dir]
        if scenario_load_parameters:
            output_dirs.append(load_parameters_dir)
        for d in output_dirs:
            d.mkdir(parents=True, exist_ok=True)

        bmps_parquet = bmps_dir / f"s{sid}.parquet"
        parcels_parquet = parcels_dir / f"s{sid}.parquet"
        metrics_parquet = metrics_dir / f"s{sid}.parquet"
        _write_parquet_atomic(pd.DataFrame(scenario_bmps), bmps_parquet, logger=logger)
        _write_parquet_atomic(pd.DataFrame(scenario_parcels), parcels_parquet, logger=logger)
        _write_parquet_atomic(summary_with_rollup, metrics_parquet, logger=logger)
        if scenario_load_parameters:
            load_parameters_parquet = load_parameters_dir / f"s{sid}.parquet"
            _write_parquet_atomic(pd.DataFrame(scenario_load_parameters), load_parameters_parquet, logger=logger)

        logger.info(f"Wrote per-scenario BMPs parquet: {bmps_parquet}")
        logger.info(f"Wrote per-scenario parcels parquet: {parcels_parquet}")
        if scenario_load_parameters:
            logger.info(f"Wrote realized PLET/RUSLE parameters parquet: {load_parameters_parquet}")
        logger.info(f"Wrote canonical scenario metrics parquet: {metrics_parquet}")
        logger.info(f"=== scenario {sid} end (cost={total_cost:.2f}, bmp={total_bmp}) ===")
    return records