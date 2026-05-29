"""
Scenario orchestration (parallel execution and per-scenario outputs).

This module coordinates:
- Preparing lookup structures for fast scenario execution
- Running scenarios in parallel
- Writing per-scenario CSVs to organized subfolders (Option A)
- Producing transposed per-scenario summaries with an "All CPS" column
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
    _get_bmp_name,
    _get_bmp_selection_probs,
    _sample_efficiency,
    _select_bmp_type,
    _simulate_grassed,
    _simulate_infield,
    _simulate_wetland,
)
from src.cost import _estimate_costs_for_probabilities, _get_bmp_cost, _select_cost_rate_median
from src.logging_utils import make_worker_logger
from src.parcel import (
    _get_delivery_coeffs,
    _get_parcel_metadata,
    _get_parcel_out_oids,
    _get_parcel_up_list,
    _sample_parcel_index,
    _sample_yield,
)
from src.sampling import _piecewise_quantile_sample, _sample_from_stats, _trunc_normal
from src.summaries import BMPSummaryCollector
from src.constants import (
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_OUTPUTS,
    CFG_PARALLEL,
    OUTPUT_PORTION_TREATED,
    COL_POLLUTANT,
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
    DATA_OUTLET_LOC,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_PARCEL_OUT_MAP,
    DATA_PARCEL_P,
    DATA_PARCEL_UP_MAP,
    DATA_PARCELS,
    DATA_POLLUTANT_YIELD,
    DATA_POLLUTANTS,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_COST_USD,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_REMOVED,
    OUTPUT_REMOVED_PREFIX,
    OUTPUT_TREATED,
    OUTPUT_TREATED_PREFIX,
    OUTPUT_WETLAND_AREA,
    XAXIS_COST,
    XAXIS_COUNT,
    YAXIS_MEAN,
    YAXIS_TARGET,
    YAXIS_TOTAL,
)


class Model:
    """Main simulation orchestrator for running multiple scenarios.

    Parameters
    ----------
    cfg : Dict[str, Any]
        User configuration (normalized to lowercase keys).
    data : Dict[str, Any]
        Validated input payload from io_utils.load_and_validate_all.
    logger : logging.Logger
        Root logger.

    Notes
    -----
    - Uses joblib for parallel scenario execution.
    - Per-scenario CSVs and logs are written by worker processes.
    """

    def __init__(self, cfg: Dict[str, Any], data: Dict[str, Any], logger: logging.Logger) -> None:
        self.cfg = cfg
        self.data = data
        self.logger = logger
        seed = data.get("random_seed", None)
        self.rng = np.random.default_rng(seed)
        self.outputs_dir: Optional[Path] = None

        # Prepared lookup structures (populated in _prepare_lookup_tables)
        self.parcel_ids: List[str]
        self.pid_to_index: Dict[str, int]
        self.pollutants: List[str]
        self.pollutant_to_index: Dict[str, int]
        self.parcel_area_ha: List[float]
        self.parcel_perim_m: List[float]
        self.parcel_out_oids: List[List[str]]
        self.parcel_up_idxs: List[List[int]]
        self.parcel_selection_ids: List[str]
        self.parcel_selection_probs: np.ndarray
        self.outlet_oids: List[str]
        self.outlet_target_map: Dict[Tuple[str, str], float]
        self.outlet_mean_map: Dict[Tuple[str, str], float]
        self.delivery_coeffs: Dict[Tuple[str, str], Dict[str, float]]
        self.bmp_efficiency_stats: Dict[int, List[Optional[Dict[str, Any]]]]
        self.pollutant_yield_stats: List[List[Optional[Dict[str, Any]]]]
        self.bmp_cps: List[int]
        self.bmp_selection_probs: np.ndarray

        # Bind helper functions used by the model instance
        self._sample_from_stats = types.MethodType(_sample_from_stats, self)
        self._piecewise_quantile_sample = types.MethodType(_piecewise_quantile_sample, self)
        self._trunc_normal = types.MethodType(_trunc_normal, self)

        self._select_bmp_type = types.MethodType(_select_bmp_type, self)
        self._get_bmp_name = types.MethodType(_get_bmp_name, self)
        self._sample_efficiency = types.MethodType(_sample_efficiency, self)
        self._simulate_wetland = types.MethodType(_simulate_wetland, self)
        self._simulate_grassed = types.MethodType(_simulate_grassed, self)
        self._simulate_infield = types.MethodType(_simulate_infield, self)
        self._get_bmp_selection_probs = types.MethodType(_get_bmp_selection_probs, self)
        self._get_bmp_cost = types.MethodType(_get_bmp_cost, self)

        self._sample_parcel_index = types.MethodType(_sample_parcel_index, self)
        self._sample_yield = types.MethodType(_sample_yield, self)
        self._get_parcel_metadata = types.MethodType(_get_parcel_metadata, self)
        self._get_parcel_up_list = types.MethodType(_get_parcel_up_list, self)
        self._get_parcel_out_oids = types.MethodType(_get_parcel_out_oids, self)
        self._delivery_coeffs = types.MethodType(_get_delivery_coeffs, self)

        self._estimate_costs_for_probabilities = types.MethodType(_estimate_costs_for_probabilities, self)
        self._select_cost_rate_median = types.MethodType(_select_cost_rate_median, self)

        self._prepare_lookup_tables()

    def _prepare_lookup_tables(self) -> None:
        """Assemble arrays and mappings used during scenario execution."""
        parcels = self.data[DATA_PARCELS]
        self.parcel_ids = parcels["pid"].astype(str).tolist()
        self.pid_to_index = {pid: idx for idx, pid in enumerate(self.parcel_ids)}

        self.pollutants = list(self.data[DATA_POLLUTANTS])
        self.pollutant_to_index = {pol: idx for idx, pol in enumerate(self.pollutants)}

        self.parcel_area_ha = parcels["area_ha"].astype(float).tolist()
        self.parcel_perim_m = parcels["perim_m"].astype(float).tolist()

        self.parcel_out_oids = [list(self.data[DATA_PARCEL_OUT_MAP].get(pid, [])) for pid in self.parcel_ids]

        self.parcel_up_idxs = []
        for pid in self.parcel_ids:
            upstream = self.data[DATA_PARCEL_UP_MAP].get(pid, [])
            self.parcel_up_idxs.append([self.pid_to_index[up_pid] for up_pid in upstream if up_pid in self.pid_to_index])

        parcel_p = self.data[DATA_PARCEL_P]
        self.parcel_selection_ids = parcel_p["pid"].astype(str).tolist()
        self.parcel_selection_probs = parcel_p["probability"].astype(float).to_numpy()

        self.outlet_oids = self.data[DATA_OUTLET_LOC]["oid"].astype(str).tolist()

        self.outlet_target_map = {}
        if self.data.get(DATA_OUTLET_TARGET) is not None:
            for _, row in self.data[DATA_OUTLET_TARGET].iterrows():
                self.outlet_target_map[(str(row["oid"]), str(row[COL_POLLUTANT]))] = float(row["target"])

        self.outlet_mean_map = {}
        if self.data.get(DATA_OUTLET_MEAN) is not None:
            for _, row in self.data[DATA_OUTLET_MEAN].iterrows():
                self.outlet_mean_map[(str(row["oid"]), str(row[COL_POLLUTANT]))] = float(row["mean"])

        self.delivery_coeffs = {}
        if self.data.get(DATA_DELIVERY_RATIOS) is not None:
            for _, row in self.data[DATA_DELIVERY_RATIOS].iterrows():
                self.delivery_coeffs[(str(row["pid"]), str(row["oid"]))] = dict(
                    sdr_f_to_s=float(row[COL_SDR_F_TO_S]),
                    sdr_s_to_o=float(row[COL_SDR_S_TO_O]),
                    ndr_f_to_s=float(row[COL_NDR_F_TO_S]),
                    ndr_s_to_o=float(row[COL_NDR_S_TO_O]),
                )

        # Efficiency stats lookup per CPS x pollutant
        self.bmp_efficiency_stats = {}
        n_pol = len(self.pollutants)
        for cps in self.data[DATA_CPS]:
            self.bmp_efficiency_stats[int(cps)] = [None] * n_pol
        for _, row in self.data[DATA_BMP_EFFICIENCY].iterrows():
            cps = int(row["cps"])
            pol = str(row[COL_POLLUTANT])
            if pol not in self.pollutant_to_index:
                continue
            pol_idx = self.pollutant_to_index[pol]
            stats = {k: row[k] for k in row.index if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())}
            self.bmp_efficiency_stats[cps][pol_idx] = stats

        # Parcel pollutant yield stats lookup per parcel x pollutant
        self.pollutant_yield_stats = [[None] * n_pol for _ in range(len(self.parcel_ids))]
        for _, row in self.data[DATA_POLLUTANT_YIELD].iterrows():
            pid = str(row["pid"])
            pol = str(row[COL_POLLUTANT])
            if pid not in self.pid_to_index or pol not in self.pollutant_to_index:
                continue
            pidx = self.pid_to_index[pid]
            pol_idx = self.pollutant_to_index[pol]
            self.pollutant_yield_stats[pidx][pol_idx] = {k: row[k] for k in row.index if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())}

        # BMP selection probabilities
        bmp_probs = self._get_bmp_selection_probs(self.cfg.get(CFG_BMP_SEL))
        self.bmp_cps = bmp_probs["cps"].astype(int).tolist()
        self.bmp_selection_probs = bmp_probs["probability"].astype(float).to_numpy()

        self.logger.debug(
            f"Prepared lookup tables: parcels={len(self.parcel_ids)}, pollutants={len(self.pollutants)}, "
            f"bmp_types={len(self.bmp_cps)}"
        )

    def _shared_payload(self) -> Dict[str, Any]:
        """Create the shared, read-only payload for joblib worker processes."""
        return dict(
            parcel_area_ha=np.asarray(self.parcel_area_ha, dtype=float),
            parcel_perim_m=np.asarray(self.parcel_perim_m, dtype=float),
            parcel_selection_ids=np.asarray(self.parcel_selection_ids, dtype=object),
            parcel_selection_probs=np.asarray(self.parcel_selection_probs, dtype=float),
            parcel_ids=np.asarray(self.parcel_ids, dtype=object),
            pid_to_index=dict(self.pid_to_index),
            pollutants=list(self.pollutants),
            outlet_oids=list(self.outlet_oids),
            outlet_target_map=dict(self.outlet_target_map),
            outlet_mean_map=dict(self.outlet_mean_map),
            parcel_out_oids=[list(x) for x in self.parcel_out_oids],
            parcel_up_idxs=[list(x) for x in self.parcel_up_idxs],
            delivery_coeffs=dict(self.delivery_coeffs),
            bmp_efficiency_stats={int(k): list(v) for k, v in self.bmp_efficiency_stats.items()},
            pollutant_yield_stats=[[dict(s) if s is not None else None for s in row] for row in self.pollutant_yield_stats],
            bmp_cps=list(self.bmp_cps),
            bmp_selection_probs=np.asarray(self.bmp_selection_probs, dtype=float),
            limit_usd=self.data.get(DATA_BMP_LIMIT_USD),
            limit_n=self.data.get(DATA_BMP_LIMIT_N),
            avg_area_ha=float(self.data[DATA_AVG_AREA_HA]),
            avg_perim_m=float(self.data[DATA_AVG_PERIM_M]),
            bmp_cost_df=self.data[DATA_BMP_COST],
            cps_list=list(self.data[DATA_CPS]),
        )

    def run_all_scenarios(
        self,
    ) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
        """Run all scenarios in parallel and return plotting records.

        Returns
        -------
        Dict[(str, str, str, str), List[(int, float, float)]]
            Mapping (pollutant, outlet_oid, x_axis, y_axis) -> list of
            (scenario_id, x, y) tuples used by plotting.

        Notes
        -----
        Each worker writes:
          - outputs/bmps/s{scenario}.csv
          - outputs/parcels/s{scenario}.csv
          - outputs/summaries/s{scenario}.csv (transposed; includes "All CPS" column)
          - outputs/logs/s{scenario}.txt
        """
        outputs_dir = Path(self.cfg.get(CFG_OUTPUTS, "./outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir

        shared = self._shared_payload()

        par_cfg = dict(self.cfg.get(CFG_PARALLEL) or self.data.get("parallel") or {})
        n_jobs = int(par_cfg.get("n_jobs", -1))
        max_nbytes = par_cfg.get("max_nbytes", "1M")
        temp_folder = par_cfg.get("temp_folder", None)

        base_seed = self.data.get("random_seed", None)
        n_scenarios = int(self.data[DATA_N_SCENARIOS])
        spawner = SeedSequence(base_seed) if base_seed is not None else SeedSequence()
        child_seeds = spawner.spawn(n_scenarios)

        self.logger.info(f"Executing {n_scenarios} scenarios in parallel (n_jobs={n_jobs})")

        results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes=max_nbytes, temp_folder=temp_folder)(
            delayed(_run_one_scenario)(shared, self.cfg, sidx, int(child_seeds[sidx].generate_state(1)[0]), outputs_dir)
            for sidx in range(n_scenarios)
        )

        scenario_records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
        for scen_rec in results:
            for k, trip in scen_rec.items():
                scenario_records[k].extend(trip)

        self.logger.info("All scenarios complete; per-scenario CSV and log files written.")
        return scenario_records


class _ScenarioContext:
    """Lightweight container for scenario helpers and state bound to a worker RNG.

    Rationale
    ---------
    Worker processes should not rely on global state. We bind necessary free
    functions as methods on this context and store only the minimal data needed
    for the simulation step.
    """

    def __init__(self, cfg: Dict[str, Any], shared: Dict[str, Any], logger: logging.Logger, seed: int) -> None:
        self.cfg = cfg
        self.logger = logger
        self.rng = default_rng(seed)

        # Arrays/dicts used by helpers
        self.parcel_ids = shared["parcel_ids"]
        self.pid_to_index = shared["pid_to_index"]
        self.pollutants = shared["pollutants"]
        self.parcel_area_ha = shared["parcel_area_ha"]
        self.parcel_perim_m = shared["parcel_perim_m"]
        self.parcel_selection_ids = shared["parcel_selection_ids"]
        self.parcel_selection_probs = shared["parcel_selection_probs"]
        self.parcel_out_oids = shared["parcel_out_oids"]
        self.parcel_up_idxs = shared["parcel_up_idxs"]

        self.outlet_oids = shared["outlet_oids"]
        self.outlet_target_map = shared["outlet_target_map"]
        self.outlet_mean_map = shared["outlet_mean_map"]
        self.delivery_coeffs = shared["delivery_coeffs"]

        self.bmp_efficiency_stats = shared["bmp_efficiency_stats"]
        self.pollutant_yield_stats = shared["pollutant_yield_stats"]

        self.bmp_cps = shared["bmp_cps"]
        self.bmp_selection_probs = shared["bmp_selection_probs"]

        # Minimal data payload for cost helpers
        self.data = {
            DATA_AVG_AREA_HA: shared["avg_area_ha"],
            DATA_AVG_PERIM_M: shared["avg_perim_m"],
            DATA_BMP_COST: shared["bmp_cost_df"],
            DATA_CPS: shared["cps_list"],
        }

        # Bind helpers
        self._sample_from_stats = types.MethodType(_sample_from_stats, self)
        self._piecewise_quantile_sample = types.MethodType(_piecewise_quantile_sample, self)
        self._trunc_normal = types.MethodType(_trunc_normal, self)

        self._select_bmp_type = types.MethodType(_select_bmp_type, self)
        self._get_bmp_name = types.MethodType(_get_bmp_name, self)
        self._sample_efficiency = types.MethodType(_sample_efficiency, self)
        self._simulate_wetland = types.MethodType(_simulate_wetland, self)
        self._simulate_grassed = types.MethodType(_simulate_grassed, self)
        self._simulate_infield = types.MethodType(_simulate_infield, self)
        self._get_bmp_selection_probs = types.MethodType(_get_bmp_selection_probs, self)

        self._sample_parcel_index = types.MethodType(_sample_parcel_index, self)
        self._sample_yield = types.MethodType(_sample_yield, self)
        self._get_parcel_metadata = types.MethodType(_get_parcel_metadata, self)
        self._get_parcel_up_list = types.MethodType(_get_parcel_up_list, self)
        self._get_parcel_out_oids = types.MethodType(_get_parcel_out_oids, self)
        self._delivery_coeffs = types.MethodType(_get_delivery_coeffs, self)

        self._get_bmp_cost = types.MethodType(_get_bmp_cost, self)
        self._estimate_costs_for_probabilities = types.MethodType(_estimate_costs_for_probabilities, self)
        self._select_cost_rate_median = types.MethodType(_select_cost_rate_median, self)


def _run_one_scenario(
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
    sidx: int,
    seed: int,
    outputs_dir: Path,
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Execute one scenario and write its outputs.

    Parameters
    ----------
    shared : Dict[str, Any]
        Read-only arrays and mappings for worker processes.
    cfg : Dict[str, Any]
        User configuration.
    sidx : int
        Zero-based scenario index; scenario id is sidx+1.
    seed : int
        RNG seed unique to this worker.
    outputs_dir : Path
        Root outputs directory.

    Returns
    -------
    Dict[(str, str, str, str), List[(int, float, float)]]
        Records for plotting keyed by (pollutant, outlet_oid, x_axis, y_axis).
    """
    sid = sidx + 1
    logger = make_worker_logger(outputs_dir, scenario_id=sid)
    ctx = _ScenarioContext(cfg, shared, logger, seed)

    n_parcels = int(len(ctx.parcel_ids))
    n_pol = int(len(ctx.pollutants))

    # Baseline yields
    yields = np.empty((n_parcels, n_pol), dtype=float)
    baseline = np.empty_like(yields)
    for parcel_idx in range(n_parcels):
        for pol_idx in range(n_pol):
            v = ctx._sample_yield(parcel_idx, pol_idx)
            yields[parcel_idx, pol_idx] = v
            baseline[parcel_idx, pol_idx] = v

    total_cost = 0.0
    total_bmp = 0
    limit_usd = shared["limit_usd"]
    limit_n = shared["limit_n"]

    collector = BMPSummaryCollector(ctx.pollutants, sid)
    pid_to_parcel_idx = {pid: idx for idx, pid in enumerate(ctx.parcel_selection_ids)}

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

    cumul: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Main placement loop
    while True:
        if limit_usd is not None and total_cost >= limit_usd:
            break
        if limit_n is not None and total_bmp >= limit_n:
            break

        parcel_idx = ctx._sample_parcel_index()
        pid = ctx.parcel_selection_ids[parcel_idx]
        cps = ctx._select_bmp_type()
        eff = [ctx._sample_efficiency(cps, pol_idx) for pol_idx in range(n_pol)]

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
        bmp_outputs = {OUTPUT_TREATED: np.zeros(n_pol, dtype=float), OUTPUT_REMOVED: np.zeros(n_pol, dtype=float)}

        # Apply BMP
        if cps in (656, 657):
            ctx._simulate_wetland(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
            quantity = float(bmp_rec[OUTPUT_WETLAND_AREA])
        elif cps in (412,):
            ctx._simulate_grassed(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
            quantity = float(bmp_rec[OUTPUT_BUFFER_AREA]) if bmp_rec[OUTPUT_BUFFER_AREA] else 0.0
        else:
            ctx._simulate_infield(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
            quantity = float(ctx.parcel_area_ha[parcel_idx])

        # Costing and totals
        cost_this = ctx._get_bmp_cost(cps, quantity)
        total_cost += cost_this
        total_bmp += 1

        bmp_rec[OUTPUT_COST_USD] = cost_this
        for pol_idx, pol in enumerate(ctx.pollutants):
            bmp_rec[f"{OUTPUT_TREATED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_TREATED][pol_idx])
            bmp_rec[f"{OUTPUT_REMOVED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
        scenario_bmps.append(bmp_rec)

        # Add to collector
        pidx_base = pid_to_parcel_idx.get(str(pid), 0)
        pid_baseline_yields = {pol: float(baseline[pidx_base, i]) for i, pol in enumerate(ctx.pollutants)}
        collector.add_bmp_record(bmp_rec, pid_baseline_yields)

        # Delivered reductions for plots
        oids = ctx._get_parcel_out_oids(parcel_idx)
        for pol_idx, pol in enumerate(ctx.pollutants):
            removed = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
            for oid in oids:
                dr = ctx._delivery_coeffs(pid, oid)
                deliver = (
                    removed * dr[COL_SDR_F_TO_S] * dr[COL_SDR_S_TO_O]
                    if pol == "TSS"
                    else removed * dr[COL_NDR_F_TO_S] * dr[COL_NDR_S_TO_O]
                )
                cumul[pol][oid] += deliver

        for pol in ctx.pollutants:
            for oid in ctx.outlet_oids:
                for xax in x_axes:
                    for yax in y_axes:
                        xval = total_bmp if xax == XAXIS_COUNT else total_cost
                        if yax == YAXIS_TOTAL:
                            yval = cumul[pol][oid]
                        elif yax == YAXIS_TARGET:
                            tgt = ctx.outlet_target_map.get((str(oid), pol), 0.0)
                            yval = (cumul[pol][oid] / tgt * 100.0) if tgt > 0 else 0.0
                        elif yax == YAXIS_MEAN:
                            mu = ctx.outlet_mean_map.get((str(oid), pol), 0.0)
                            yval = (cumul[pol][oid] / mu * 100.0) if mu > 0 else 0.0
                        else:
                            yval = 0.0
                        records[(pol, oid, xax, yax)].append((sid, xval, yval))

    # Parcel-level before/after
    for parcel_idx, pid_i in enumerate(ctx.parcel_selection_ids):
        row = dict(scenario=sid, pid=str(pid_i))
        for pol_idx, pol in enumerate(ctx.pollutants):
            row[f"baseline_{pol}"] = float(baseline[parcel_idx, pol_idx])
            row[f"final_{pol}"] = float(yields[parcel_idx, pol_idx])
        scenario_parcels.append(row)

    # Write CSVs (Option A)
    bmps_dir = outputs_dir / "bmps"
    parcels_dir = outputs_dir / "parcels"
    summaries_dir = outputs_dir / "summaries"
    for d in (bmps_dir, parcels_dir, summaries_dir):
        d.mkdir(parents=True, exist_ok=True)

    bmps_path = bmps_dir / f"s{sid}.csv"
    parcels_path = parcels_dir / f"s{sid}.csv"
    summary_path = summaries_dir / f"s{sid}.csv"

    pd.DataFrame(scenario_bmps).to_csv(bmps_path, index=False)
    pd.DataFrame(scenario_parcels).to_csv(parcels_path, index=False)

    # Transposed per-scenario summary + "All CPS" rollup
    summary_df = collector.generate_summary_dataframe()
    rollup = collector.generate_rollup_summary()
    summary_with_rollup = pd.concat([summary_df, pd.DataFrame([rollup])], ignore_index=True)

    col_labels = []
    for _, r in summary_with_rollup.iterrows():
        if str(r["cps_name"]) == "All CPS":
            col_labels.append(f"s{int(r['scenario'])}-All CPS")
        else:
            col_labels.append(f"s{int(r['scenario'])}-{str(r['cps_name'])}({int(r['cps'])})")

    tdf = summary_with_rollup.T
    tdf.columns = col_labels
    tdf = tdf.reset_index().rename(columns={"index": "field"})
    tdf.to_csv(summary_path, index=False)

    logger.info(f"Wrote per-scenario BMPs: {bmps_path}")
    logger.info(f"Wrote per-scenario parcels: {parcels_path}")
    logger.info(f"Wrote transposed BMP summary with All CPS: {summary_path}")
    logger.info(f"=== scenario {sid} end (cost={total_cost:.2f}, bmp={total_bmp}) ===")
    return records