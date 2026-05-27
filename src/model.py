import logging
import os
import numpy as np
import pandas as pd
import types
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from joblib import Parallel, delayed
from numpy.random import SeedSequence, default_rng

from src.bmp import (
    _select_bmp_type,
    _get_bmp_name,
    _sample_efficiency,
    _simulate_grassed,
    _simulate_infield,
    _simulate_wetland,
    _get_bmp_selection_probs,
)
from src.parcel import (
    _sample_parcel_index,
    _sample_yield,
    _get_parcel_metadata,
    _get_parcel_up_list,
    _get_parcel_out_oids,
    _get_delivery_coeffs,
)
from src.cost import (
    _get_bmp_cost,
    _estimate_costs_for_probabilities,
    _select_cost_rate_median,
)
from src.sampling import (
    _sample_from_stats,
    _piecewise_quantile_sample,
    _trunc_normal,
)
from src.logging_utils import make_worker_logger

from src.constants import (
    BMP_CPS_NAME_MAP,
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_OUTPUTS,
    CFG_PARALLEL,
    COL_AREA_HA,
    COL_CPS,
    COL_OID,
    COL_PID,
    COL_POLLUTANT,
    COL_PROBABILITY,
    COL_PERIM_M,
    COL_SD,
    COL_MIN,
    COL_MAX,
    PERCENTILE_PREFIX,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    COL_TARGET,
    COL_MEAN,
    DATA_BMP_COST,
    DATA_BMP_EFFICIENCY,
    DATA_POLLUTANT_YIELD,
    DATA_PARCEL_P,
    DATA_PARCELS,
    DATA_PARCEL_UP_MAP,
    DATA_PARCEL_OUT_MAP,
    DATA_POLLUTANTS,
    DATA_CPS,
    DATA_OUTLET_LOC,
    DATA_OUTLET_TARGET,
    DATA_OUTLET_MEAN,
    DATA_BMP_LIMIT_N,
    DATA_BMP_LIMIT_USD,
    DATA_N_SCENARIOS,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_DELIVERY_RATIOS,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_COST_USD,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_PORTION_TREATED,
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
    def __init__(self, cfg: Dict[str, Any], data: Dict[str, Any], logger: logging.Logger) -> None:
        """Create a simulation instance with config, validated inputs, and logging."""
        self.cfg = cfg
        self.data = data
        self.logger = logger
        seed = data.get("random_seed", None)
        self.rng = np.random.default_rng(seed)
        self.outputs_dir: Optional[Path] = None

        # Lookup structures
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

        # Bind helper functions
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

        self._compute_bmp_cost = types.MethodType(_get_bmp_cost, self)
        self._estimate_costs_for_probabilities = types.MethodType(_estimate_costs_for_probabilities, self)
        self._select_cost_rate_median = types.MethodType(_select_cost_rate_median, self)

        # Prepare lookup tables
        self._prepare_lookup_tables()

    def _prepare_lookup_tables(self) -> None:
        """Build static lookup tables and selection arrays for fast scenario execution."""
        parcels = self.data[DATA_PARCELS]
        self.parcel_ids = parcels[COL_PID].astype(str).tolist()
        self.pid_to_index = {pid: idx for idx, pid in enumerate(self.parcel_ids)}

        self.pollutants = list(self.data[DATA_POLLUTANTS])
        self.pollutant_to_index = {pol: idx for idx, pol in enumerate(self.pollutants)}

        self.parcel_area_ha = parcels[COL_AREA_HA].astype(float).tolist()
        self.parcel_perim_m = parcels[COL_PERIM_M].astype(float).tolist()

        self.parcel_out_oids = [
            list(self.data[DATA_PARCEL_OUT_MAP].get(pid, [])) for pid in self.parcel_ids
        ]
        self.parcel_up_idxs = []
        for pid in self.parcel_ids:
            upstream = self.data[DATA_PARCEL_UP_MAP].get(pid, [])
            self.parcel_up_idxs.append(
                [self.pid_to_index[up_pid] for up_pid in upstream if up_pid in self.pid_to_index]
            )

        parcel_p = self.data[DATA_PARCEL_P]
        self.parcel_selection_ids = parcel_p[COL_PID].astype(str).tolist()
        self.parcel_selection_probs = parcel_p[COL_PROBABILITY].astype(float).to_numpy()

        self.outlet_oids = self.data[DATA_OUTLET_LOC][COL_OID].astype(str).tolist()
        self.outlet_target_map = {}
        if self.data.get(DATA_OUTLET_TARGET) is not None:
            for _, row in self.data[DATA_OUTLET_TARGET].iterrows():
                self.outlet_target_map[(str(row[COL_OID]), str(row[COL_POLLUTANT]))] = float(row[COL_TARGET])
        self.outlet_mean_map = {}
        if self.data.get(DATA_OUTLET_MEAN) is not None:
            for _, row in self.data[DATA_OUTLET_MEAN].iterrows():
                self.outlet_mean_map[(str(row[COL_OID]), str(row[COL_POLLUTANT]))] = float(row[COL_MEAN])

        self.delivery_coeffs = {}
        if self.data.get(DATA_DELIVERY_RATIOS) is not None:
            for _, row in self.data[DATA_DELIVERY_RATIOS].iterrows():
                self.delivery_coeffs[(str(row[COL_PID]), str(row[COL_OID]))] = dict(
                    sdr_f_to_s=float(row[COL_SDR_F_TO_S]),
                    sdr_s_to_o=float(row[COL_SDR_S_TO_O]),
                    ndr_f_to_s=float(row[COL_NDR_F_TO_S]),
                    ndr_s_to_o=float(row[COL_NDR_S_TO_O]),
                )

        self.bmp_efficiency_stats = {}
        n_pollutants = len(self.pollutants)
        for cps in self.data[DATA_CPS]:
            self.bmp_efficiency_stats[int(cps)] = [None] * n_pollutants
        for _, row in self.data[DATA_BMP_EFFICIENCY].iterrows():
            cps = int(row[COL_CPS])
            pol = str(row[COL_POLLUTANT])
            if pol not in self.pollutant_to_index:
                continue
            pol_idx = self.pollutant_to_index[pol]
            stats = {
                k: row[k]
                for k in row.index
                if k in (COL_MEAN, COL_SD, COL_MIN, COL_MAX)
                or (str(k).startswith(PERCENTILE_PREFIX) and str(k)[1:].isdigit())
            }
            self.bmp_efficiency_stats[cps][pol_idx] = stats

        self.pollutant_yield_stats = [[None] * n_pollutants for _ in range(len(self.parcel_ids))]
        for _, row in self.data[DATA_POLLUTANT_YIELD].iterrows():
            pid = str(row[COL_PID])
            pol = str(row[COL_POLLUTANT])
            if pid not in self.pid_to_index or pol not in self.pollutant_to_index:
                continue
            pidx = self.pid_to_index[pid]
            pol_idx = self.pollutant_to_index[pol]
            self.pollutant_yield_stats[pidx][pol_idx] = {
                k: row[k]
                for k in row.index
                if k in (COL_MEAN, COL_SD, COL_MIN, COL_MAX)
                or (str(k).startswith(PERCENTILE_PREFIX) and str(k)[1:].isdigit())
            }

        bmp_probs = self._get_bmp_selection_probs(self.cfg.get(CFG_BMP_SEL))
        self.bmp_cps = bmp_probs[COL_CPS].astype(int).tolist()
        self.bmp_selection_probs = bmp_probs[COL_PROBABILITY].astype(float).to_numpy()
        self.logger.debug(
            f"Prepared lookup tables: parcels={len(self.parcel_ids)}, pollutants={len(self.pollutants)}, "
            f"bmp_types={len(self.bmp_cps)}"
        )

    def _shared_payload(self) -> Dict[str, Any]:
        return dict(
        # arrays promoted to numpy for joblib memmapping
        parcel_area_ha=np.asarray(self.parcel_area_ha, dtype=float),
        parcel_perim_m=np.asarray(self.parcel_perim_m, dtype=float),
        parcel_selection_ids=np.asarray(self.parcel_selection_ids, dtype=object),
        parcel_selection_probs=np.asarray(self.parcel_selection_probs, dtype=float),
        # add these two lines
        parcel_ids=np.asarray(self.parcel_ids, dtype=object),
        pid_to_index=dict(self.pid_to_index),
        # small lists/dicts
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
        # scenario limits
        limit_usd=self.data.get(DATA_BMP_LIMIT_USD),
        limit_n=self.data.get(DATA_BMP_LIMIT_N),
        # cost inputs
        avg_area_ha=float(self.data[DATA_AVG_AREA_HA]),
        avg_perim_m=float(self.data[DATA_AVG_PERIM_M]),
        bmp_cost_df=self.data[DATA_BMP_COST],
        cps_list=list(self.data[DATA_CPS]),
        )

    def run_all_scenarios(
        self
    ) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
        """Run all scenarios in parallel and return plotting records.

        Each worker writes:
          - outputs/bmps_s{scenario}.csv
          - outputs/parcels_s{scenario}.csv
          - outputs/log_s{scenario}.txt
        """
        # outputs dir
        outputs_dir = Path(self.cfg.get(CFG_OUTPUTS, "./outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir

        # Shared, read-only payload for workers
        shared = self._shared_payload()

        # Parallel execution config
        par_cfg = dict(self.cfg.get(CFG_PARALLEL) or self.data.get("parallel") or {})
        n_jobs = int(par_cfg.get("n_jobs", -1))
        max_nbytes = par_cfg.get("max_nbytes", "1M")
        temp_folder = par_cfg.get("temp_folder", None)

        # Seeds for reproducibility
        base_seed = self.data.get("random_seed", None)
        n_scenarios = int(self.data[DATA_N_SCENARIOS])
        spawner = SeedSequence(base_seed) if base_seed is not None else SeedSequence()
        child_seeds = spawner.spawn(n_scenarios)

        self.logger.info(f"Executing {n_scenarios} scenarios in parallel (n_jobs={n_jobs})")

        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            max_nbytes=max_nbytes,
            temp_folder=temp_folder,
        )(
            delayed(_run_one_scenario)(
                shared, self.cfg, sidx, int(child_seeds[sidx].generate_state(1)[0]), outputs_dir
            )
            for sidx in range(n_scenarios)
        )

        # Merge scenario plotting records for plotting
        scenario_records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
        for scen_rec in results:
            for k, trip in scen_rec.items():
                scenario_records[k].extend(trip)

        # Parent no longer writes consolidated CSVs; each scenario has its own files.
        self.logger.info("All scenarios complete; per-scenario CSV and log files written.")
        return scenario_records


class _ScenarioContext:
    """Lightweight context carrying all attributes used by helpers and scenario logic."""

    def __init__(self, cfg: Dict[str, Any], shared: Dict[str, Any], logger: logging.Logger, seed: int) -> None:
        self.cfg = cfg
        self.logger = logger
        self.rng = default_rng(seed)

        # Assign arrays and structures expected by helper functions
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

        # Minimal data dict for cost helpers
        self.data = {
            DATA_AVG_AREA_HA: shared["avg_area_ha"],
            DATA_AVG_PERIM_M: shared["avg_perim_m"],
            DATA_BMP_COST: shared["bmp_cost_df"],
            DATA_CPS: shared["cps_list"],
        }

        # Bind helper methods to this context
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
    """Execute a single scenario, write its CSV outputs and log, and return plotting records."""
    sid = sidx + 1
    logger = make_worker_logger(outputs_dir, scenario_id=sid)
    ctx = _ScenarioContext(cfg, shared, logger, seed)

    n_parcels = int(len(ctx.parcel_ids))
    n_pollutants = int(len(ctx.pollutants))

    logger.info(f"=== scenario {sid} start ===")

    # Initialize yields and baseline
    yields = np.empty((n_parcels, n_pollutants), dtype=float)
    baseline = np.empty_like(yields)
    logger.debug("setting baseline pollutant yields")
    for parcel_idx in range(n_parcels):
        for pol_idx, _ in enumerate(ctx.pollutants):
            y = ctx._sample_yield(parcel_idx, pol_idx)
            yields[parcel_idx, pol_idx] = y
            baseline[parcel_idx, pol_idx] = y

    # Log summary stats by pollutant
    mean_pollutant = baseline.mean(axis=0).tolist()
    min_pollutant = baseline.min(axis=0).tolist()
    max_pollutant = baseline.max(axis=0).tolist()
    std_pollutant = baseline.std(axis=0).tolist()
    for pol_idx, pol in enumerate(ctx.pollutants):
        logger.debug(
            f"baseline yields for {pol}: "
            f"min={min_pollutant[pol_idx]:.2f} "
            f"mean={mean_pollutant[pol_idx]:.2f} "
            f"std={std_pollutant[pol_idx]:.2f} "
            f"max={max_pollutant[pol_idx]:.2f}"
        )

    total_cost = 0.0
    total_bmp = 0
    limit_usd = shared["limit_usd"]
    limit_n = shared["limit_n"]
    logger.debug(f"setting scenario limits: limit_usd={limit_usd} limit_n={limit_n}")

    # Plotting axes for this scenario
    x_axes: List[str] = [XAXIS_COST] if cfg.get(CFG_BMP_COST) else [XAXIS_COUNT]
    y_axes: List[str] = [YAXIS_TOTAL]
    if ctx.outlet_target_map:
        y_axes.append(YAXIS_TARGET)
    if ctx.outlet_mean_map:
        y_axes.append(YAXIS_MEAN)

    scenario_bmps: List[Dict[str, Any]] = []
    scenario_parcels: List[Dict[str, Any]] = []
    scenario_records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
    cumul: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Main scenario loop
    while True:
        if limit_usd is not None and total_cost >= limit_usd:
            logger.debug(f"reached USD limit: total_cost={total_cost:.2f}, limit_usd={float(limit_usd):.2f}")
            break
        if limit_n is not None and total_bmp >= limit_n:
            logger.debug(f"reached BMP count limit: total_bmp={total_bmp}, limit_n={int(limit_n)}")
            break

        parcel_idx = ctx._sample_parcel_index()
        pid = ctx.parcel_selection_ids[parcel_idx]

        cps = ctx._select_bmp_type()
        eff = [ctx._sample_efficiency(cps, pol_idx) for pol_idx in range(n_pollutants)]

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
        bmp_outputs: Dict[str, np.ndarray] = {
            OUTPUT_TREATED: np.zeros(n_pollutants, dtype=float),
            OUTPUT_REMOVED: np.zeros(n_pollutants, dtype=float),
        }

        # Apply the BMP
        if cps in (656, 657):
            ctx._simulate_wetland(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
            quantity = float(bmp_rec[OUTPUT_WETLAND_AREA])
        elif cps in (412,):
            ctx._simulate_grassed(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
            quantity = float(bmp_rec[OUTPUT_BUFFER_AREA]) if bmp_rec[OUTPUT_BUFFER_AREA] else 0.0
        else:
            ctx._simulate_infield(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
            quantity = float(ctx.parcel_area_ha[parcel_idx])

        # Cost
        cost_this = ctx._get_bmp_cost(cps, quantity)
        logger.debug(f"computed cost for this bmp application: {cost_this:.2f} USD (quantity={quantity:.4f})")
        total_cost += cost_this
        total_bmp += 1
        logger.debug(f"updated total_cost={total_cost:.2f} total_bmp={total_bmp}")

        # Finalize record
        for k, v in bmp_rec.items():
            logger.debug(f"{k}: {v}")
        bmp_rec[OUTPUT_COST_USD] = cost_this
        for pol_idx, pol in enumerate(ctx.pollutants):
            bmp_rec[f"{OUTPUT_TREATED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_TREATED][pol_idx])
            bmp_rec[f"{OUTPUT_REMOVED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
        logger.debug("final bmp record:")
        for k, v in bmp_rec.items():
            logger.debug(f"{k}: {v}")

        scenario_bmps.append(bmp_rec)

        # Delivery and plotting records
        oids = ctx._get_parcel_out_oids(parcel_idx)
        for pol_idx, pol in enumerate(ctx.pollutants):
            removed_load = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
            for oid in oids:
                dr = ctx._delivery_coeffs(pid, oid)
                if pol == "TSS":
                    deliver = removed_load * dr[COL_SDR_F_TO_S] * dr[COL_SDR_S_TO_O]
                else:
                    deliver = removed_load * dr[COL_NDR_F_TO_S] * dr[COL_NDR_S_TO_O]
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
                        scenario_records[(pol, oid, xax, yax)].append((sid, xval, yval))

    # Parcel-level summary for this scenario
    for parcel_idx, pid_i in enumerate(ctx.parcel_selection_ids):
        rec: Dict[str, Any] = dict(scenario=sid, pid=str(pid_i))
        for pol_idx, pol in enumerate(ctx.pollutants):
            rec[f"baseline_{pol}"] = float(baseline[parcel_idx, pol_idx])
            rec[f"final_{pol}"] = float(yields[parcel_idx, pol_idx])
        scenario_parcels.append(rec)

    # Write per-scenario CSVs
    bmps_path = Path(outputs_dir) / f"bmps_s{sid}.csv"
    parcels_path = Path(outputs_dir) / f"parcels_s{sid}.csv"
    pd.DataFrame(scenario_bmps).to_csv(bmps_path, index=False)
    pd.DataFrame(scenario_parcels).to_csv(parcels_path, index=False)
    logger.info(f"Wrote per-scenario BMPs: {bmps_path}")
    logger.info(f"Wrote per-scenario parcels: {parcels_path}")

    logger.info(f"=== scenario {sid} end (cost={total_cost:.2f}, bmp={total_bmp}) ===")
    return scenario_records