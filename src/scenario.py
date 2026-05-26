import logging
import os, sys
import numpy as np
import pandas as pd
import types
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union


from .bmp import (
    _select_bmp_type,
    _get_bmp_name,
    _sample_efficiency,
    _simulate_grassed,
    _simulate_infield,
    _simulate_wetland,
    _sample_efficiency,
    _get_bmp_selection_probs,
)


from .parcel import (
    _sample_parcel_index,
    _sample_yield,
    _get_parcel_metadata,
    _get_parcel_up_list,
    _get_parcel_out_oids,
    _get_delivery_coeffs
)


from .cost import (
    _compute_bmp_cost,
    _compute_bmp_cost_usd,
    _estimate_costs_for_probabilities,
    _select_cost_rate_median,
)


from .sampling import (
    _sample_from_stats,
    _piecewise_quantile_sample,
    _trunc_normal,
)


from .constants import (
    BMP_CPS_NAME_MAP,
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_OUTPUTS,
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
        
        # define instance variables for config, data, and logger
        self.cfg = cfg
        self.data = data
        self.logger = logger
        seed = data.get("random_seed", None)
        self.rng = np.random.default_rng(seed)
        self.outputs_dir: Optional[Path] = None
        self.parcel_ids: list[str]
        self.pid_to_index: dict[str, int]
        self.pollutants: list[str]
        self.pollutant_to_index: dict[str, int]
        self.parcel_area_ha: list[float]
        self.parcel_perim_m: list[float]
        self.parcel_out_oids: list[list[str]]
        self.parcel_up_idxs: list[list[int]]
        self.parcel_selection_ids: list[str]
        self.parcel_selection_probs: np.ndarray
        self.outlet_oids: list[str]
        self.outlet_target_map: dict[tuple[str, str], float]
        self.outlet_mean_map: dict[tuple[str, str], float]
        self.delivery_coeffs: dict[tuple[str, str], dict[str, float]]
        self.bmp_efficiency_stats: dict[int, list[Optional[dict[str, Any]]]]
        self.pollutant_yield_stats: list[list[Optional[dict[str, Any]]]]
        self.bmp_cps: list[int]
        self.bmp_selection_probs: np.ndarray

        # bind sampling function
        self._sample_from_stats = types.MethodType(_sample_from_stats, self)  # bind to instance method for easier mocking in tests
        self._piecewise_quantile_sample = types.MethodType(_piecewise_quantile_sample, self)
        self._trunc_normal = types.MethodType(_trunc_normal, self)

        # bind bmp functions
        self._select_bmp_type = types.MethodType(_select_bmp_type, self)
        self._get_bmp_name = types.MethodType(_get_bmp_name, self)
        self._sample_efficiency = types.MethodType(_sample_efficiency, self)
        self._simulate_wetland = types.MethodType(_simulate_wetland, self)
        self._simulate_grassed = types.MethodType(_simulate_grassed, self)
        self._simulate_infield = types.MethodType(_simulate_infield, self)
        self._get_bmp_selection_probs = types.MethodType(_get_bmp_selection_probs, self)
        self._compute_bmp_cost = types.MethodType(_compute_bmp_cost, self)

        # bind parcel functions
        self._sample_parcel_index = types.MethodType(_sample_parcel_index, self)
        self._sample_yield = types.MethodType(_sample_yield, self)
        self._get_parcel_metadata = types.MethodType(_get_parcel_metadata, self)
        self._get_parcel_up_list = types.MethodType(_get_parcel_up_list, self)
        self._get_parcel_out_oids = types.MethodType(_get_parcel_out_oids, self)
        self._delivery_coeffs = types.MethodType(_get_delivery_coeffs, self)

        # bind cost functions
        self._compute_bmp_cost = types.MethodType(_compute_bmp_cost, self)
        self._compute_bmp_cost_usd = types.MethodType(_compute_bmp_cost_usd, self)
        self._estimate_costs_for_probabilities = types.MethodType(_estimate_costs_for_probabilities, self)
        self._select_cost_rate_median = types.MethodType(_select_cost_rate_median, self)
        
        # prepare lookup tables
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
            list(self.data[DATA_PARCEL_OUT_MAP].get(pid, []))
            for pid in self.parcel_ids
        ]
        self.parcel_up_idxs = []
        for pid in self.parcel_ids:
            upstream = self.data[DATA_PARCEL_UP_MAP].get(pid, [])
            self.parcel_up_idxs.append([self.pid_to_index[up_pid] for up_pid in upstream if up_pid in self.pid_to_index])

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

        self.cfg.get(CFG_BMP_SEL)
        bmp_probs = self._get_bmp_selection_probs(self.cfg.get(CFG_BMP_SEL))
        self.bmp_cps = bmp_probs[COL_CPS].astype(int).tolist()
        self.bmp_selection_probs = bmp_probs[COL_PROBABILITY].astype(float).to_numpy()
        self.logger.debug(
            f"Prepared lookup tables: parcels={len(self.parcel_ids)}, pollutants={len(self.pollutants)}, "
            f"bmp_types={len(self.bmp_cps)}"
        )


    def run_all_scenarios(self) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
        """Run all configured scenarios, persist outputs, and return plotting records."""

        # set output directory
        outputs_dir = Path(self.cfg.get(CFG_OUTPUTS, "./outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir

        # define paths for outputs
        parcels_path = os.path.join(outputs_dir, "parcels.csv")
        bmps_path = os.path.join(outputs_dir, "bmps.csv")
        first_write = True

        # init scenario records for plotting
        scenario_records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
        x_axes: List[str]

        # define number of parcels and pollutants for array initializations
        n_parcels = len(self.parcel_ids)
        n_pollutants = len(self.pollutants)

        # iterate over scenarios
        for sidx in range(self.data[DATA_N_SCENARIOS]):
            self.logger.info(f"executing scenario {sidx + 1} of {self.data['n_scenarios']}")

            # initialize yields and baseline arrays for this scenario
            yields = np.empty((n_parcels, n_pollutants), dtype=float)
            baseline = np.empty_like(yields)
            self.logger.debug(f" setting baseline pollutant yields")
            for parcel_idx in range(n_parcels):
                for pol_idx, pol in enumerate(self.pollutants):
                    y = self._sample_yield(parcel_idx, pol_idx)
                    yields[parcel_idx, pol_idx] = y
                    baseline[parcel_idx, pol_idx] = y
            mean_pollutant = baseline.mean(axis=0).tolist()
            min_pollutant = baseline.min(axis=0).tolist()
            max_pollutant = baseline.max(axis=0).tolist()
            std_pollutant = baseline.std(axis=0).tolist()
            for pol_idx, pol in enumerate(self.pollutants):
                self.logger.debug(
                    f" baseline yields for {pol} (across bmp impacted parcels): "
                    f"min={min_pollutant[pol_idx]:.2f}"
                    f"mean={mean_pollutant[pol_idx]:.2f}"
                    f"std={std_pollutant[pol_idx]:.2f}"
                    f"max={max_pollutant[pol_idx]:.2f}"
                )

            # init bmp count and cost to 0
            total_cost = 0.0
            total_bmp = 0

            # init scenario limits
            limit_usd = self.data[DATA_BMP_LIMIT_USD]
            limit_n = self.data[DATA_BMP_LIMIT_N]
            self.logger.debug(f" setting scenario limits: limit_usd={limit_usd} limit_n={limit_n}")

            # cumumulative pollutant reduction by outlet and pollutant for this scenario; used for plotting at end of each scenario
            cumul: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

            # init plotting axes for this scenario
            x_axes: List[str] = []
            if self.cfg.get(CFG_BMP_COST):
                x_axes.append(XAXIS_COST)
            else:
                x_axes.append(XAXIS_COUNT)
            y_axes: List[str] = [YAXIS_TOTAL]
            if self.outlet_target_map:
                y_axes.append(YAXIS_TARGET)
            if self.outlet_mean_map:
                y_axes.append(YAXIS_MEAN)

            # init scenario bmp and parcel records for output
            scenario_bmps: List[Dict[str, Any]] = []
            scenario_parcels: List[Dict[str, Any]] = []

            # while limits are not reached, keep adding bmps to the scenario
            while True:
                if limit_usd is not None and total_cost >= limit_usd:
                    self.logger.debug(f" reached USD limit: total_cost={total_cost:.2f}, limit_usd={limit_usd:.2f}")
                    break
                if limit_n is not None and total_bmp >= limit_n:
                    self.logger.debug(f" reached BMP count limit: total_bmp={total_bmp}, limit_n={limit_n}")
                    break

                # select parcel
                parcel_idx = self._sample_parcel_index()
                pid = self.parcel_selection_ids[parcel_idx]

                # select bmp type
                cps = self._select_bmp_type()

                # set effectiveness for each pollutant for this bmp type
                eff = [self._sample_efficiency(cps, pol_idx) for pol_idx in range(n_pollutants)]

                # init bmp record with metadata and empty outputs
                bmp_rec: Dict[str, Any] = dict(
                    scenario=sidx + 1,
                    cps=cps,
                    cps_name=self._get_bmp_name(cps),
                    pid=pid,
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

                # apply the bmp to the parcel and get the outputs
                if cps in (656, 657):
                    self._simulate_wetland(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
                    quantity = float(bmp_rec[OUTPUT_WETLAND_AREA])
                elif cps in (412,):
                    self._simulate_grassed(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
                    quantity = float(bmp_rec[OUTPUT_BUFFER_AREA]) if bmp_rec[OUTPUT_BUFFER_AREA] else 0.0
                else:
                    self._simulate_infield(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
                    quantity = float(self.parcel_area_ha[parcel_idx])

                # determine cost
                cost_this = self._compute_bmp_cost(cps, None, quantity, self.logger)
                self.logger.debug(f" computed cost for this bmp application: {cost_this:.2f} USD (quantity={quantity:.4f})")

                # advance the cost and bmp count totals
                total_cost += cost_this
                total_bmp += 1
                self.logger.debug(f" updated total_cost={total_cost:.2f} total_bmp={total_bmp}")

                # log final bmp record and outputs for this scenario
                for k, v in bmp_rec.items():
                    self.logger.debug(f"  {k}: {v}")
                bmp_rec[OUTPUT_COST_USD] = cost_this
                for pol_idx, pol in enumerate(self.pollutants):
                    bmp_rec[f"{OUTPUT_TREATED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_TREATED][pol_idx])
                    bmp_rec[f"{OUTPUT_REMOVED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
                self.logger.debug(f" final bmp record: ")
                for k, v in bmp_rec.items():
                    self.logger.debug(f"  {k}: {v}")

                # add bmp record to scenario bmps list
                scenario_bmps.append(bmp_rec)

                # get outlet ids for parcel (>=1 outlets)
                oids = self._get_parcel_out_oids(parcel_idx)

                # compute cumulative delivered pollutant reduction for each outlet and pollutant, and record for plotting
                for pol_idx, pol in enumerate(self.pollutants):

                    # get pollutant removed 
                    removed_load = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
                    
                    # for each outlet
                    for oid in oids:

                        # get delivery ratio for this parcel and outlet
                        dr = self._delivery_coeffs(pid, oid)
                        # TODO: save and print to output file, print to debug log

                        # apply delivery ratio
                        if pol == "TSS":
                            deliver = removed_load * dr[COL_SDR_F_TO_S] * dr[COL_SDR_S_TO_O]
                        else:
                            deliver = removed_load * dr[COL_NDR_F_TO_S] * dr[COL_NDR_S_TO_O]

                        # add 
                        cumul[pol][oid] += deliver

                for pol in self.pollutants:
                    for oid in self.outlet_oids:
                        for xax in x_axes:
                            for yax in y_axes:
                                xval = total_bmp if xax == "count" else total_cost
                                if yax == YAXIS_TOTAL:
                                    yval = cumul[pol][oid]
                                elif yax == YAXIS_TARGET:
                                    tgt = self.outlet_target_map.get((str(oid), pol), 0.0)
                                    yval = (cumul[pol][oid] / tgt * 100.0) if tgt > 0 else 0.0
                                elif yax == YAXIS_MEAN:
                                    mu = self.outlet_mean_map.get((str(oid), pol), 0.0)
                                    yval = (cumul[pol][oid] / mu * 100.0) if mu > 0 else 0.0
                                else:
                                    yval = 0.0
                                scenario_records[(pol, oid, xax, yax)].append((sidx + 1, xval, yval))

            for parcel_idx, pid_i in enumerate(self.parcel_ids):
                rec: Dict[str, Any] = dict(scenario=sidx + 1, pid=pid_i)
                for pol_idx, pol in enumerate(self.pollutants):
                    rec[f"baseline_{pol}"] = float(baseline[parcel_idx, pol_idx])
                    rec[f"final_{pol}"] = float(yields[parcel_idx, pol_idx])
                scenario_parcels.append(rec)

            pd.DataFrame(scenario_bmps).to_csv(
                bmps_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            pd.DataFrame(scenario_parcels).to_csv(
                parcels_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False

        return scenario_records
