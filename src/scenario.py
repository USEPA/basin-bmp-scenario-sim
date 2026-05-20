import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from .bmp import (
    compute_bmp_cost,
    simulate_grassed,
    simulate_infield,
    simulate_wetland,
)
from .sampling import sample_from_stats
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
from .selection import estimate_costs_for_probabilities


class Simulator:
    def __init__(self, cfg: Dict[str, Any], data: Dict[str, Any], logger: logging.Logger) -> None:
        """Create a simulation instance with config, validated inputs, and logging."""
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

        bmp_probs = self._get_bmp_selection_probs()
        self.bmp_cps = bmp_probs[COL_CPS].astype(int).tolist()
        self.bmp_selection_probs = bmp_probs[COL_PROBABILITY].astype(float).to_numpy()

    def _get_bmp_selection_probs(self) -> pd.DataFrame:
        """Return BMP type selection probabilities for the scenario loop.

        If an explicit probability file is provided via cfg[bmp_sel], use it.
        Otherwise derive weights from estimated costs so lower-cost BMPs are more likely.
        """
        import pandas as pd

        bmp_sel_path = self.cfg.get(CFG_BMP_SEL)
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
            return df[[COL_CPS, COL_PROBABILITY]]
        else:
            if self.data[DATA_BMP_COST] is None:
                probs = np.full(len(self.data[DATA_CPS]), 1.0 / len(self.data[DATA_CPS]))
                return pd.DataFrame({COL_CPS: self.data[DATA_CPS], COL_PROBABILITY: probs})
            else:
                df = estimate_costs_for_probabilities(
                    self.rng,
                    self.data[DATA_BMP_COST],
                    self.data[DATA_CPS],
                    self.data[DATA_AVG_AREA_HA],
                    self.data[DATA_AVG_PERIM_M],
                    overrides={},
                )
                return df

    def _sample_efficiency(self, cps: Union[int, str], pol_idx: int) -> float:
        """Sample BMP efficiency for a given CPS type and pollutant index."""
        cps_key = int(cps)
        stats = self.bmp_efficiency_stats[cps_key][pol_idx]
        if stats is None:
            raise KeyError(f"No BMP efficiency stats found for cps={cps_key}, pollutant={self.pollutants[pol_idx]}")
        return sample_from_stats(
            self.rng,
            stats,
            kind="efficiency",
            verbose_logger=self.logger,
            ctx=f"cps={cps_key},pollutant={self.pollutants[pol_idx]}",
        )

    def _sample_yield(self, parcel_idx: int, pol_idx: int) -> float:
        """Sample baseline pollutant yield for a parcel and pollutant index."""
        stats = self.pollutant_yield_stats[parcel_idx][pol_idx]
        if stats is None:
            raise KeyError(
                f"No pollutant yield stats found for pid={self.parcel_ids[parcel_idx]}, pollutant={self.pollutants[pol_idx]}"
            )
        return sample_from_stats(
            self.rng,
            stats,
            kind="yield",
            verbose_logger=self.logger,
            ctx=f"pid={self.parcel_ids[parcel_idx]},pollutant={self.pollutants[pol_idx]}",
        )

    def _select_parcel_index(self) -> int:
        """Choose a parcel index randomly from parcel selection probabilities."""
        idx = self.rng.choice(len(self.parcel_selection_ids), p=self.parcel_selection_probs)
        self.logger.debug(
            f"Random parcel selection idx={idx} pid={self.parcel_selection_ids[idx]} probs_sample={self.parcel_selection_probs} ctx=parcel_selection"
        )
        return idx

    def _select_bmp_type(self) -> int:
        """Choose a BMP type code from the probability distribution."""
        idx = self.rng.choice(len(self.bmp_cps), p=self.bmp_selection_probs)
        cps = int(self.bmp_cps[idx])
        self.logger.debug(
            f"Random BMP type selection idx={idx} cps={cps} probs_sample={self.bmp_selection_probs} ctx=bmp_selection"
        )
        return cps

    def _get_bmp_name(self, cps: Union[int, str]) -> str:
        """Return the human-readable name for the BMP CPS code."""
        key = int(cps)
        return BMP_CPS_NAME_MAP.get(key, f"CPS {key}")

    def _parcel_record(self, pid: Union[int, str]) -> pd.Series:
        """Return parcel metadata for a given parcel ID, raising if missing."""
        sub = self.data[DATA_PARCELS]
        match = sub[sub[COL_PID].astype(str) == str(pid)]
        if match.empty:
            raise KeyError(
                f"Selected pid {pid} not found in parcels after clipping. "
                f"Ensure parcel_p PIDs exist in parcels and are within the domain."
            )
        return match.iloc[0]

    def _parcel_up_list(self, pid: Union[int, str]) -> List[str]:
        """Return the ordered list of up-gradient parcel IDs for the given parcel."""
        return list(self.data[DATA_PARCEL_UP_MAP].get(str(pid), []))

    def _parcel_out_oids(self, parcel_idx: int) -> List[str]:
        """Return the outlet IDs associated with a parcel index."""
        return list(self.parcel_out_oids[parcel_idx])

    def _delivery_coeffs(self, pid: Union[int, str], oid: Union[int, str]) -> Dict[str, float]:
        """Get delivery coefficients for a parcel-to-outlet pair.

        Defaults to 1.0 when no explicit delivery ratios are supplied.
        """
        return self.delivery_coeffs.get(
            (str(pid), str(oid)),
            dict(sdr_f_to_s=1.0, sdr_s_to_o=1.0, ndr_f_to_s=1.0, ndr_s_to_o=1.0),
        )

    def _compute_bmp_cost(self, cps: Union[int, str], quantity: float) -> float:
        """Compute the cost for a BMP instance using the configured cost table."""
        return compute_bmp_cost(self.rng, self.data.get("bmp_cost"), cps, quantity, self.logger)

    def _simulate_wetland(
        self,
        parcel_idx: int,
        eff: List[float],
        yields: np.ndarray,
        bmp_rec: Dict[str, Any],
        bmp_outputs: Dict[str, np.ndarray],
    ) -> None:
        """Wrap wetland BMP simulation logic into the scenario engine."""
        simulate_wetland(
            self.rng,
            parcel_idx,
            eff,
            yields,
            bmp_rec,
            bmp_outputs,
            self.parcel_area_ha,
            self.parcel_up_idxs,
            self.parcel_ids,
            self.pollutants,
            logger=self.logger,
        )

    def _simulate_grassed(
        self,
        parcel_idx: int,
        eff: List[float],
        yields: np.ndarray,
        bmp_rec: Dict[str, Any],
        bmp_outputs: Dict[str, np.ndarray],
    ) -> None:
        """Wrap grassed/buffer BMP simulation logic into the scenario engine."""
        simulate_grassed(
            self.rng,
            parcel_idx,
            eff,
            yields,
            bmp_rec,
            bmp_outputs,
            self.parcel_area_ha,
            self.parcel_perim_m,
            self.cfg,
            self.pollutants,
            logger=self.logger,
        )

    def _simulate_infield(
        self,
        parcel_idx: int,
        eff: List[float],
        yields: np.ndarray,
        bmp_rec: Dict[str, Any],
        bmp_outputs: Dict[str, np.ndarray],
    ) -> None:
        """Wrap in-field BMP simulation logic into the scenario engine."""
        simulate_infield(
            parcel_idx,
            eff,
            yields,
            bmp_rec,
            bmp_outputs,
            self.parcel_area_ha,
            self.pollutants,
        )

    def run_all_scenarios(self) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
        """Run all configured scenarios, persist outputs, and return plotting records."""
        outputs_dir = Path(self.cfg.get(CFG_OUTPUTS, "./outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir



        parcels_path = outputs_dir / "parcels.csv"
        bmps_path = outputs_dir / "bmps.csv"
        first_write = True

        scenario_records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
        x_axes: List[str]

        n_parcels = len(self.parcel_ids)
        n_pollutants = len(self.pollutants)

        for sidx in range(self.data[DATA_N_SCENARIOS]):
            self.logger.info(f"Scenario {sidx + 1}/{self.data['n_scenarios']}")
            yields = np.empty((n_parcels, n_pollutants), dtype=float)
            baseline = np.empty_like(yields)
            for parcel_idx in range(n_parcels):
                for pol_idx, pol in enumerate(self.pollutants):
                    y = self._sample_yield(parcel_idx, pol_idx)
                    yields[parcel_idx, pol_idx] = y
                    baseline[parcel_idx, pol_idx] = y

            total_cost = 0.0
            total_bmp = 0
            limit_usd = self.data[DATA_BMP_LIMIT_USD]
            limit_n = self.data[DATA_BMP_LIMIT_N]
            cumul: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

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

            scenario_bmps: List[Dict[str, Any]] = []
            scenario_parcels: List[Dict[str, Any]] = []

            while True:
                if limit_usd is not None and total_cost >= limit_usd:
                    break
                if limit_n is not None and total_bmp >= limit_n:
                    break

                parcel_sel_idx = self._select_parcel_index()
                pid = self.parcel_selection_ids[parcel_sel_idx]
                parcel_idx = self.pid_to_index[pid]
                cps = self._select_bmp_type()

                eff = [self._sample_efficiency(cps, pol_idx) for pol_idx in range(n_pollutants)]

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

                if cps in (656, 657):
                    self._simulate_wetland(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
                    quantity = float(bmp_rec[OUTPUT_WETLAND_AREA])
                elif cps in (412,):
                    self._simulate_grassed(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
                    quantity = float(bmp_rec[OUTPUT_BUFFER_AREA]) if bmp_rec[OUTPUT_BUFFER_AREA] else 0.0
                else:
                    self._simulate_infield(parcel_idx, eff, yields, bmp_rec, bmp_outputs)
                    quantity = float(self.parcel_area_ha[parcel_idx])

                cost_this = self._compute_bmp_cost(cps, quantity)
                total_cost += cost_this
                total_bmp += 1

                bmp_rec[OUTPUT_COST_USD] = cost_this
                for pol_idx, pol in enumerate(self.pollutants):
                    bmp_rec[f"{OUTPUT_TREATED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_TREATED][pol_idx])
                    bmp_rec[f"{OUTPUT_REMOVED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
                scenario_bmps.append(bmp_rec)

                oids = self._parcel_out_oids(parcel_idx)
                for pol_idx, pol in enumerate(self.pollutants):
                    removed_load = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
                    for oid in oids:
                        dr = self._delivery_coeffs(pid, oid)
                        if pol == "TSS":
                            deliver = removed_load * dr[COL_SDR_F_TO_S] * dr[COL_SDR_S_TO_O]
                        else:
                            deliver = removed_load * dr[COL_NDR_F_TO_S] * dr[COL_NDR_S_TO_O]
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
