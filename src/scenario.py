import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from .bmp import (
    compute_bmp_cost,
    sample_efficiency,
    sample_yield,
    simulate_grassed,
    simulate_infield,
    simulate_wetland,
)
from .constants import (
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_OUTPUTS,
    CFG_OUTLET_MEAN,
    CFG_OUTLET_TARGET,
    COL_AREA_HA,
    COL_CPS,
    COL_OID,
    COL_PID,
    COL_POLLUTANT,
    COL_PROBABILITY,
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
    OUTPUT_EFFICIENCY_JSON,
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

    def _sample_efficiency(self, cps: Union[int, str], pollutant: str) -> float:
        """Sample BMP efficiency for a given CPS type and pollutant."""
        return sample_efficiency(self.rng, self.data[DATA_BMP_EFFICIENCY], cps, pollutant, self.logger)

    def _sample_yield(self, pid: Union[int, str], pollutant: str) -> float:
        """Sample baseline pollutant load for a parcel and pollutant."""
        return sample_yield(self.rng, self.data[DATA_POLLUTANT_YIELD], pid, pollutant, self.logger)

    def _select_parcel(self) -> str:
        """Select a parcel ID randomly from parcel probabilities."""
        df = self.data[DATA_PARCEL_P]
        probs = df[COL_PROBABILITY].values
        idx = self.rng.choice(len(df), p=probs)
        return str(df.iloc[idx][COL_PID])

    def _select_bmp_type(self, bmp_probs: pd.DataFrame) -> int:
        """Choose a BMP type code from the probability distribution."""
        probs = bmp_probs[COL_PROBABILITY].values
        idx = self.rng.choice(len(bmp_probs), p=probs)
        return int(bmp_probs.iloc[idx][COL_CPS])

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

    def _parcel_out_oids(self, pid: Union[int, str]) -> List[str]:
        """Return the outlet IDs associated with a parcel."""
        return list(self.data[DATA_PARCEL_OUT_MAP].get(str(pid), []))

    def _delivery_coeffs(self, pid: Union[int, str], oid: Union[int, str]) -> Dict[str, float]:
        """Get delivery coefficients for a parcel-to-outlet pair.

        Defaults to 1.0 when no explicit delivery ratios are supplied.
        """
        dr = self.data.get(DATA_DELIVERY_RATIOS)
        if dr is None:
            return dict(sdr_f_to_s=1.0, sdr_s_to_o=1.0, ndr_f_to_s=1.0, ndr_s_to_o=1.0)
        sub = dr[(dr[COL_PID].astype(str) == str(pid)) & (dr[COL_OID].astype(str) == str(oid))]
        if not len(sub):
            return dict(sdr_f_to_s=1.0, sdr_s_to_o=1.0, ndr_f_to_s=1.0, ndr_s_to_o=1.0)
        r = sub.iloc[0]
        return dict(
            sdr_f_to_s=float(r[COL_SDR_F_TO_S]),
            sdr_s_to_o=float(r[COL_SDR_S_TO_O]),
            ndr_f_to_s=float(r[COL_NDR_F_TO_S]),
            ndr_s_to_o=float(r[COL_NDR_S_TO_O]),
        )

    def _compute_bmp_cost(self, cps: Union[int, str], quantity: float) -> float:
        """Compute the cost for a BMP instance using the configured cost table."""
        return compute_bmp_cost(self.rng, self.data.get("bmp_cost"), cps, quantity, self.logger)

    def _simulate_wetland(
        self,
        pid: Union[int, str],
        eff: Dict[str, float],
        yields_map: Dict[Tuple[str, str], float],
        bmp_rec: Dict[str, Any],
        bmp_outputs: Dict[str, Dict[str, float]],
    ) -> None:
        """Wrap wetland BMP simulation logic into the scenario engine."""
        simulate_wetland(
            self.rng,
            pid,
            eff,
            yields_map,
            bmp_rec,
            bmp_outputs,
            self._parcel_record,
            self._parcel_up_list,
            self.data[DATA_POLLUTANTS],
        )

    def _simulate_grassed(
        self,
        pid: Union[int, str],
        eff: Dict[str, float],
        yields_map: Dict[Tuple[str, str], float],
        bmp_rec: Dict[str, Any],
        bmp_outputs: Dict[str, Dict[str, float]],
    ) -> None:
        """Wrap grassed/buffer BMP simulation logic into the scenario engine."""
        simulate_grassed(
            self.rng,
            pid,
            eff,
            yields_map,
            bmp_rec,
            bmp_outputs,
            self._parcel_record,
            self.cfg,
            self.data[DATA_POLLUTANTS],
        )

    def _simulate_infield(
        self,
        pid: Union[int, str],
        eff: Dict[str, float],
        yields_map: Dict[Tuple[str, str], float],
        bmp_rec: Dict[str, Any],
        bmp_outputs: Dict[str, Dict[str, float]],
    ) -> None:
        """Wrap in-field BMP simulation logic into the scenario engine."""
        simulate_infield(
            pid,
            eff,
            yields_map,
            bmp_rec,
            bmp_outputs,
            self._parcel_record,
            self.data[DATA_POLLUTANTS],
        )

    def run_all_scenarios(self) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
        """Run all configured scenarios, persist outputs, and return plotting records."""
        outputs_dir = Path(self.cfg.get(CFG_OUTPUTS, "./outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir

        parcels = self.data[DATA_PARCELS]
        pollutants = self.data[DATA_POLLUTANTS]

        bmp_probs = self._get_bmp_selection_probs()
        scenario_records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)

        for sidx in range(self.data[DATA_N_SCENARIOS]):
            self.logger.info(f"Scenario {sidx + 1}/{self.data['n_scenarios']}")
            yields_map: Dict[Tuple[str, str], float] = {}
            baseline_map: Dict[Tuple[str, str], float] = {}
            for _, r in parcels.iterrows():
                pid = str(r[COL_PID])
                for pol in pollutants:
                    y = self._sample_yield(pid, pol)
                    yields_map[(pid, pol)] = y
                    baseline_map[(pid, pol)] = y

            total_cost = 0.0
            total_bmp = 0
            limit_usd = self.data[DATA_BMP_LIMIT_USD]
            limit_n = self.data[DATA_BMP_LIMIT_N]
            cumul: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

            x_axes = []
            if self.cfg.get(CFG_BMP_COST):
                x_axes.append(XAXIS_COST)
            x_axes.append(XAXIS_COUNT)

            y_axes = [YAXIS_TOTAL]
            if self.cfg.get(CFG_OUTLET_TARGET):
                y_axes.append(YAXIS_TARGET)
            if self.cfg.get(CFG_OUTLET_MEAN):
                y_axes.append(YAXIS_MEAN)

            scenario_bmps: List[Dict[str, Any]] = []
            scenario_parcels: Dict[str, Dict[str, Any]] = {}

            while True:
                if limit_usd is not None:
                    if total_cost >= limit_usd:
                        break
                else:
                    if total_bmp >= limit_n:
                        break

                pid = self._select_parcel()
                cps = self._select_bmp_type(bmp_probs)
                row = self._parcel_record(pid)

                eff: Dict[str, float] = {}
                for pol in pollutants:
                    eff[pol] = self._sample_efficiency(cps, pol)

                bmp_rec: Dict[str, Any] = dict(
                    scenario=sidx + 1,
                    cps=cps,
                    pid=str(pid),
                    **{
                        OUTPUT_IMPACTED_PIDS: "",
                        OUTPUT_EFFICIENCY_JSON: json.dumps(eff, separators=(",", ":")),
                        OUTPUT_LINEAR_LENGTH: None,
                        OUTPUT_BUFFER_AREA: None,
                        OUTPUT_PORTION_TREATED: None,
                        OUTPUT_WETLAND_AREA: None,
                        OUTPUT_CATCHMENT_RATIO: None,
                    },
                )
                bmp_outputs: Dict[str, Dict[str, float]] = dict(
                    **{OUTPUT_TREATED: defaultdict(float), OUTPUT_REMOVED: defaultdict(float)}
                )

                if cps in (656, 657):
                    self._simulate_wetland(pid, eff, yields_map, bmp_rec, bmp_outputs)
                    quantity = float(bmp_rec[OUTPUT_WETLAND_AREA])
                elif cps in (412,):
                    self._simulate_grassed(pid, eff, yields_map, bmp_rec, bmp_outputs)
                    quantity = float(bmp_rec[OUTPUT_BUFFER_AREA]) if bmp_rec[OUTPUT_BUFFER_AREA] else 0.0
                else:
                    self._simulate_infield(pid, eff, yields_map, bmp_rec, bmp_outputs)
                    quantity = float(row[COL_AREA_HA])

                cost_this = self._compute_bmp_cost(cps, quantity)
                total_cost += cost_this
                total_bmp += 1

                bmp_rec[OUTPUT_COST_USD] = cost_this
                for pol in pollutants:
                    bmp_rec[f"{OUTPUT_TREATED_PREFIX}{pol}"] = bmp_outputs[OUTPUT_TREATED][pol]
                    bmp_rec[f"{OUTPUT_REMOVED_PREFIX}{pol}"] = bmp_outputs[OUTPUT_REMOVED][pol]
                scenario_bmps.append(bmp_rec)

                oids = self._parcel_out_oids(pid)
                for pol in pollutants:
                    removed_load = bmp_outputs[OUTPUT_REMOVED][pol]
                    for oid in oids:
                        dr = self._delivery_coeffs(pid, oid)
                        if pol == "TSS":
                            deliver = removed_load * dr[COL_SDR_F_TO_S] * dr[COL_SDR_S_TO_O]
                        else:
                            deliver = removed_load * dr[COL_NDR_F_TO_S] * dr[COL_NDR_S_TO_O]
                        cumul[pol][oid] += deliver

                for pol in pollutants:
                    for oid in self.data[DATA_OUTLET_LOC][COL_OID].astype(str).tolist():
                        for xax in x_axes:
                            for yax in y_axes:
                                xval = total_bmp if xax == "count" else total_cost
                                if yax == YAXIS_TOTAL:
                                    yval = cumul[pol][oid]
                                elif yax == YAXIS_TARGET:
                                    tgt = 0.0
                                    if self.data.get(CFG_OUTLET_TARGET) is not None:
                                        sub = self.data[DATA_OUTLET_TARGET]
                                        m = sub[(sub[COL_OID].astype(str) == str(oid)) & (sub[COL_POLLUTANT] == pol)]
                                        if len(m):
                                            tgt = float(m.iloc[0][COL_TARGET])
                                    yval = (cumul[pol][oid] / tgt * 100.0) if tgt > 0 else 0.0
                                elif yax == YAXIS_MEAN:
                                    mu = 0.0
                                    if self.data.get(CFG_OUTLET_MEAN) is not None:
                                        sub = self.data[DATA_OUTLET_MEAN]
                                        m = sub[(sub[COL_OID].astype(str) == str(oid)) & (sub[COL_POLLUTANT] == pol)]
                                        if len(m):
                                            mu = float(m.iloc[0][COL_MEAN])
                                    yval = (cumul[pol][oid] / mu * 100.0) if mu > 0 else 0.0
                                scenario_records[(pol, oid, xax, yax)].append((sidx + 1, xval, yval))

            for _, r in parcels.iterrows():
                pid_i = str(r[COL_PID])
                rec: Dict[str, Any] = dict(scenario=sidx + 1, pid=pid_i)
                for pol in pollutants:
                    rec[f"baseline_{pol}"] = baseline_map[(pid_i, pol)]
                    rec[f"final_{pol}"] = yields_map[(pid_i, pol)]
                scenario_parcels[pid_i] = rec

            pd.DataFrame(scenario_bmps).to_csv(self.outputs_dir / f"scenario_{sidx + 1:03d}_bmps.csv", index=False)
            pd.DataFrame(list(scenario_parcels.values())).to_csv(self.outputs_dir / f"scenario_{sidx + 1:03d}_parcels.csv", index=False)

        return scenario_records
