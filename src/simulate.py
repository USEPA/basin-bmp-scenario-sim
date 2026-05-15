import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from .sampling import sample_from_stats
from .selection import estimate_costs_for_probabilities
from .costs import compute_bmp_cost_usd

FT_TO_M = 0.3048  # meters per foot


class Simulator:
    def __init__(self, cfg, data, logger):
        self.cfg = cfg
        self.data = data
        self.logger = logger
        seed = data.get("random_seed", None)
        self.rng = np.random.default_rng(seed)
        self.outputs_dir = None

    def _get_bmp_selection_probs(self):
        # If bmp_sel path is provided in cfg, load probabilities from it.
        # Otherwise, derive inverse-cost probabilities from bmp_cost.
        import pandas as pd

        bmp_sel_path = self.cfg.get("bmp_sel")
        if bmp_sel_path:
            df = pd.read_csv(bmp_sel_path)
            df.columns = [c.lower() for c in df.columns]
            df = df[df["cps"].astype(int).isin(self.data["cps"])].copy()
            if "probability" not in df.columns and "pr" in df.columns:
                df["probability"] = df["pr"]
            elif "probability" not in df.columns and "p" in df.columns:
                df["probability"] = df["p"]
            s = df["probability"].sum()
            if s <= 0:
                raise ValueError("bmp_sel probabilities sum to zero or negative")
            df["probability"] = df["probability"] / s
            return df[["cps", "probability"]]
        else:
            if self.data["bmp_cost"] is None:
                probs = np.full(len(self.data["cps"]), 1.0 / len(self.data["cps"]))
                return pd.DataFrame({"cps": self.data["cps"], "probability": probs})
            else:
                df = estimate_costs_for_probabilities(
                    self.rng,
                    self.data["bmp_cost"],
                    self.data["cps"],
                    self.data["avg_area_ha"],
                    self.data["avg_perim_m"],
                    overrides={},
                )
                return df

    def _sample_efficiency(self, cps, pollutant):
        sub = self.data["bmp_eff"]
        row = sub[(sub["cps"].astype(int) == int(cps)) & (sub["pollutant"] == pollutant)].iloc[0]
        stats = {k: row[k] for k in row.index if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())}
        return sample_from_stats(self.rng, stats, kind="efficiency", verbose_logger=self.logger)

    def _sample_yield(self, pid, pollutant):
        sub = self.data["pollutant_yield"]
        row = sub[(sub["pid"].astype(str) == str(pid)) & (sub["pollutant"] == pollutant)].iloc[0]
        stats = {k: row[k] for k in row.index if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())}
        return sample_from_stats(self.rng, stats, kind="yield", verbose_logger=self.logger)

    def _select_parcel(self):
        df = self.data["parcel_p"]
        probs = df["probability"].values
        idx = self.rng.choice(len(df), p=probs)
        return str(df.iloc[idx]["pid"])

    def _select_bmp_type(self, bmp_probs):
        probs = bmp_probs["probability"].values
        idx = self.rng.choice(len(bmp_probs), p=probs)
        return int(bmp_probs.iloc[idx]["cps"])

    def _parcel_record(self, pid):
        # Optional guardrail: give a clear error if a PID is missing
        sub = self.data["parcels"]
        match = sub[sub["pid"].astype(str) == str(pid)]
        if match.empty:
            raise KeyError(
                f"Selected pid {pid} not found in parcels after clipping. "
                f"Ensure parcel_p PIDs exist in parcels and are within the domain."
            )
        return match.iloc[0]

    def _parcel_up_list(self, pid):
        return list(self.data["parcel_up_map"].get(str(pid), []))

    def _parcel_out_oids(self, pid):
        return list(self.data["parcel_out_map"].get(str(pid), []))

    def _delivery_coeffs(self, pid, oid):
        dr = self.data.get("delivery_ratios")
        if dr is None:
            # defaults to 1.0 if delivery ratios are not provided
            return dict(sdr_f_to_s=1.0, sdr_s_to_o=1.0, ndr_f_to_s=1.0, ndr_s_to_o=1.0)
        sub = dr[(dr["pid"].astype(str) == str(pid)) & (dr["oid"].astype(str) == str(oid))]
        if not len(sub):
            return dict(sdr_f_to_s=1.0, sdr_s_to_o=1.0, ndr_f_to_s=1.0, ndr_s_to_o=1.0)
        r = sub.iloc[0]
        return dict(
            sdr_f_to_s=float(r["sdr_f_to_s"]),
            sdr_s_to_o=float(r["sdr_s_to_o"]),
            ndr_f_to_s=float(r["ndr_f_to_s"]),
            ndr_s_to_o=float(r["ndr_s_to_o"]),
        )

    def _compute_bmp_cost(self, cps, quantity):
        if self.data["bmp_cost"] is None:
            return 0.0
        sub = self.data["bmp_cost"][self.data["bmp_cost"]["cps"].astype(int) == int(cps)]
        if sub.empty:
            return 0.0
        unit_row = sub.iloc[0]
        return compute_bmp_cost_usd(self.rng, cps, unit_row, quantity, self.logger)

    def _simulate_wetland(self, pid, eff, yields_map, bmp_rec, bmp_outputs):
        row = self._parcel_record(pid)
        area_field_ha = float(row["area_ha"])

        # Wetland area (ha): min/max/mean with truncation
        wet_area_stats = {"min": 0.1, "max": 10.0, "mean": 0.4}
        wet_area = sample_from_stats(self.rng, wet_area_stats, kind=None, verbose_logger=self.logger)
        wet_area = min(wet_area, area_field_ha)

        # Catchment-to-wetland ratio
        ratio_stats = {"min": 2.0, "max": 100.0, "mean": 5.0}
        cat_ratio = sample_from_stats(self.rng, ratio_stats, kind=None, verbose_logger=self.logger)
        catchment_area_ha = cat_ratio * wet_area
        impacted_area_ha = wet_area + catchment_area_ha

        # Determine impacted parcels (this parcel and possibly upgradient parcels)
        up_list = self._parcel_up_list(pid)
        impacted_pids = [str(pid)]
        total_available_ha = area_field_ha
        if impacted_area_ha > area_field_ha and len(up_list):
            for up_pid in up_list:
                r = self._parcel_record(up_pid)
                impacted_pids.append(str(up_pid))
                total_available_ha += float(r["area_ha"])
                if total_available_ha >= impacted_area_ha:
                    break

        # If still not enough area, trim impacted_area_ha and adjust ratio
        if impacted_area_ha > total_available_ha:
            impacted_area_ha = total_available_ha
            cat_ratio = max(0.0, (impacted_area_ha - wet_area) / max(wet_area, 1e-9))

        bmp_rec["wetland_area_ha"] = wet_area
        bmp_rec["catchment_to_wetland_ratio"] = cat_ratio
        bmp_rec["impacted_pids"] = ",".join(impacted_pids if len(impacted_pids) > 1 else [])

        # Apply reductions parcel-by-parcel until we satisfy impacted_area_ha
        remaining = impacted_area_ha
        for p in impacted_pids:
            r = self._parcel_record(p)
            A = float(r["area_ha"])
            if remaining <= 0:
                frac = 0.0
            elif remaining < A:
                frac = remaining / A
            else:
                frac = 1.0

            for pollutant in self.data["pollutants"]:
                y = yields_map[(p, pollutant)]
                reduction = y * (A * frac) * eff[pollutant]
                bmp_outputs["treated"][pollutant] += y * (A * frac)
                bmp_outputs["removed"][pollutant] += reduction
                y_new = y - reduction / A
                yields_map[(p, pollutant)] = max(0.0, y_new)

            remaining -= A

    def _simulate_grassed(self, pid, eff, yields_map, bmp_rec, bmp_outputs):
        row = self._parcel_record(pid)
        perim_m = float(row["perim_m"])
        # Portion of field treated (fraction of perimeter length)
        frac_stats = {"min": 0.1, "max": 0.5, "mean": 0.25}
        frac = sample_from_stats(self.rng, frac_stats, kind=None, verbose_logger=self.logger)
        length_m = perim_m * frac
        depth_m = float(self.cfg.get("buffer_depth_ft", 35.0)) * FT_TO_M
        area_ha = (length_m * depth_m) / 10000.0
        bmp_rec["linear_length_m"] = length_m
        bmp_rec["buffer_area_ha"] = area_ha
        bmp_rec["portion_treated"] = frac

        A = float(row["area_ha"])
        for pollutant in self.data["pollutants"]:
            y = yields_map[(str(pid), pollutant)]
            reduction = y * (A * frac) * eff[pollutant]
            bmp_outputs["treated"][pollutant] += y * (A * frac)
            bmp_outputs["removed"][pollutant] += reduction
            y_new = y - reduction / A
            yields_map[(str(pid), pollutant)] = max(0.0, y_new)

    def _simulate_infield(self, pid, eff, yields_map, bmp_rec, bmp_outputs):
        row = self._parcel_record(pid)
        A = float(row["area_ha"])
        for pollutant in self.data["pollutants"]:
            y = yields_map[(str(pid), pollutant)]
            reduction = y * A * eff[pollutant]
            bmp_outputs["treated"][pollutant] += y * A
            bmp_outputs["removed"][pollutant] += reduction
            y_new = y - reduction / A
            yields_map[(str(pid), pollutant)] = max(0.0, y_new)

    def run_all_scenarios(self):
        outputs_dir = Path(self.cfg.get("outputs", "./outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir

        parcels = self.data["parcels"]
        pollutants = self.data["pollutants"]

        # Precompute selection probabilities
        bmp_probs = self._get_bmp_selection_probs()

        # Collect points for plotting
        scenario_records = defaultdict(list)

        for sidx in range(self.data["n_scenarios"]):
            self.logger.info(f"Scenario {sidx + 1}/{self.data['n_scenarios']}")
            # Draw baseline yields once per parcel per pollutant for this scenario
            yields_map = {}
            baseline_map = {}
            for _, r in parcels.iterrows():
                pid = str(r["pid"])
                for pol in pollutants:
                    y = self._sample_yield(pid, pol)
                    yields_map[(pid, pol)] = y
                    baseline_map[(pid, pol)] = y

            total_cost = 0.0
            total_bmp = 0
            limit_usd = self.data["bmp_limit_usd"]
            limit_n = self.data["bmp_limit_n"]

            # Accumulate delivered reductions to outlets
            cumul = defaultdict(lambda: defaultdict(float))

            x_axes = []
            if self.cfg.get("bmp_cost"):
                x_axes.append("cost")
            x_axes.append("count")

            y_axes = ["total"]
            if self.cfg.get("outlet_target"):
                y_axes.append("target")
            if self.cfg.get("outlet_mean"):
                y_axes.append("mean")

            scenario_bmps = []
            scenario_parcels = {}

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

                # Efficiency per pollutant
                eff = {}
                for pol in pollutants:
                    eff[pol] = self._sample_efficiency(cps, pol)

                bmp_rec = dict(
                    scenario=sidx + 1,
                    cps=cps,
                    pid=str(pid),
                    impacted_pids="",
                    efficiency_json=json.dumps(eff, separators=(",", ":")),
                    linear_length_m=None,
                    buffer_area_ha=None,
                    portion_treated=None,
                    wetland_area_ha=None,
                    catchment_to_wetland_ratio=None,
                )
                bmp_outputs = dict(treated=defaultdict(float), removed=defaultdict(float))

                # Determine BMP type and simulate
                if cps in (656, 657):  # wetlands
                    self._simulate_wetland(pid, eff, yields_map, bmp_rec, bmp_outputs)
                    quantity = float(bmp_rec["wetland_area_ha"])  # for cost (USD/ha)
                elif cps in (412,):  # grassed waterway / buffer-like
                    self._simulate_grassed(pid, eff, yields_map, bmp_rec, bmp_outputs)
                    quantity = float(bmp_rec["buffer_area_ha"]) if bmp_rec["buffer_area_ha"] else 0.0
                else:  # in-field
                    self._simulate_infield(pid, eff, yields_map, bmp_rec, bmp_outputs)
                    quantity = float(row["area_ha"])

                cost_this = self._compute_bmp_cost(cps, quantity)
                total_cost += cost_this
                total_bmp += 1

                bmp_rec["cost_usd"] = cost_this
                for pol in pollutants:
                    bmp_rec[f"treated_{pol}"] = bmp_outputs["treated"][pol]
                    bmp_rec[f"removed_{pol}"] = bmp_outputs["removed"][pol]
                scenario_bmps.append(bmp_rec)

                # Update cumulative delivered reductions for plotting
                oids = self._parcel_out_oids(pid)
                for pol in pollutants:
                    removed_load = bmp_outputs["removed"][pol]
                    for oid in oids:
                        dr = self._delivery_coeffs(pid, oid)
                        if pol.lower().startswith("sed"):
                            deliver = removed_load * dr["sdr_f_to_s"] * dr["sdr_s_to_o"]
                        else:
                            deliver = removed_load * dr["ndr_f_to_s"] * dr["ndr_s_to_o"]
                        cumul[pol][oid] += deliver

                # Record point for each plot configuration
                for pol in pollutants:
                    for oid in self.data["outlet_loc"]["oid"].astype(str).tolist():
                        for xax in x_axes:
                            for yax in y_axes:
                                xval = total_bmp if xax == "count" else total_cost
                                if yax == "total":
                                    yval = cumul[pol][oid]
                                elif yax == "target":
                                    tgt = 0.0
                                    if self.data.get("outlet_target") is not None:
                                        sub = self.data["outlet_target"]
                                        m = sub[(sub["oid"].astype(str) == str(oid)) & (sub["pollutant"] == pol)]
                                        if len(m):
                                            tgt = float(m.iloc[0]["target"])
                                    yval = (cumul[pol][oid] / tgt * 100.0) if tgt > 0 else 0.0
                                elif yax == "mean":
                                    mu = 0.0
                                    if self.data.get("outlet_mean") is not None:
                                        sub = self.data["outlet_mean"]
                                        m = sub[(sub["oid"].astype(str) == str(oid)) & (sub["pollutant"] == pol)]
                                        if len(m):
                                            mu = float(m.iloc[0]["mean"])
                                    yval = (cumul[pol][oid] / mu * 100.0) if mu > 0 else 0.0
                                scenario_records[(pol, oid, xax, yax)].append((sidx + 1, xval, yval))

            # After BMP loop, write parcel-level outputs (baseline and final)
            for _, r in parcels.iterrows():
                pid_i = str(r["pid"])
                rec = dict(scenario=sidx + 1, pid=pid_i)
                for pol in pollutants:
                    rec[f"baseline_{pol}"] = baseline_map[(pid_i, pol)]
                    rec[f"final_{pol}"] = yields_map[(pid_i, pol)]
                scenario_parcels[pid_i] = rec

            # Persist CSVs per scenario
            pd.DataFrame(scenario_bmps).to_csv(self.outputs_dir / f"scenario_{sidx + 1:03d}_bmps.csv", index=False)
            pd.DataFrame(list(scenario_parcels.values())).to_csv(self.outputs_dir / f"scenario_{sidx + 1:03d}_parcels.csv", index=False)

        return scenario_records