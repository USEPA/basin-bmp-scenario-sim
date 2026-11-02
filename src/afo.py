"""Optional PLET-style animal feeding operations and spatial manure exposure.

This is deliberately a concentration/area model, *not* a manure mass balance.
See docs/animal_feeding_operations.md for inputs, assumptions, and limitations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import sqlite3

import numpy as np
import pandas as pd

from .input_distributions import stats_from_row, statistic_columns
from .plet_rusle import (
    plet_annual_surface_runoff_in,
    INCH_OVER_HA_TO_LITERS,
    normalize_plet_land_cover,
)

_FEEDLOT_CN = (91.0, 92.0, 93.0, 94.0)
_FEEDLOT_MAX_CONC = {"TN": 1500.0, "TP": 300.0, "BOD": 2000.0}
_STATS = {"value", "mean", "sd", "min", "max"}


def _read_table(conn: sqlite3.Connection, name: str, *, optional: bool = False) -> pd.DataFrame:
    names = {str(x[0]) for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if name not in names:
        if optional:
            return pd.DataFrame()
        raise ValueError(f"AFO GeoPackage is missing table {name!r}")
    frame = pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    return frame


def _need_columns(df: pd.DataFrame, name: str, required: Sequence[str]) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _integer_pid(value: Any, name: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{name} must be an integer parcel ID")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer parcel ID: {value!r}") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be a finite integer parcel ID: {value!r}")
    return str(int(number))


def _finite(value: Any, name: str, low: float = 0., high: float = float("inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or not low <= out <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return out


def _sample_stats(ctx: Any, row: Mapping[str, Any], *, prefix: str = "", kind: str = "nonnegative") -> float:
    if prefix:
        stats = {str(key)[len(prefix):]: value for key, value in row.items()
                 if str(key).startswith(prefix) and not pd.isna(value)}
    else:
        stats = stats_from_row(row, {"land_cover", "pollutant", "sample_group", "units"})
    if not stats:
        raise ValueError(f"Missing fixed value or distribution for {prefix or 'concentration'}")
    return float(ctx._sample_from_stats(stats, kind=kind))


def load_afo_inputs(path: str | Path, parcels: pd.DataFrame, pollutants: Sequence[str]) -> dict[str, Any]:
    """Load and fully validate optional AFO tables before worker processes start."""
    package = Path(path)
    if not package.is_file():
        raise FileNotFoundError(f"AFO GeoPackage not found: {package}")
    with sqlite3.connect(package) as conn:
        operations = _read_table(conn, "afo_operations")
        animals = _read_table(conn, "afo_animals")
        fields = _read_table(conn, "afo_manure_fields", optional=True)
        increments = _read_table(conn, "afo_manure_concentration_increments")
    _need_columns(operations, "afo_operations", ["afo_id", "pid", "feedlot_area_ha", "paved_pct", "allocation", "manure_months"])
    _need_columns(animals, "afo_animals", ["afo_id", "species", "count"])
    _need_columns(increments, "afo_manure_concentration_increments", ["land_cover", "pollutant"])
    if not fields.empty:
        _need_columns(fields, "afo_manure_fields", ["afo_id", "pid"])
    if operations.empty:
        raise ValueError("AFO package requires at least one operation")
    operations["afo_id"] = operations["afo_id"].astype(str).str.strip()
    operations["pid"] = operations["pid"].astype(object)
    if operations["afo_id"].eq("").any() or operations["afo_id"].duplicated().any():
        raise ValueError("afo_operations.afo_id must contain distinct nonblank IDs")
    op_ids = set(operations["afo_id"])
    pid_set = set(parcels["pid"].astype(str))
    area_by_pid = dict(zip(parcels["pid"].astype(str), parcels["area_ha"].astype(float)))
    for k, row in operations.iterrows():
        op_id = row["afo_id"]
        pid = _integer_pid(row["pid"], f"afo_operations[{op_id}].pid")
        if pid not in pid_set:
            raise ValueError(f"Operation {op_id} refers to unknown host pid {pid}")
        operations.at[k, "pid"] = pid
        area = _finite(row["feedlot_area_ha"], f"{op_id}.feedlot_area_ha", low=1e-12)
        if area > area_by_pid[pid] + 1e-8:
            raise ValueError(f"Operation {op_id} feedlot area exceeds host parcel area")
        _finite(row["paved_pct"], f"{op_id}.paved_pct", 0., 100.)
        _finite(row["manure_months"], f"{op_id}.manure_months", 0., 12.)
        mode = str(row["allocation"]).strip().lower()
        if mode not in {"explicit", "sampled"}:
            raise ValueError(f"Operation {op_id} allocation must be explicit or sampled")
        operations.at[k, "allocation"] = mode
        if mode == "sampled":
            selection = str(row.get("selection", "nearest") or "nearest").strip().lower()
            if selection not in {"nearest", "random"}:
                raise ValueError(f"Operation {op_id} selection must be nearest or random")
            operations.at[k, "selection"] = selection
            if not any(str(c).startswith("n_fields_") and pd.notna(row[c]) for c in operations.columns):
                raise ValueError(f"Operation {op_id} requires n_fields_value or n_fields_* distribution")
            for col in operations.columns:
                if str(col).startswith("n_fields_") and pd.notna(row[col]):
                    _finite(row[col], f"{op_id}.{col}")
    host_area = {}
    for _, row in operations.iterrows():
        host_area[row["pid"]] = host_area.get(row["pid"], 0.) + float(row["feedlot_area_ha"])
    for pid, total in host_area.items():
        if total > area_by_pid[pid] + 1e-8:
            raise ValueError(f"Combined feedlot footprints exceed parcel area for pid {pid}")
    for op_id in op_ids:
        if not animals["afo_id"].astype(str).str.strip().eq(op_id).any():
            raise ValueError(f"Operation {op_id} has no animal inventory")
    for _, row in animals.iterrows():
        op_id = str(row["afo_id"]).strip()
        if op_id not in op_ids:
            raise ValueError(f"afo_animals references unknown AFO {op_id}")
        if not str(row["species"]).strip():
            raise ValueError("afo_animals.species must not be blank")
        _finite(row["count"], f"{op_id}.count")
        for pol in set(p.upper() for p in pollutants).intersection(_FEEDLOT_MAX_CONC):
            factor = f"factor_{pol.lower()}"
            if factor not in animals or pd.isna(row[factor]):
                raise ValueError(f"afo_animals requires {factor} for {op_id}; PLET equivalent animal factor")
            _finite(row[factor], f"{op_id}.{factor}")
    if fields.empty and "afo_id" not in fields:
        fields = pd.DataFrame(columns=["afo_id", "pid"])
    fields["afo_id"] = fields["afo_id"].astype(str).str.strip()
    fields["pid"] = fields["pid"].astype(object)
    if fields.duplicated(["afo_id", "pid"]).any():
        raise ValueError("Duplicate operation-to-receiving-parcel relation")
    for k, row in fields.iterrows():
        op_id = row["afo_id"]
        pid = _integer_pid(row["pid"], "afo_manure_fields.pid")
        if op_id not in op_ids or pid not in pid_set:
            raise ValueError(f"Invalid manure field reference {op_id}/{pid}")
        fields.at[k, "pid"] = pid
    if fields["afo_id"].isin(set(operations.loc[operations.allocation == "sampled", "afo_id"])).any():
        raise ValueError("Sampled operations must not also specify explicit manure fields")
    for op_id in operations.loc[operations.allocation == "explicit", "afo_id"]:
        if not fields["afo_id"].eq(op_id).any():
            raise ValueError(f"Explicit operation {op_id} has no manure-receiving fields")
    increments["land_cover"] = increments["land_cover"].map(normalize_plet_land_cover)
    increments["pollutant"] = increments["pollutant"].astype(str).str.upper().str.strip()
    if increments.duplicated(["land_cover", "pollutant"]).any():
        raise ValueError("Duplicate manure concentration increments by land_cover/pollutant")
    if not set(increments["land_cover"]).issubset({"cropland", "pastureland"}):
        raise ValueError("Manure concentration increments apply only to cropland/pastureland")
    if not set(increments["pollutant"]).issubset(set(map(str.upper, pollutants))):
        raise ValueError("Manure concentration increments reference unknown pollutants")
    if not any(col in increments.columns for col in statistic_columns(increments.columns)):
        raise ValueError("Manure concentration increments require value or distribution statistics")
    for _, row in increments.iterrows():
        stats = stats_from_row(row, {"land_cover", "pollutant", "sample_group", "units"})
        if not stats:
            raise ValueError("Each manure concentration increment needs a value or distribution")
        for name, value in stats.items():
            _finite(value, f"afo_manure_concentration_increments.{name}")
    # Treat the field's whole current PLET land cover as eligible, except the host
    # operation footprint. Mixed-use parcel subareas are not represented in PLET.
    return {"operations": operations.to_dict("records"), "animals": animals.to_dict("records"),
            "fields": fields.to_dict("records"), "manure_concentration_increments": increments.to_dict("records"),
            "geometry": {str(r.pid): r.geometry for r in parcels.itertuples()},
            "area_ha": area_by_pid}


def _feedlot_runoff_kg(op: Mapping[str, Any], animals: Sequence[Mapping[str, Any]],
                       parameters: Mapping[str, Any], pollutants: Sequence[str]) -> dict[str, float]:
    """PLET feedlot CN (91–94), EAU density, manure pack, concentration and load."""
    paved = float(op["paved_pct"])
    cn = _FEEDLOT_CN[min(int(paved // 25), 3)]
    forcing = dict(parameters)
    forcing["cn"] = cn
    # PLET's feedlot uses same event rainfall/day forcing as other land uses.
    _, _, _, annual_runoff_in = plet_annual_surface_runoff_in(forcing)
    area_ha = float(op["feedlot_area_ha"])
    area_ac = area_ha / 0.40468564224
    outputs = {}
    for pollutant in pollutants:
        pol = str(pollutant).upper()
        if pol not in _FEEDLOT_MAX_CONC:
            continue
        eau = sum(float(animal["count"]) * float(animal[f"factor_{pol.lower()}"])
                  for animal in animals if str(animal["afo_id"]) == str(op["afo_id"]))
        pack = min(eau / area_ac, 100.) / 100.  # EAU per acre, saturated at 100
        concentration = pack * _FEEDLOT_MAX_CONC[pol]
        # mg/L * L/ha * ha -> kg/yr, using same unit handling as PLET/RUSLE.
        outputs[pol] = annual_runoff_in * INCH_OVER_HA_TO_LITERS * area_ha * concentration / 1_000_000.
    return outputs


def apply_afo_baseline(ctx: Any, state: Any) -> list[dict[str, Any]]:
    """Apply operation runoff and manure-exposure to sampled PLET pathway loads.

    Called only for enabled AFO runs, once per worker before BMPs. The sampled
    manure assignment is shared by that worker's baseline and BMP trajectory.
    """
    inputs = getattr(ctx, "afo_inputs", None)
    if not inputs:
        return []
    pid_to_index = {pid: k for k, pid in enumerate(state.parcel_ids)}
    operations = inputs["operations"]
    animals = inputs["animals"]
    fields = inputs["fields"]
    areas = inputs["area_ha"]
    geometry = inputs["geometry"]
    # The sampled parcel PLET classification decides eligibility, not a static
    # shapefile field. Purely urban/forest parcels cannot receive field manure.
    agricultural = {pid for pid, idx in pid_to_index.items()
                    if state.parameters[idx]["land_cover"] in {"cropland", "pastureland"}}
    feedlot_area = {pid: 0. for pid in pid_to_index}
    for op in operations:
        feedlot_area[str(op["pid"])] += float(op["feedlot_area_ha"])
    eligible_area = {pid: areas[pid] - feedlot_area[pid] for pid in agricultural
                     if areas[pid] - feedlot_area[pid] > 1e-10}
    manure_months: dict[str, float] = {}
    assignments: list[dict[str, Any]] = []
    # Treating manure application extent rather than mass is intentional: PLET's
    # manured runoff concentration is not scaled to tonnage per acre.
    for op in sorted(operations, key=lambda r: str(r["afo_id"])):
        op_id, host = str(op["afo_id"]), str(op["pid"])
        if op["allocation"] == "explicit":
            chosen = [str(r["pid"]) for r in fields if str(r["afo_id"]) == op_id]
            if len(chosen) != len(set(chosen)):
                raise ValueError(f"Duplicate receiving parcel in {op_id}")
        else:
            candidates = sorted(eligible_area, key=lambda p: int(p))
            n_draw = _sample_stats(ctx, op, prefix="n_fields_", kind="nonnegative")
            if not np.isfinite(n_draw) or n_draw < 1:
                raise ValueError(f"{op_id}: sampled n_fields must be at least 1")
            n_fields = int(np.floor(n_draw + 0.5))
            if n_fields > len(candidates):
                raise ValueError(f"{op_id}: sampled {n_fields} fields but only {len(candidates)} eligible")
            chosen = [host] if host in eligible_area else []
            candidates = [p for p in candidates if p != host]
            if op.get("selection", "nearest") == "random":
                candidate_order = list(ctx.rng.permutation(candidates))
            else:
                candidate_order = sorted(candidates, key=lambda p: (geometry[host].distance(geometry[p]), int(p)))
            chosen.extend(candidate_order[:n_fields - len(chosen)])
        for pid in chosen:
            if pid not in eligible_area:
                raise ValueError(f"{op_id}: manure receiving pid {pid} lacks eligible agricultural area")
            manure_months[pid] = max(manure_months.get(pid, 0.), float(op["manure_months"]))
            assignments.append({"afo_id": op_id, "host_pid": host, "receiving_pid": pid,
                                "manure_months": float(op["manure_months"]),
                                "eligible_area_ha": float(eligible_area[pid]),
                                "allocation": op["allocation"]})
    # An increment is a standalone nonnegative concentration added to each
    # parcel's sampled non-manured concentration. It is deliberately NOT an
    # absolute manured concentration, and is drawn independently of the base.
    # Multiple AFOs on the same field do not create repeated full increments.
    manure_increment_rows = {
        (r["land_cover"], r["pollutant"]): r
        for r in inputs["manure_concentration_increments"]
    }
    drawn_increments = {}
    field_component = state.pathway_load_rates.copy()
    manure_component = np.zeros_like(state.pathway_load_rates)
    for pid, months in manure_months.items():
        if months == 0.:
            continue
        idx = pid_to_index[pid]
        cover = state.parameters[idx]["land_cover"]
        for pol_idx, pol in enumerate(ctx.pollutants):
            pol = str(pol).upper()
            if pol not in {"TN", "TP", "BOD"}:
                continue
            key = (cover, pol)
            if key not in manure_increment_rows:
                raise ValueError(f"Missing manure concentration increment for {cover}/{pol}")
            if key not in drawn_increments:
                increment = _sample_stats(ctx, manure_increment_rows[key], kind="nonnegative")
                drawn_increments[key] = _finite(
                    increment, f"sampled manure concentration increment {cover}/{pol}"
                )
            base = float(state.concentrations[idx][pol])
            fraction = float(months) / 12.
            # The parcel-average concentration reflects only the eligible
            # non-feedlot agricultural acreage of this receiving parcel.
            area_fraction = eligible_area[pid] / areas[pid]
            incremental_concentration = fraction * drawn_increments[key] * area_fraction
            state.concentrations[idx][pol] = base + incremental_concentration
            runoff_in = plet_annual_surface_runoff_in(state.parameters[idx])[3]
            increment = runoff_in * INCH_OVER_HA_TO_LITERS * incremental_concentration / 1_000_000.
            state.pathway_load_rates[idx, pol_idx, 0] += increment * float(state.parameters[idx][f"load_multiplier_{pol.lower()}"])
            # Field component is later restricted to non-feedlot acreage.
            field_component[idx, pol_idx, 0] += (increment / area_fraction) * float(state.parameters[idx][f"load_multiplier_{pol.lower()}"])
            manure_component[idx, pol_idx, 0] += (increment / area_fraction) * float(state.parameters[idx][f"load_multiplier_{pol.lower()}"])
    # Separate direct feedlot source so feedlot-only and agricultural BMPs do
    # not automatically treat one another's loads.
    feedlot_component = np.zeros_like(state.pathway_load_rates)
    for op in operations:
        pid = str(op["pid"])
        idx = pid_to_index[pid]
        loads = _feedlot_runoff_kg(op, animals, state.parameters[idx], ctx.pollutants)
        for pol_idx, pol in enumerate(ctx.pollutants):
            pol = str(pol).upper()
            kg_ha = loads.get(pol, 0.) / areas[pid]
            feedlot_component[idx, pol_idx, 0] += kg_ha
    for pid, fraction_area in feedlot_area.items():
        if fraction_area <= 0.:
            continue
        idx = pid_to_index[pid]
        fraction = fraction_area / areas[pid]
        # Displace background PLET loads over the feedlot footprint. The
        # feedlot groundwater pathway is not modeled; avoid double counting.
        field_component[idx, :, :] *= (1. - fraction)
        manure_component[idx, :, :] *= (1. - fraction)
    state.pathway_load_rates[:, :, :] = field_component + feedlot_component
    state.baseline_pathway_load_rates = state.pathway_load_rates.copy()
    state.baseline_concentrations = [dict(x) for x in state.concentrations]
    ctx.current_afo_feedlot_pathway_rates = feedlot_component
    ctx.current_afo_manure_pathway_rates = manure_component
    ctx.baseline_afo_feedlot_pathway_rates = feedlot_component.copy()
    ctx.baseline_afo_manure_pathway_rates = manure_component.copy()
    return assignments
