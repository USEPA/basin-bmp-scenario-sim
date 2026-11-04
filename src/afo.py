"""Optional PLET-style animal feeding operations and spatial manure exposure.

Supports a legacy concentration-increment mode and optional mass-based mode.
See docs/animal_feeding_operations.md for inputs, assumptions, and limitations.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence
import sqlite3

import numpy as np
import pandas as pd

from .constants import (
    AFO_FEEDLOT_CURVE_NUMBERS as _FEEDLOT_CN,
    AFO_FEEDLOT_MAX_CONCENTRATIONS as _FEEDLOT_MAX_CONC,
    AFO_SEASONAL_P_WEIGHTS as _SEASONS,
    AFO_SPECIES_ALIASES as _SPECIES_ALIAS,
    INCH_OVER_HA_TO_LITERS,
)
from .input_distributions import stats_from_row, statistic_columns
from .plet_rusle import plet_annual_surface_runoff_in, normalize_plet_land_cover


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


def load_afo_inputs(path: str | Path, parcels: pd.DataFrame, pollutants: Sequence[str],
                    *, method: str = "concentration_increment", mode: str = "plet_rusle",
                    pathways: Sequence[str] = ("surface",), surface_pathway: str = "surface",
                    parcel_path: str | Path | None = None) -> dict[str, Any]:
    """Load and fully validate optional AFO tables before worker processes start."""
    package = Path(path)
    if not package.is_file():
        raise FileNotFoundError(f"AFO GeoPackage not found: {package}")
    with sqlite3.connect(package) as conn:
        operations = _read_table(conn, "afo_operations")
        animals = _read_table(conn, "afo_animals")
        fields = _read_table(conn, "afo_manure_fields", optional=True)
        increments = _read_table(conn, "afo_manure_concentration_increments", optional=method == "mass_based")
        management = _read_table(conn, "afo_manure_management", optional=method != "mass_based")
        eligibility = _read_table(conn, "afo_field_eligibility", optional=mode != "statistical")
        hydrology = _read_table(conn, "afo_hydrology", optional=mode != "statistical")
    operation_columns = ["afo_id", "pid", "feedlot_area_ha", "paved_pct", "allocation"]
    if method == "concentration_increment":
        operation_columns.append("manure_months")
    _need_columns(operations, "afo_operations", operation_columns)
    if "manure_months" not in operations.columns:
        operations["manure_months"] = 12.0
    _need_columns(animals, "afo_animals", ["afo_id", "species"] + (["count"] if method == "concentration_increment" else []))
    if method not in {"mass_based", "concentration_increment"}:
        raise ValueError("manure_loading.method must be mass_based or concentration_increment")
    if method == "concentration_increment":
        _need_columns(increments, "afo_manure_concentration_increments", ["land_cover", "pollutant"])
    else:
        _need_columns(management, "afo_manure_management", ["afo_id", "season"])
        if mode == "statistical":
            _need_columns(eligibility, "afo_field_eligibility", ["pid", "eligible"])
            _need_columns(hydrology, "afo_hydrology", ["pid"])
            if surface_pathway not in pathways:
                raise ValueError(f"Configured surface pathway {surface_pathway!r} is not an active statistical pathway")
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
        if pd.notna(row["manure_months"]):
            _finite(row["manure_months"], f"{op_id}.manure_months", 0., 12.)
        allocation_mode = str(row["allocation"]).strip().lower()
        if allocation_mode not in {"explicit", "sampled"}:
            raise ValueError(f"Operation {op_id} allocation must be explicit or sampled")
        operations.at[k, "allocation"] = allocation_mode
        if allocation_mode == "sampled":
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
        if "count" not in row or pd.isna(row["count"]):
            if method != "mass_based" or not any(k.startswith("count_") and pd.notna(row[k]) for k in animals.columns):
                raise ValueError(f"{op_id} requires animal count or count_* distribution")
        else:
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
    if method == "concentration_increment":
        increments["land_cover"] = increments["land_cover"].map(normalize_plet_land_cover)
        increments["pollutant"] = increments["pollutant"].astype(str).str.upper().str.strip()
        if increments.duplicated(["land_cover", "pollutant"]).any():
            raise ValueError("Duplicate manure concentration increments by land_cover/pollutant")
        if not set(increments["land_cover"]).issubset({"cropland", "pastureland"}):
            raise ValueError("Manure concentration increments apply only to cropland/pastureland")
        if not set(increments["pollutant"]).issubset(set(map(str.upper, pollutants))):
            raise ValueError("Manure concentration increments reference unknown pollutants")
        for _, row in increments.iterrows():
            stats = stats_from_row(row, {"land_cover", "pollutant", "sample_group", "units"})
            if not stats:
                raise ValueError("Each manure concentration increment needs a value or distribution")
            for name, value in stats.items():
                _finite(value, f"afo_manure_concentration_increments.{name}")
    else:
        known = reference_excretion_factors()
        if management["afo_id"].astype(str).duplicated().any() or set(management["afo_id"].astype(str)) != op_ids:
            raise ValueError("afo_manure_management requires exactly one row per operation")
        for _, row in management.iterrows():
            if str(row["season"]).strip().lower() not in {"winter", "spring", "summer", "fall"}:
                raise ValueError("afo_manure_management.season must be winter/spring/summer/fall")
            for name in ("f_n", "wep_fraction", "solids_pct"):
                fixed_present = name in management.columns and pd.notna(row.get(name))
                stats_present = any(k.startswith(name + "_") and pd.notna(row[k])
                                    for k in management.columns)
                if not (fixed_present or stats_present):
                    raise ValueError(f"afo_manure_management requires {name} or {name}_* distribution")
            for name in ("f_n", "wep_fraction", "collection_fraction", "export_fraction",
                         "n_retention_fraction", "p_retention_fraction", "incorporated_fraction"):
                if name in row and pd.notna(row[name]):
                    _finite(row[name], name, 0, 1)
            if "solids_pct" in row and pd.notna(row["solids_pct"]):
                _finite(row["solids_pct"], "solids_pct", 0, 100)
        for _, row in animals.iterrows():
            species = str(row["species"]).strip().lower()
            species = _SPECIES_ALIAS.get(species, species)
            if species not in known and not all(
                    name in animals.columns and pd.notna(row.get(name)) or
                    any(k.startswith(name + "_") and pd.notna(row[k]) for k in animals.columns)
                    for name in ("manure_kg_head_day", "tn_kg_head_day", "tp_kg_head_day")):
                raise ValueError(f"Unsupported animal species {species!r}; supply all three production rates")
    eligible_by_pid = {}
    hydrology_by_pid = {}
    if method == "mass_based" and mode == "statistical":
        for _, row in eligibility.iterrows():
            pid = _integer_pid(row["pid"], "afo_field_eligibility.pid")
            if pid not in pid_set or pid in eligible_by_pid:
                raise ValueError(f"Unknown/duplicate AFO eligibility pid {pid}")
            if float(row["eligible"]) not in (0., 1.):
                raise ValueError("AFO eligibility must be 0 or 1")
            eligible_by_pid[pid] = bool(row["eligible"])
        # Inheritance: parcel-specific -> HUC12 -> NULL-pid domain default.
        # HUC12 keys are resolved from the canonical parcel_huc12 table.
        mapping = {}
        if parcel_path is not None:
            with sqlite3.connect(parcel_path) as con:
                names = {x[0] for x in con.execute("SELECT name FROM sqlite_master")}
                if "parcel_huc12" in names:
                    cols = {info[1] for info in con.execute("PRAGMA table_info(parcel_huc12)")}
                    huc_col = "huc_id" if "huc_id" in cols else "huc12"
                    mapping = {str(int(pid)): str(huc) for pid, huc in
                               con.execute(f"SELECT pid,{huc_col} FROM parcel_huc12")}
        by_pid, by_huc, default = {}, {}, None
        for _, row in hydrology.iterrows():
            rec = row.to_dict()
            pid_value = rec.get("pid")
            huc_value = rec.get("huc12")
            has_pid = pid_value is not None and not pd.isna(pid_value)
            has_huc = huc_value is not None and not pd.isna(huc_value)
            if has_pid and has_huc:
                raise ValueError("afo_hydrology row cannot specify both pid and huc12")
            if has_pid:
                pid = _integer_pid(pid_value, "afo_hydrology.pid")
                if pid not in pid_set or pid in by_pid:
                    raise ValueError(f"Unknown/duplicate AFO hydrology pid {pid}")
                by_pid[pid] = rec
            elif has_huc:
                key = str(huc_value).strip()
                if key in by_huc:
                    raise ValueError(f"Duplicate AFO hydrology huc12 {key}")
                by_huc[key] = rec
            else:
                if default is not None:
                    raise ValueError("Duplicate default AFO hydrology row")
                default = rec
            for name in ("annual_runoff_mm", "annual_precip_mm", "feedlot_runoff_mm"):
                if name in row and pd.notna(row[name]):
                    _finite(row[name], name)
        for pid in pid_set:
            rec = by_pid.get(pid) or by_huc.get(mapping.get(pid, "")) or default
            if rec is not None:
                hydrology_by_pid[pid] = rec
        active_hosts = {str(op["pid"]) for op in operations.to_dict("records")}
        for pid in set(eligible_by_pid).union(active_hosts):
            if pid not in hydrology_by_pid:
                raise ValueError(f"Missing statistical AFO hydrology for pid {pid}")
            row = hydrology_by_pid[pid]
            required = ["annual_precip_mm", "annual_runoff_mm"]
            if pid in active_hosts:
                required.append("feedlot_runoff_mm")
            for name in required:
                if not (name in row and pd.notna(row[name])) and not any(
                    k.startswith(name + "_") and pd.notna(v) for k,v in row.items()
                ):
                    raise ValueError(f"Statistical AFO pid {pid} missing {name} or distribution")
        if not any(eligible_by_pid.values()):
            raise ValueError("Statistical AFO requires at least one eligible field")
    return {"operations": operations.to_dict("records"), "animals": animals.to_dict("records"),
            "fields": fields.to_dict("records"),
            "manure_concentration_increments": increments.to_dict("records") if not increments.empty else [],
            "management": management.to_dict("records") if not management.empty else [],
            "eligible": eligible_by_pid, "hydrology": hydrology_by_pid,
            "surface_pathway": surface_pathway, "method": method,
            "geometry": {str(r.pid): r.geometry for r in parcels.itertuples()},
            "area_ha": area_by_pid}


def _select_manure_fields(ctx: Any, inputs: Mapping[str, Any], eligible_area: Mapping[str, float]) -> list[dict[str, Any]]:
    """Select receivers once per realization with shared explicit/stochastic rules."""
    operations = inputs["operations"]
    fields = inputs["fields"]
    geometry = inputs["geometry"]
    manure_months: dict[str, float] = {}
    assignments: list[dict[str, Any]] = []
    # Reuse exactly the same spatial selection for the legacy concentration
    # mode and the new mass-conserving nutrient-allocation mode.
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
            rec = {"afo_id": op_id, "host_pid": host, "receiving_pid": pid,
                                "manure_months": float(op["manure_months"]),
                                "eligible_area_ha": float(eligible_area[pid]),
                                "allocation": op["allocation"]}
            field_row = next((r for r in fields if str(r["afo_id"]) == op_id and str(r["pid"]) == pid), None)
            if field_row is not None and field_row.get("weight") is not None and not pd.isna(field_row["weight"]):
                rec["weight"] = _finite(field_row["weight"], "afo_manure_fields.weight")
            assignments.append(rec)
    return assignments



def _feedlot_concentration(op: Mapping[str, Any], animals: Sequence[Mapping[str, Any]], pol: str) -> float:
    """PLET feedlot mg/L from pollutant-specific equivalent animal density."""
    if pol not in _FEEDLOT_MAX_CONC:
        return 0.
    area_ac = float(op["feedlot_area_ha"]) / 0.40468564224
    eau = sum(float(animal["count"]) * float(animal[f"factor_{pol.lower()}"])
              for animal in animals if str(animal["afo_id"]) == str(op["afo_id"]))
    return min(eau / area_ac, 100.) / 100. * _FEEDLOT_MAX_CONC[pol]


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
        concentration = _feedlot_concentration(op, animals, pol)
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
    assignments = _select_manure_fields(ctx, inputs, eligible_area)
    manure_months = {}
    for rec in assignments:
        pid = rec["receiving_pid"]
        manure_months[pid] = max(manure_months.get(pid, 0.), rec["manure_months"])
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


def reference_excretion_factors() -> dict[str, dict[str, float]]:
    """Documented starting animal-class factors in kg/head/day (as excreted)."""
    path = Path(__file__).parent / "data" / "manure_excretion_reference.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["species"]: {
                key: float(row[key])
                for key in ("manure_kg_head_day", "tn_kg_head_day", "tp_kg_head_day")
            }
            for row in csv.DictReader(handle)
        }


def _draw(
    ctx: Any,
    row: Mapping[str, Any],
    key: str,
    *,
    default: float | None = None,
    fraction: bool = False,
) -> float:
    """Use existing inline distribution syntax or a deterministic named value."""
    stats = {
        column[len(key) + 1:]: value
        for column, value in row.items()
        if column.startswith(key + "_") and value is not None and not _nan(value)
    }
    if stats:
        value = _sample_stats(
            ctx, row, prefix=key + "_", kind="fraction" if fraction else "nonnegative"
        )
    elif key in row and row[key] is not None and not _nan(row[key]):
        value = float(row[key])
    elif default is not None:
        value = default
    else:
        raise ValueError(f"Missing required manure parameter {key}")
    return _finite(value, key, 0, 1 if fraction else float("inf"))


def _nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def animal_production(
    ctx: Any, animals: list[dict[str, Any]], op_id: str
) -> dict[str, float]:
    """Calculate as-excreted annual wet manure and nutrient N/P inventories.

    Availability/handling losses are applied separately; animal-days is the
    number of days each enumerated head is present in the modeled year.
    """
    factors = reference_excretion_factors()
    totals = {"manure": 0.0, "TN": 0.0, "TP": 0.0}
    for animal in animals:
        if str(animal["afo_id"]) != op_id:
            continue
        raw_species = str(animal["species"]).strip().lower()
        species = _SPECIES_ALIAS.get(raw_species, raw_species)
        known = factors.get(species, {})
        number = _draw(ctx, animal, "count")
        days = _draw(ctx, animal, "animal_days", default=365.)
        if days > 366:
            raise ValueError(f"{op_id}: animal_days cannot exceed 366")
        for nutrient, field in (
            ("manure", "manure_kg_head_day"),
            ("TN", "tn_kg_head_day"),
            ("TP", "tp_kg_head_day"),
        ):
            rate = _draw(ctx, animal, field, default=known.get(field))
            totals[nutrient] += number * days * rate
    return totals


def manure_surface_export_kg(
    *,
    applied_n_kg: float,
    applied_p_kg: float,
    runoff_mm: float,
    precipitation_mm: float,
    available_n_fraction: float,
    n_exponent: float,
    p_wep_fraction: float,
    solids_pct: float,
    incorporated_fraction: float,
    season: str,
) -> dict[str, float]:
    """Annual manure-derived surface N and dissolved P exports in kg.

    APLE 2.5 annual manure P release equations (1), (3), (6) and same-year
    mineralization/fall-availability assumptions. A supplied incorporation
    fraction represents effective surface P removed; APLE's full injection
    volume/depth-dependent subroutine is *not* reproduced here.
    """
    n = _finite(applied_n_kg, "applied_n_kg")
    p = _finite(applied_p_kg, "applied_p_kg")
    q = _finite(runoff_mm, "runoff_mm")
    rain = _finite(precipitation_mm, "precipitation_mm")
    f_n = _finite(available_n_fraction, "available_n_fraction", 0, 1)
    exponent = _finite(n_exponent, "n_exponent", 0)
    wep = _finite(p_wep_fraction, "p_wep_fraction", 0, 1)
    solids = _finite(solids_pct, "solids_pct", 0, 100)
    incorporated = _finite(incorporated_fraction, "incorporated_fraction", 0, 1)
    season = str(season).lower().strip()
    if season not in _SEASONS:
        raise ValueError(f"season must be one of {sorted(_SEASONS)}")
    if q > rain + 1e-9:
        raise ValueError("Annual surface runoff cannot exceed annual precipitation")
    if rain == 0 or q == 0:
        return {"TN": 0., "TP": 0., "surface_available_p_kg": 0.}
    ratio = min(q / rain, 1.0)
    n_export = n * f_n * ratio ** exponent
    surface_p = p * (0.4 if solids < 15 else 1.0) * (1.0 - incorporated)
    wep_available = surface_p * (wep + (1. - wep) * _SEASONS[season])
    if season == "fall":
        # APLE postpones 25% of available fall WEP to the following year;
        # the remainder is not exported or tracked in our single-year model.
        wep_available *= .75
    if solids < 15:
        cover = .5
        wep_available *= 2.2 * (250 * cover) / (250 * cover + 300.1)
    p_export = wep_available * ratio ** 1.225
    if p_export > p + 1e-8 or n_export > n + 1e-8:
        raise AssertionError("Nutrient export exceeds applied manure mass")
    return {"TN": n_export, "TP": p_export, "surface_available_p_kg": wep_available}


def mass_afo_realization(
    ctx: Any, *, state: Any = None
) -> tuple[list[dict], list[dict]]:
    """Add AFO loads to PLET or statistical pathway arrays, once per scenario.

    Loads are kg/ha/year in current_pathway_load_rates. The manure allocation
    ledger records kg, and is unaffected by subsequent BMP applications.
    """
    inputs = ctx.afo_inputs
    if not inputs or inputs.get("method") != "mass_based":
        return [], []
    pids = list(map(str, ctx.parcel_ids))
    idx_map = {pid: i for i, pid in enumerate(pids)}
    areas = inputs["area_ha"]
    ops = inputs["operations"]
    # Draw animal inventory once per realization; feedlot density and manure
    # production must use the same realized animal counts.
    realized_animals = []
    for animal in inputs["animals"]:
        sampled = dict(animal)
        sampled["count"] = _draw(ctx, animal, "count")
        for key in list(sampled):
            if key.startswith("count_"):
                del sampled[key]
        realized_animals.append(sampled)
    footprint = {pid: 0. for pid in pids}
    for op in ops:
        footprint[str(op["pid"])] += float(op["feedlot_area_ha"])
    if state is not None:
        eligible = {
            pid: areas[pid] - footprint[pid]
            for pid in pids
            if state.parameters[idx_map[pid]]["land_cover"] in {"cropland", "pastureland"}
            and areas[pid] - footprint[pid] > 1e-12
        }
        surface_idx = 0
    else:
        eligible = {
            pid: areas[pid] - footprint[pid]
            for pid in pids
            if inputs["eligible"].get(pid, False)
            and areas[pid] - footprint[pid] > 1e-12
        }
        surface = inputs["surface_pathway"]
        if surface not in ctx.pathway_names:
            raise ValueError(f"AFO surface pathway {surface!r} not present in statistical pathways")
        surface_idx = ctx.pathway_names.index(surface)
    assignments = _select_manure_fields(ctx, inputs, eligible)
    by_op: dict[str, list[dict]] = {}
    for rec in assignments:
        by_op.setdefault(rec["afo_id"], []).append(rec)
    n_parcels, n_pol, n_paths = ctx.current_pathway_load_rates.shape
    feedlot = np.zeros((n_parcels, n_pol, n_paths), float)
    manure = np.zeros_like(feedlot)
    ledger: list[dict] = []
    management = {str(rec["afo_id"]): rec for rec in inputs["management"]}
    for op in sorted(ops, key=lambda row: str(row["afo_id"])):
        op_id = str(op["afo_id"])
        mgt = management[op_id]
        produced = animal_production(ctx, realized_animals, op_id)
        collection = _draw(ctx, mgt, "collection_fraction", default=1, fraction=True)
        export_fraction = _draw(ctx, mgt, "export_fraction", default=0, fraction=True)
        available = {
            "manure": produced["manure"] * collection * (1 - export_fraction),
            "TN": produced["TN"] * collection * (1 - export_fraction)
            * _draw(ctx, mgt, "n_retention_fraction", default=1, fraction=True),
            "TP": produced["TP"] * collection * (1 - export_fraction)
            * _draw(ctx, mgt, "p_retention_fraction", default=1, fraction=True),
        }
        available_n = _draw(ctx, mgt, "f_n", fraction=True)
        exponent_n = _draw(ctx, mgt, "alpha_n", default=1.)
        p_wep = _draw(ctx, mgt, "wep_fraction", fraction=True)
        solids = _draw(ctx, mgt, "solids_pct")
        incorporated = _draw(ctx, mgt, "incorporated_fraction", default=0, fraction=True)
        season = str(mgt["season"]).strip().lower()
        chosen = by_op[op_id]
        custom_weights = [rec.get("weight") for rec in chosen]
        if any(weight is not None for weight in custom_weights):
            if not all(weight is not None for weight in custom_weights):
                raise ValueError(f"{op_id}: all receiving parcels must have a weight if one does")
            weights = [float(weight) for weight in custom_weights]
        else:
            weights = [float(eligible[rec["receiving_pid"]]) for rec in chosen]
        if sum(weights) <= 0:
            raise ValueError(f"{op_id}: allocated field weights must sum positive")
        for rec, weight in zip(chosen, weights):
            pid = rec["receiving_pid"]
            idx = idx_map[pid]
            frac = weight / sum(weights)
            applied = {key: total * frac for key, total in available.items()}
            runoff_mm, precip_mm, _ = _hydrology(inputs, pid, state, idx, ctx)
            exported = manure_surface_export_kg(
                applied_n_kg=applied["TN"],
                applied_p_kg=applied["TP"],
                runoff_mm=runoff_mm,
                precipitation_mm=precip_mm,
                available_n_fraction=available_n,
                n_exponent=exponent_n,
                p_wep_fraction=p_wep,
                solids_pct=solids,
                incorporated_fraction=incorporated,
                season=season,
            )
            for pol_idx, pol in enumerate(ctx.pollutants):
                pol_upper = str(pol).upper()
                add = exported.get(pol_upper, 0.) / areas[pid]
                manure[idx, pol_idx, surface_idx] += add
                if state is not None and runoff_mm > 0 and pol_upper in ("TN", "TP"):
                    # 1 mm over 1 hectare = 10,000 L; kg/ha -> mg/L = *100/mm.
                    state.concentrations[idx][pol_upper] += 100 * add / runoff_mm
            ledger.append(dict(
                **rec,
                animal_manure_produced_kg=produced["manure"],
                animal_n_produced_kg=produced["TN"],
                animal_p_produced_kg=produced["TP"],
                operation_available_manure_kg=available["manure"],
                operation_available_tn_kg=available["TN"],
                operation_available_tp_kg=available["TP"],
                manure_applied_kg=applied["manure"],
                tn_applied_kg=applied["TN"],
                tp_applied_kg=applied["TP"],
                tn_exported_kg=exported["TN"],
                tp_dissolved_exported_kg=exported["TP"],
                precipitation_mm=precip_mm,
                runoff_mm=runoff_mm,
                tn_application_kg_ha=applied["TN"] / eligible[pid],
                tp_application_kg_ha=applied["TP"] / eligible[pid],
                f_n=available_n,
                alpha_n=exponent_n,
                wep_fraction=p_wep,
                season=season,
                incorporated_fraction=incorporated,
            ))
        # Feedlot source is separate from manure-field loading.
        host = str(op["pid"])
        host_idx = idx_map[host]
        if state is not None:
            direct_kg = _feedlot_runoff_kg(
                op, realized_animals, state.parameters[host_idx], ctx.pollutants
            )
        else:
            _, host_precip_mm, feedlot_runoff_mm = _hydrology(
                inputs, host, state, host_idx, ctx
            )
            if feedlot_runoff_mm is None:
                raise ValueError(f"Statistical AFO host {host} requires feedlot_runoff_mm")
            if feedlot_runoff_mm > host_precip_mm:
                raise ValueError(
                    f"Statistical AFO feedlot runoff exceeds annual precipitation for host {host}"
                )
            direct_kg = {
                str(pol).upper(): feedlot_runoff_mm * 10000 / 1e6
                * op["feedlot_area_ha"]
                * _feedlot_concentration(op, realized_animals, str(pol).upper())
                for pol in ctx.pollutants
            }
        for pol_idx, pol in enumerate(ctx.pollutants):
            feedlot[host_idx, pol_idx, surface_idx] += (
                direct_kg.get(str(pol).upper(), 0.) / areas[host]
            )
    # Baseline over the AFO footprint is displaced, not counted twice. Manure
    # loads are already absolute parcel area-normalized and must not be scaled.
    baseline = ctx.current_pathway_load_rates
    for pid, footprint_ha in footprint.items():
        if footprint_ha > 0:
            baseline[idx_map[pid], :, :] *= 1 - footprint_ha / areas[pid]
    baseline += manure + feedlot
    if state is not None:
        state.baseline_pathway_load_rates = baseline.copy()
        state.baseline_concentrations = [dict(row) for row in state.concentrations]
    ctx.current_afo_feedlot_pathway_rates = feedlot
    ctx.current_afo_manure_pathway_rates = manure
    ctx.baseline_afo_feedlot_pathway_rates = feedlot.copy()
    ctx.baseline_afo_manure_pathway_rates = manure.copy()
    return assignments, ledger


def _hydrology(
    inputs: Mapping, pid: str, state: Any, idx: int, ctx: Any
) -> tuple[float, float, float | None]:
    """Runoff and actual (not runoff-day-corrected) precipitation, in mm/yr."""
    if state is None:
        supplied = inputs["hydrology"].get(pid)
        if not supplied:
            raise ValueError(f"Statistical AFO parcel {pid} is missing runoff/precipitation inputs")
        # Inherited hydrology draws are shared, and each parcel's draw is reused
        # when multiple operations use that field.
        cache = getattr(ctx, "_afo_hydro_cache", None)
        if cache is None:
            cache = {}
            ctx._afo_hydro_cache = cache
        key = id(supplied)
        if key not in cache:
            cache[key] = (
                _draw(ctx, supplied, "annual_runoff_mm"),
                _draw(ctx, supplied, "annual_precip_mm"),
                _draw(ctx, supplied, "feedlot_runoff_mm")
                if any(
                    column == "feedlot_runoff_mm"
                    or column.startswith("feedlot_runoff_mm_")
                    for column in supplied
                )
                else None,
            )
        return cache[key]
    params = state.parameters[idx]
    runoff_mm = plet_annual_surface_runoff_in(params)[3] * 25.4
    rain = params.get("annual_precip_in")
    if rain is None:
        correction = float(params.get("rain_correction_fraction", 1.0))
        rain = (
            float(params["avg_rain_in"])
            * float(params["rain_days"])
            * float(params["runoff_day_fraction"])
            / correction
        )
    return runoff_mm, float(rain) * 25.4, None
