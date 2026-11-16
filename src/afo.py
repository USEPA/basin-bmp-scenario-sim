"""Optional PLET-style animal feeding operations and spatial manure exposure.

Supports a legacy concentration-increment mode and optional mass-based mode.
See docs/animal_feeding_operations.md for inputs, assumptions, and limitations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import sqlite3
import numpy as np
import pandas as pd

from .constants import (
    AFO_FEEDLOT_CURVE_NUMBERS as _FEEDLOT_CN,
    AFO_FEEDLOT_MAX_CONCENTRATIONS as _FEEDLOT_MAX_CONC,
    AFO_SPECIES_ALIASES as _SPECIES_ALIAS,
    AFO_SPECIES,
    INCH_OVER_HA_TO_LITERS,
)
from .input_distributions import sample_group_key, stats_from_row, statistic_columns
from .input_validation import validate_stats_rows, validate_unique_rows
from .io_utils import read_csv_table
from .plet_rusle import plet_annual_surface_runoff_in, normalize_plet_land_cover
from .utils import normalize_columns


_EXCRETION_PARAMETERS = (
    "representative_mass_kg", "tn_kg_1000kg_day", "tp_kg_1000kg_day",
)


def _animal_species(value: Any) -> str:
    species = str(value).strip().lower()
    species = _SPECIES_ALIAS.get(species, species)
    if species not in AFO_SPECIES:
        raise ValueError(f"Unknown AFO species {value!r}; expected one of {', '.join(AFO_SPECIES)}")
    return species


def _read_table(conn: sqlite3.Connection, name: str, *, optional: bool = False) -> pd.DataFrame:
    names = {str(x[0]) for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if name not in names:
        if optional:
            return pd.DataFrame()
        raise ValueError(f"AFO GeoPackage is missing table {name!r}")
    # AFO example GeoPackages expose operation point geometry and explanatory
    # per-record notes in QGIS. Neither is a numerical model input; in
    # particular, the notes field must not be mistaken for a distribution
    # statistic in legacy concentration-increment mode.
    model_columns = [col[1] for col in conn.execute(f'PRAGMA table_info("{name}")')
                     if str(col[1]).strip().lower() not in {"fid", "geom", "notes"}]
    projection = ", ".join('"' + col.replace('"', '""') + '"' for col in model_columns)
    frame = pd.read_sql_query(f'SELECT {projection} FROM "{name}"', conn)
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
                    parcel_path: str | Path | None = None,
                    excretion_factors_path: str | Path | None = None,
                    runoff_adjusted_export: bool = False,
                    reference_runoff_mm: float | None = None,
                    feedlot_bmp_cps: Sequence[int] = ()) -> dict[str, Any]:
    """Load and validate AFO inputs and the optional mass-based runoff adjustment."""
    if not isinstance(runoff_adjusted_export, bool):
        raise ValueError("manure_loading.runoff_adjusted_export must be boolean")
    if method == "mass_based":
        if runoff_adjusted_export:
            if reference_runoff_mm is None:
                raise ValueError("runoff-adjusted manure export requires reference_runoff_mm")
            reference_runoff_mm = _finite(
                reference_runoff_mm, "manure_loading.reference_runoff_mm", low=1e-12
            )
        elif reference_runoff_mm is not None:
            raise ValueError("reference_runoff_mm requires runoff_adjusted_export: true")
    elif runoff_adjusted_export or reference_runoff_mm is not None:
        raise ValueError("runoff-adjusted manure export requires method: mass_based")
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
        recovery = _read_table(conn, "afo_feedlot_manure_effects", optional=True)
    operation_columns = ["afo_id", "pid", "feedlot_area_ha", "paved_pct", "allocation"]
    excretion_factors = {}
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
        _need_columns(management, "afo_manure_management", ["afo_id"])
        if mode == "statistical":
            _need_columns(eligibility, "afo_field_eligibility", ["pid", "eligible"])
            _need_columns(hydrology, "afo_hydrology", ["pid"])
            if surface_pathway not in pathways:
                raise ValueError(f"Configured surface pathway {surface_pathway!r} is not an active statistical pathway")
    if not fields.empty:
        _need_columns(fields, "afo_manure_fields", ["afo_id", "pid"])
    if not recovery.empty:
        if method != "mass_based":
            raise ValueError("afo_feedlot_manure_effects requires mass_based manure loading")
        _need_columns(recovery, "afo_feedlot_manure_effects", ["afo_id", "cps"])
    if operations.empty:
        raise ValueError("AFO package requires at least one operation")
    operations["afo_id"] = operations["afo_id"].astype(str).str.strip()
    operations["pid"] = operations["pid"].astype(object)
    if operations["afo_id"].eq("").any() or operations["afo_id"].duplicated().any():
        raise ValueError("afo_operations.afo_id must contain distinct nonblank IDs")
    op_ids = set(operations["afo_id"])
    if not recovery.empty:
        cps_allowed = {int(c) for c in feedlot_bmp_cps}
        recovery["afo_id"] = recovery["afo_id"].astype(str).str.strip()
        cps_normalized = []
        for _, row in recovery.iterrows():
            op_id = row["afo_id"]
            if op_id not in op_ids:
                raise ValueError(f"afo_feedlot_manure_effects references unknown operation {op_id!r}")
            cps = _integer_pid(row["cps"], "afo_feedlot_manure_effects.cps")
            cps = int(cps)
            if cps not in cps_allowed:
                raise ValueError(f"afo_feedlot_manure_effects CPS {cps} must be in feedlot_bmp_cps")
            cps_normalized.append(cps)
            key = "additional_collection_fraction"
            fixed = row.get(key)
            stats = {k[len(key)+1:]: v for k, v in row.items()
                     if str(k).startswith(key + "_") and pd.notna(v)}
            if pd.notna(fixed):
                if stats:
                    raise ValueError(f"{op_id}/CPS {cps}: cannot mix fixed and distributed {key}")
                stats = {"value": fixed}
            if not stats:
                raise ValueError(f"{op_id}/CPS {cps}: requires {key} or {key}_* statistics")
            validate_stats_rows(pd.DataFrame([{"parameter": key, **stats}]),
                                f"afo_feedlot_manure_effects[{op_id}/{cps}]")
            for name, value in stats.items():
                _finite(value, f"afo_feedlot_manure_effects[{op_id}/{cps}].additional_collection_fraction_{name}",
                        0, float('inf') if name == 'sd' else 1)
        recovery["cps"] = cps_normalized
        if recovery.duplicated(["afo_id", "cps"]).any():
            raise ValueError("Duplicate afo_feedlot_manure_effects operation/CPS pair")
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
        if method == "concentration_increment":
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
    animals["afo_id"] = animals["afo_id"].astype(str).str.strip()
    for op_id in op_ids:
        if not animals["afo_id"].eq(op_id).any():
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
        excretion_factors = reference_excretion_factors(excretion_factors_path)
        if management["afo_id"].astype(str).duplicated().any() or set(management["afo_id"].astype(str)) != op_ids:
            raise ValueError("afo_manure_management requires exactly one row per operation")
        # APLE-specific coefficients are not interchangeable with annual
        # export fractions. Fail rather than silently ignoring old inputs.
        deprecated = (
            "f_n", "alpha_n", "wep_fraction", "solids_pct", "season",
            "incorporated_fraction",
        )
        for _, row in management.iterrows():
            op_id = str(row["afo_id"])
            old = [name for name in deprecated if any(
                (column == name or column.startswith(name + "_")) and pd.notna(value)
                for column, value in row.items()
            )]
            if old:
                raise ValueError(
                    f"afo_manure_management[{op_id}] uses obsolete APLE parameters {old}; "
                    "replace with tn_surface_export_fraction and tp_surface_export_fraction "
                    "(do not convert coefficients mechanically)"
                )
            for name in ("tn_surface_export_fraction", "tp_surface_export_fraction"):
                value = row.get(name)
                stats = {
                    key[len(name) + 1:]: val for key, val in row.items()
                    if key.startswith(name + "_") and pd.notna(val)
                }
                if pd.notna(value):
                    if stats:
                        raise ValueError(f"{op_id}: {name} cannot mix fixed and distribution inputs")
                    stats = {"value": value}
                if not stats:
                    raise ValueError(f"afo_manure_management[{op_id}] requires {name} or {name}_* distribution")
                validate_stats_rows(
                    pd.DataFrame([{"parameter": name, **stats}]),
                    f"afo_manure_management[{op_id}].{name}",
                )
                for stat, stat_value in stats.items():
                    if stat == "sd":
                        _finite(stat_value, f"{op_id}.{name}_sd")
                    else:
                        _finite(stat_value, f"{op_id}.{name}_{stat}", 0, 1)
            for name in ("collection_fraction", "export_fraction",
                         "n_retention_fraction", "p_retention_fraction"):
                if name in row and pd.notna(row[name]):
                    _finite(row[name], name, 0, 1)
        for _, row in animals.iterrows():
            species = _animal_species(row["species"])
            unsupported = [
                column for column in animals.columns
                if any(column == name or column.startswith(name + "_")
                       for name in ("manure_kg_head_day", "tn_kg_head_day", "tp_kg_head_day",
                                    "manure_kg_1000kg_day"))
                and pd.notna(row[column])
            ]
            if unsupported:
                raise ValueError(
                    f"afo_animals contains unsupported excretion inputs {unsupported}; "
                    f"supported parameters are {', '.join(_EXCRETION_PARAMETERS)}"
                )
            for parameter in _EXCRETION_PARAMETERS:
                stats = _excretion_override_stats(row, parameter)
                if stats:
                    validate_stats_rows(
                        pd.DataFrame([{"parameter": parameter, **stats}]),
                        f"afo_animals[{row['afo_id']}/{species}].{parameter}",
                    )
                elif parameter not in excretion_factors.get(species, {}):
                    raise ValueError(
                        f"afo_animals[{row['afo_id']}/{species}] requires {parameter}; "
                        "supply an animal-row override or a species row in the excretion CSV"
                    )
    eligible_by_pid = {}
    hydrology_by_pid = {}
    if method == "mass_based" and mode == "statistical":
        for _, row in eligibility.iterrows():
            pid = _integer_pid(row["pid"], "afo_field_eligibility.pid")
            if pid not in pid_set or pid in eligible_by_pid:
                raise ValueError(f"Unknown/duplicate AFO eligibility pid {pid}")
            if float(row["eligible"]) not in (0., 1.):
                raise ValueError("AFO eligibility must be 0 or 1")
            eligible_by_pid[pid] = bool(float(row["eligible"]))
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
        # Constant-export mode needs no receiving-field hydrology. Direct
        # feedlot loading still needs precipitation and feedlot runoff at hosts.
        required_pids = active_hosts | ({pid for pid, yes in eligible_by_pid.items() if yes} if runoff_adjusted_export else set())
        for pid in required_pids:
            if pid not in hydrology_by_pid:
                raise ValueError(f"Missing statistical AFO hydrology for pid {pid}")
            row = hydrology_by_pid[pid]
            required = ["annual_runoff_mm"] if runoff_adjusted_export and eligible_by_pid.get(pid) else []
            if pid in active_hosts:
                required.extend(("annual_precip_mm", "feedlot_runoff_mm"))
            for name in required:
                if not (name in row and pd.notna(row[name])) and not any(
                    k.startswith(name + "_") and pd.notna(v) for k,v in row.items()
                ):
                    raise ValueError(f"Statistical AFO pid {pid} missing {name} or distribution")
        if not any(eligible_by_pid.values()):
            raise ValueError("Statistical AFO requires at least one eligible field")
    return {"operations": operations.to_dict("records"), "animals": animals.to_dict("records"),
            "excretion_factors": excretion_factors,
            "fields": fields.to_dict("records"),
            "manure_concentration_increments": increments.to_dict("records") if not increments.empty else [],
            "management": management.to_dict("records") if not management.empty else [],
            "feedlot_manure_effects": recovery.to_dict("records") if not recovery.empty else [],
            "eligible": eligible_by_pid, "hydrology": hydrology_by_pid,
            "surface_pathway": surface_pathway, "method": method,
            "runoff_adjusted_export": runoff_adjusted_export,
            "reference_runoff_mm": reference_runoff_mm,
            "geometry": {str(r.pid): r.geometry for r in parcels.itertuples()},
            "area_ha": area_by_pid}


def _select_manure_fields(ctx: Any, inputs: Mapping[str, Any], eligible_area: Mapping[str, float]) -> list[dict[str, Any]]:
    """Select receivers once per realization with shared explicit/stochastic rules."""
    operations = inputs["operations"]
    fields = inputs["fields"]
    geometry = inputs["geometry"]
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
            rec = {"afo_id": op_id, "host_pid": host, "receiving_pid": pid,
                   "eligible_area_ha": float(eligible_area[pid]),
                   "allocation": op["allocation"]}
            if inputs.get("method") != "mass_based":
                rec["manure_months"] = float(op["manure_months"])
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


def reference_excretion_factors(
    path: str | Path | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load native D384.1 rate statistics and required representative masses."""
    if path is None:
        path = Path(__file__).parent / "data" / "afo" / "manure_excretion_reference.csv"
    label = f"manure excretion factors ({path})"
    table = normalize_columns(read_csv_table(path))
    _need_columns(table, label, ["species", "parameter"])
    if table.empty:
        raise ValueError(f"{label} requires at least one species")
    for column in ("species", "parameter"):
        if table[column].isna().any() or table[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label}.{column} must not be blank")
        table[column] = table[column].astype(str).str.strip().str.lower()
    table["species"] = table["species"].map(_animal_species)
    unexpected = set(table["parameter"]) - set(_EXCRETION_PARAMETERS)
    if unexpected:
        raise ValueError(f"{label} contains unsupported parameters: {sorted(unexpected)}")
    validate_unique_rows(table, ["species", "parameter"], label)
    validate_stats_rows(table, label)
    factors = {}
    for species, rows in table.groupby("species", sort=False):
        missing = set(_EXCRETION_PARAMETERS) - set(rows["parameter"])
        if missing:
            raise ValueError(f"{label} species {species!r} is missing parameters: {sorted(missing)}")
        factors[species] = {}
        for _, row in rows.iterrows():
            stats = stats_from_row(row)
            group = row.get("sample_group")
            record: dict[str, Any] = dict(stats)
            if group is not None and pd.notna(group) and str(group).strip():
                record["sample_group"] = str(group).strip()
            factors[species][row["parameter"]] = record
    return factors


def _excretion_override_stats(animal: Mapping[str, Any], parameter: str) -> dict[str, float]:
    """Select a complete animal-row override without merging reference statistics."""
    prefix = parameter + "_"
    row = {key[len(prefix):]: value for key, value in animal.items() if key.startswith(prefix)}
    if parameter in animal and pd.notna(animal[parameter]):
        if "value" in row and pd.notna(row["value"]):
            raise ValueError(f"afo_animals supplies both {parameter} and {parameter}_value")
        row["value"] = animal[parameter]
    return stats_from_row({"parameter": parameter, **row})


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
    ctx: Any, animals: list[dict[str, Any]], op_id: str,
    *, excretion_cache: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    """Calculate as-excreted annual nutrient N/P inventories.

    Availability/handling losses are applied separately; animal-days is the
    number of days each enumerated head is present in the modeled year.
    """
    factors = ctx.afo_inputs["excretion_factors"]
    if excretion_cache is None:
        excretion_cache = {}
    totals = {"TN": 0.0, "TP": 0.0}
    for animal in animals:
        if str(animal["afo_id"]) != op_id:
            continue
        species = _animal_species(animal["species"])
        known = factors.get(species, {})
        number = _draw(ctx, animal, "count")
        days = _draw(ctx, animal, "animal_days", default=365.)
        if days > 366:
            raise ValueError(f"{op_id}: animal_days cannot exceed 366")
        realized = {}
        for field in _EXCRETION_PARAMETERS:
            kind = "positive" if field == "representative_mass_kg" else "nonnegative"
            stats = _excretion_override_stats(animal, field)
            if stats:
                rate = float(ctx._sample_from_stats(stats, kind=kind))
            else:
                if field not in known:
                    raise ValueError(f"{op_id}/{species}: missing manure excretion parameter {field}")
                reference = known[field]
                key = sample_group_key(reference, pid=repr((op_id, species)), variable=field)
                if key not in excretion_cache:
                    excretion_cache[key] = _sample_stats(ctx, reference, kind=kind)
                rate = excretion_cache[key]
            rate = _finite(rate, f"{op_id}/{species}.{field}")
            if field == "representative_mass_kg" and rate <= 0:
                raise ValueError(f"{op_id}/{species}.representative_mass_kg must be positive")
            realized[field] = rate
        scale = number * days * realized["representative_mass_kg"] / 1000.
        for nutrient, field in zip(("TN", "TP"), _EXCRETION_PARAMETERS[1:]):
            totals[nutrient] += scale * realized[field]
    return totals


def manure_surface_export_kg(
    *,
    applied_n_kg: float,
    applied_p_kg: float,
    tn_surface_export_fraction: float,
    tp_surface_export_fraction: float,
    runoff_adjusted_export: bool = False,
    runoff_mm: float | None = None,
    reference_runoff_mm: float | None = None,
) -> dict[str, float]:
    """Same-year manure-attributable surface TN/TP exports (kg/year).

    Default: applied nutrient mass times its annual export fraction.
    Opt-in: multiply each fraction by annual runoff/reference runoff and cap
    at unity. This is an empirical screening approximation, not APLE.
    """
    n = _finite(applied_n_kg, "applied_n_kg")
    p = _finite(applied_p_kg, "applied_p_kg")
    n_fraction = _finite(tn_surface_export_fraction, "tn_surface_export_fraction", 0, 1)
    p_fraction = _finite(tp_surface_export_fraction, "tp_surface_export_fraction", 0, 1)
    if not isinstance(runoff_adjusted_export, bool):
        raise ValueError("runoff_adjusted_export must be boolean")
    if runoff_adjusted_export:
        if runoff_mm is None or reference_runoff_mm is None:
            raise ValueError("Runoff adjustment requires runoff_mm and reference_runoff_mm")
        runoff = _finite(runoff_mm, "runoff_mm")
        reference = _finite(reference_runoff_mm, "reference_runoff_mm", low=1e-12)
        factor = runoff / reference
        n_fraction = min(1., n_fraction * factor)
        p_fraction = min(1., p_fraction * factor)
    return {"TN": n * n_fraction, "TP": p * p_fraction}


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
    # Sparse per-operation field sources allow later feedlot BMPs to change
    # manure availability without erasing earlier receiving-field BMP effects.
    operation_state: dict[str, dict[str, Any]] = {}
    sources_by_parcel: dict[int, list[dict[str, Any]]] = {}
    management = {str(rec["afo_id"]): rec for rec in inputs["management"]}
    excretion_cache: dict[tuple[str, str], float] = {}
    for op in sorted(ops, key=lambda row: str(row["afo_id"])):
        op_id = str(op["afo_id"])
        mgt = management[op_id]
        produced = animal_production(ctx, realized_animals, op_id, excretion_cache=excretion_cache)
        collection = _draw(ctx, mgt, "collection_fraction", default=1, fraction=True)
        export_fraction = _draw(ctx, mgt, "export_fraction", default=0, fraction=True)
        available_full = {
            "TN": produced["TN"] * (1 - export_fraction)
            * _draw(ctx, mgt, "n_retention_fraction", default=1, fraction=True),
            "TP": produced["TP"] * (1 - export_fraction)
            * _draw(ctx, mgt, "p_retention_fraction", default=1, fraction=True),
        }
        available = {pol: mass * collection for pol, mass in available_full.items()}
        operation_state[op_id] = dict(host_pid=str(op["pid"]),
                                      initial_collection=collection,
                                      current_collection=collection,
                                      available_full=available_full,
                                      sources=[])
        tn_export_fraction = _draw(ctx, mgt, "tn_surface_export_fraction", fraction=True)
        tp_export_fraction = _draw(ctx, mgt, "tp_surface_export_fraction", fraction=True)
        adjust_for_runoff = inputs.get("runoff_adjusted_export", False)
        reference_runoff = inputs.get("reference_runoff_mm")
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
            # For the default method, manure export needs no runoff input.
            # No artificial zero-runoff constraint is imposed in that mode.
            runoff_mm = None
            if adjust_for_runoff:
                runoff_mm, _, _ = _hydrology(inputs, pid, state, idx, ctx)
            elif state is not None:
                # Observed PLET runoff is used only for reporting and an
                # equivalent concentration, never as a factor in Option A.
                runoff_mm = plet_annual_surface_runoff_in(state.parameters[idx])[3] * 25.4
            exported_full = manure_surface_export_kg(
                applied_n_kg=available_full["TN"] * frac,
                applied_p_kg=available_full["TP"] * frac,
                tn_surface_export_fraction=tn_export_fraction,
                tp_surface_export_fraction=tp_export_fraction,
                runoff_adjusted_export=adjust_for_runoff,
                runoff_mm=runoff_mm,
                reference_runoff_mm=reference_runoff,
            )
            exported = {pol: val * collection for pol, val in exported_full.items()}
            for pol_idx, pol in enumerate(ctx.pollutants):
                pol_upper = str(pol).upper()
                full_rate = exported_full.get(pol_upper, 0.) / areas[pid]
                add = full_rate * collection
                manure[idx, pol_idx, surface_idx] += add
                if pol_upper in ("TN", "TP"):
                    source = dict(op_id=op_id, parcel_idx=idx, pol_idx=pol_idx,
                                  path_idx=surface_idx, full_rate=full_rate,
                                  current_rate=add)
                    operation_state[op_id]["sources"].append(source)
                    sources_by_parcel.setdefault(idx, []).append(source)
                if state is not None and runoff_mm > 0 and pol_upper in ("TN", "TP"):
                    # 1 mm over 1 hectare = 10,000 L; kg/ha -> mg/L = *100/mm.
                    state.concentrations[idx][pol_upper] += 100 * add / runoff_mm
            ledger.append(dict(
                **rec,
                animal_n_produced_kg=produced["TN"],
                animal_p_produced_kg=produced["TP"],
                operation_available_tn_kg=available["TN"],
                operation_available_tp_kg=available["TP"],
                tn_applied_kg=applied["TN"],
                tp_applied_kg=applied["TP"],
                tn_exported_kg=exported["TN"],
                tp_exported_kg=exported["TP"],
                runoff_adjusted_export=adjust_for_runoff,
                reference_runoff_mm=reference_runoff,
                runoff_mm=runoff_mm,
                tn_application_kg_ha=applied["TN"] / eligible[pid],
                tp_application_kg_ha=applied["TP"] / eligible[pid],
                tn_surface_export_fraction=tn_export_fraction,
                tp_surface_export_fraction=tp_export_fraction,
                tn_effective_export_fraction=(exported["TN"] / applied["TN"] if applied["TN"] else 0.),
                tp_effective_export_fraction=(exported["TP"] / applied["TP"] if applied["TP"] else 0.),
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
    ctx.afo_bmp_operation_state = operation_state
    ctx.afo_bmp_sources_by_parcel = sources_by_parcel
    effect_index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for effect in sorted(inputs.get("feedlot_manure_effects", []),
                         key=lambda rec: str(rec["afo_id"])):
        host = operation_state[str(effect["afo_id"])]["host_pid"]
        effect_index.setdefault((host, int(effect["cps"])), []).append(effect)
    ctx.afo_bmp_effects_by_host_cps = effect_index
    ctx.afo_bmp_effect_records = []
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
            def optional_draw(field):
                if any(
                    (column == field or column.startswith(field + "_"))
                    and value is not None and not _nan(value)
                    for column, value in supplied.items()
                ):
                    return _draw(ctx, supplied, field)
                return None
            cache[key] = (
                optional_draw("annual_runoff_mm"),
                optional_draw("annual_precip_mm"),
                optional_draw("feedlot_runoff_mm"),
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


def apply_afo_feedlot_manure_effect(
    ctx: Any, host_pid: str, cps: int, load_rates: np.ndarray,
    *, bmp_effect_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Apply an explicitly configured manure-collection gain after a feedlot BMP.

    The gain is the recovered fraction of the *remaining uncollected* manure,
    applied sequentially. The returned changes in kg/year are signed increases
    in receiving-field surface exports, after earlier field BMP treatment.
    No recovery effect is inferred from a CPS code or a runoff efficiency.
    """
    inputs = getattr(ctx, "afo_inputs", None) or {}
    if inputs.get("method") != "mass_based":
        return []
    operation_state = getattr(ctx, "afo_bmp_operation_state", {})
    if not operation_state:
        return []
    bmp_effect_scale = _finite(bmp_effect_scale, "bmp_effect_scale", 0, 1)
    rows = getattr(ctx, "afo_bmp_effects_by_host_cps", {}).get(
        (str(host_pid), int(cps)), []
    )
    changes: list[dict[str, Any]] = []
    for effect in rows:
        op_id = str(effect["afo_id"])
        op = operation_state[op_id]
        gain = _draw(ctx, effect, "additional_collection_fraction", fraction=True)
        gain *= bmp_effect_scale
        old = op["current_collection"]
        new = old + (1. - old) * gain
        for source in op["sources"]:
            updated_rate = source["full_rate"] * new
            delta_rate = updated_rate - source["current_rate"]
            source["current_rate"] = updated_rate
            idx, pol_idx, path_idx = source["parcel_idx"], source["pol_idx"], source["path_idx"]
            ctx.current_pathway_load_rates[idx, pol_idx, path_idx] += delta_rate
            ctx.current_afo_manure_pathway_rates[idx, pol_idx, path_idx] += delta_rate
            load_rates[idx, pol_idx] += delta_rate
            if delta_rate:
                changes.append(dict(afo_id=op_id, cps=int(cps),
                                    receiving_pid=str(ctx.parcel_ids[idx]),
                                    parcel_idx=idx, pol_idx=pol_idx,
                                    added_export_kg=delta_rate * float(ctx.parcel_area_ha[idx])))
        op["current_collection"] = new
        ctx.afo_bmp_effect_records.append(dict(
            afo_id=op_id, cps=int(cps), host_pid=str(host_pid),
            recovery_fraction=gain, collection_before=old,
            collection_after=new,
            added_available_tn_kg=(new - old) * op["available_full"]["TN"],
            added_available_tp_kg=(new - old) * op["available_full"]["TP"],
            added_surface_tn_kg=sum(rec["added_export_kg"] for rec in changes
                                    if rec["afo_id"] == op_id and
                                    str(ctx.pollutants[rec["pol_idx"]]).upper() == "TN"),
            added_surface_tp_kg=sum(rec["added_export_kg"] for rec in changes
                                    if rec["afo_id"] == op_id and
                                    str(ctx.pollutants[rec["pol_idx"]]).upper() == "TP"),
        ))
    return changes
