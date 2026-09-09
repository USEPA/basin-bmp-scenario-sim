"""Validation rules for model configuration and normalized input tables.

This module contains model-facing validation policy only. File deserialization
lives in :mod:`src.io_utils`, while defaults, normalization, input loading, and
assembly live in :mod:`src.input_config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import (
    CFG_BMP_COST,
    CFG_BMP_EFFICIENCY,
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_BUFFER_DEPTH_FT,
    CFG_N_SCENARIOS,
    CFG_PARALLEL,
    CFG_POLLUTANT_LOAD_RATE,
    COL_CPS,
    COL_PATHWAY,
    COL_PID,
    COL_POLLUTANT,
    COL_PROBABILITY,
    LOAD_CONCENTRATIONS,
    LOAD_GROUNDWATER_CONCENTRATIONS,
)
from .input_distributions import DISTRIBUTION_ID, stats_from_row
from .utils import ci_get


@dataclass(frozen=True)
class PhysicalDomain:
    """Allowed physical domain for a user-supplied numeric quantity.

    Parameters
    ----------
    low : float or None
        Lower bound, or ``None`` when no lower bound applies.
    high : float or None
        Upper bound, or ``None`` when no upper bound applies.
    low_inclusive : bool, optional
        Whether the lower bound is included.
    high_inclusive : bool, optional
        Whether the upper bound is included.
    """

    low: Optional[float] = None
    high: Optional[float] = None
    low_inclusive: bool = True
    high_inclusive: bool = True

    def contains(self, value: float) -> bool:
        """Return whether ``value`` is finite and inside the domain."""
        if not np.isfinite(float(value)):
            return False
        numeric = float(value)
        if self.low is not None:
            if self.low_inclusive:
                if numeric < self.low:
                    return False
            elif numeric <= self.low:
                return False
        if self.high is not None:
            if self.high_inclusive:
                if numeric > self.high:
                    return False
            elif numeric >= self.high:
                return False
        return True

    def describe(self) -> str:
        """Return a compact human-readable description of the domain."""
        if self.low is None and self.high is None:
            return "all finite values"
        if self.low is None:
            operator = "<=" if self.high_inclusive else "<"
            return f"{operator} {self.high:g}"
        if self.high is None:
            operator = ">=" if self.low_inclusive else ">"
            return f"{operator} {self.low:g}"
        left = "[" if self.low_inclusive else "("
        right = "]" if self.high_inclusive else ")"
        return f"{left}{self.low:g}, {self.high:g}{right}"


NONNEGATIVE_DOMAIN = PhysicalDomain(low=0.0)
POSITIVE_DOMAIN = PhysicalDomain(low=0.0, low_inclusive=False)
FRACTION_DOMAIN = PhysicalDomain(low=0.0, high=1.0)
EFFICIENCY_DOMAIN = PhysicalDomain(high=1.0)
PERCENT_DOMAIN = PhysicalDomain(low=0.0, high=100.0)
CN_DOMAIN = PhysicalDomain(low=0.0, high=100.0, low_inclusive=False)
IA_RATIO_DOMAIN = PhysicalDomain(low=0.0, high=0.20)


_TABLE_PHYSICAL_DOMAINS: Dict[str, PhysicalDomain] = {
    CFG_POLLUTANT_LOAD_RATE: NONNEGATIVE_DOMAIN,
    CFG_BMP_EFFICIENCY: EFFICIENCY_DOMAIN,
    CFG_BMP_COST: NONNEGATIVE_DOMAIN,
    LOAD_CONCENTRATIONS: NONNEGATIVE_DOMAIN,
    LOAD_GROUNDWATER_CONCENTRATIONS: NONNEGATIVE_DOMAIN,
}


_PARAMETER_PHYSICAL_DOMAINS: Dict[str, PhysicalDomain] = {
    "annual_precip_in": NONNEGATIVE_DOMAIN,
    "rain_days": NONNEGATIVE_DOMAIN,
    "rain_correction_fraction": FRACTION_DOMAIN,
    "runoff_day_fraction": FRACTION_DOMAIN,
    "cn": CN_DOMAIN,
    "ia_ratio": IA_RATIO_DOMAIN,
    "infiltration_fraction": FRACTION_DOMAIN,
    "runoff_multiplier": NONNEGATIVE_DOMAIN,
    "groundwater_multiplier": NONNEGATIVE_DOMAIN,
    "r": NONNEGATIVE_DOMAIN,
    "k": NONNEGATIVE_DOMAIN,
    "ls": NONNEGATIVE_DOMAIN,
    "c": NONNEGATIVE_DOMAIN,
    "p": NONNEGATIVE_DOMAIN,
    "sdr": FRACTION_DOMAIN,
    "sediment_multiplier": NONNEGATIVE_DOMAIN,
    "sediment_delivery_multiplier": NONNEGATIVE_DOMAIN,
    "sediment_n_pct": PERCENT_DOMAIN,
    "sediment_p_pct": PERCENT_DOMAIN,
    "enrichment_ratio": NONNEGATIVE_DOMAIN,
    "fraction_subsurface_shallow": FRACTION_DOMAIN,
}


def physical_parameter_domain(parameter: Any) -> Optional[PhysicalDomain]:
    """Return the physical domain for a canonical model parameter, if known."""
    name = str(parameter).strip().lower()
    if name.startswith("load_multiplier_"):
        return NONNEGATIVE_DOMAIN
    return _PARAMETER_PHYSICAL_DOMAINS.get(name)


_SAMPLING_KIND_PHYSICAL_DOMAINS: Dict[str, PhysicalDomain] = {
    "efficiency": EFFICIENCY_DOMAIN,
    "load_rate": NONNEGATIVE_DOMAIN,
    "nonnegative": NONNEGATIVE_DOMAIN,
    "fraction": FRACTION_DOMAIN,
    "cn": CN_DOMAIN,
    "ia_ratio": IA_RATIO_DOMAIN,
    "percent": PERCENT_DOMAIN,
    "positive": POSITIVE_DOMAIN,
}


def physical_domain_for_sampling_kind(kind: Optional[str]) -> Optional[PhysicalDomain]:
    """Return the physical domain represented by a sampling semantic hint."""
    if kind is None:
        return None
    return _SAMPLING_KIND_PHYSICAL_DOMAINS.get(str(kind).strip().lower())


def sampling_kind_for_parameter(parameter: Any) -> Optional[str]:
    """Return the sampler semantic required by a physical parameter domain."""
    name = str(parameter).strip().lower()
    if name == "cn":
        return "cn"
    if name == "ia_ratio":
        return "ia_ratio"
    if name in {"infiltration_fraction", "rain_correction_fraction", "runoff_day_fraction", "sdr", "fraction_subsurface_shallow"}:
        return "fraction"
    if name in {"sediment_n_pct", "sediment_p_pct"}:
        return "percent"
    domain = physical_parameter_domain(name)
    if domain == NONNEGATIVE_DOMAIN:
        return "nonnegative"
    return None


def validate_scalar_in_domain(value: Any, domain: PhysicalDomain, label: str) -> float:
    """Validate one scalar against an explicit physical domain and return it."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    if not domain.contains(numeric):
        raise ValueError(
            f"{label}={numeric:g} is outside the allowed physical domain {domain.describe()}"
        )
    return numeric


def validate_parameter_value(parameter: Any, value: Any, *, label: Optional[str] = None) -> float:
    """Validate one model parameter value against its centralized physical domain."""
    name = str(parameter).strip().lower()
    domain = physical_parameter_domain(name)
    if domain is None:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{label or name} must be finite")
        return numeric
    return validate_scalar_in_domain(value, domain, label or name)


def validate_numeric_columns_in_domain(
    df: pd.DataFrame,
    columns: Sequence[str],
    domain: PhysicalDomain,
    label: str,
) -> None:
    """Validate dataframe columns against one physical numeric domain.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the columns to validate.
    columns : sequence of str
        Numeric columns that share the physical domain.
    domain : PhysicalDomain
        Allowed physical domain.
    label : str
        Dataset label used in validation errors.
    """
    if df is None or df.empty:
        return
    for column in columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
        if domain.low is not None:
            invalid |= (numeric < domain.low).to_numpy() if domain.low_inclusive else (numeric <= domain.low).to_numpy()
        if domain.high is not None:
            invalid |= (numeric > domain.high).to_numpy() if domain.high_inclusive else (numeric >= domain.high).to_numpy()
        if np.any(invalid):
            bad_index = df.index[np.asarray(invalid, dtype=bool)][0]
            bad_value = df.loc[bad_index, column]
            raise ValueError(
                f"{label} row {bad_index} {column}={bad_value!r} is outside the "
                f"allowed physical domain {domain.describe()}"
            )


def validate_physical_distribution_rows(df: pd.DataFrame, label: str) -> None:
    """Validate supplied distribution statistics against physical input domains.

    Normal rows are allowed for bounded quantities because the sampler treats
    the physical domain as explicit truncation support. The supplied mean and
    any explicit endpoints/percentiles must nevertheless be physically valid.
    Standard deviation is a spread parameter and is checked separately by
    :func:`validate_numeric_distribution_rows`.
    """
    if df is None or df.empty:
        return
    table_domain = _TABLE_PHYSICAL_DOMAINS.get(str(label).strip().lower())
    for index, row in df.iterrows():
        domain = table_domain
        parameter = None
        if "parameter" in row.index and _nonblank(row.get("parameter")):
            parameter = str(row.get("parameter")).strip().lower()
            domain = physical_parameter_domain(parameter) or domain
        if domain is None:
            continue
        stats = _row_stats_raw(row)
        for statistic, value in stats.items():
            if statistic == "sd":
                continue
            item = f"{parameter}.{statistic}" if parameter else statistic
            validate_scalar_in_domain(
                value, domain, f"{label} row {index} {item}"
            )


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str, logger: Any = None) -> None:
    """Require a normalized input table to contain the requested columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        required : Sequence[str]
            Required column names.
        label : str
            Context label used in diagnostics and validation errors.
        logger : Any
            Logger used for diagnostic and progress messages.

        Raises
        ------
        ValueError
            If one or more required columns are absent.
        
    """
    del logger
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _nonblank(value: Any) -> bool:
    """Return whether an input cell contains a nonblank value.

        Parameters
        ----------
        value : Any
            Input value to normalize or evaluate.

        Returns
        -------
        bool
            ``True`` when the value is nonblank; otherwise ``False``.
        
    """
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _row_stats_raw(row: Mapping[str, Any]) -> Dict[str, float]:
    """Return normalized numeric statistics from an input row.

        Parameters
        ----------
        row : Mapping[str, Any]
            Input table row.

        Returns
        -------
        Dict[str, float]
            Normalized numeric statistics extracted from the row.
        
    """
    return stats_from_row(row)


def validate_numeric_distribution_rows(df: pd.DataFrame, label: str) -> None:
    """Validate the standardized numeric value/distribution contract row-by-row.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        label : str
            Context label used in diagnostics and validation errors.

        Raises
        ------
        ValueError
            If a row contains incomplete, conflicting, non-finite, or non-monotonic distribution statistics.
        
    """
    if df is None or df.empty:
        return
    for index, row in df.iterrows():
        distribution_id = row.get(DISTRIBUTION_ID)
        stats = _row_stats_raw(row)
        if not stats:
            suffix = f" (distribution_id={distribution_id!r})" if _nonblank(distribution_id) else ""
            raise ValueError(f"{label} row {index} has no fixed value or distribution statistics{suffix}")
        has_value = "value" in stats
        distribution_keys = set(stats) - {"value"}
        if has_value and distribution_keys:
            raise ValueError(
                f"{label} row {index} mixes fixed value with distribution statistics; "
                "use either value or a distribution, not both"
            )
        for key, value in stats.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{label} row {index} statistic {key!r} must be finite")
        if has_value:
            continue
        has_mean = "mean" in stats
        has_sd = "sd" in stats
        has_min = "min" in stats
        has_max = "max" in stats
        percentiles = sorted((q, stats[f"p{q}"]) for q in range(1, 100) if f"p{q}" in stats)
        if has_sd and float(stats["sd"]) < 0.0:
            raise ValueError(f"{label} row {index} sd must be >= 0")
        if has_min != has_max:
            raise ValueError(f"{label} row {index} must provide both min and max when either is supplied")
        if has_min and float(stats["min"]) > float(stats["max"]):
            raise ValueError(f"{label} row {index} has min > max")
        if not ((has_mean and has_sd) or (has_min and has_max)):
            raise ValueError(
                f"{label} row {index} needs value, mean+sd, or min+max (with optional percentiles)"
            )
        if percentiles and not (has_min and has_max):
            raise ValueError(f"{label} row {index} percentile distributions require min and max endpoints")
        if percentiles and (has_mean or has_sd):
            raise ValueError(
                f"{label} row {index} mixes percentile and normal-distribution statistics; "
                "use min/max + percentile columns OR mean/sd (optionally with min/max)"
            )
        if has_min and has_max:
            points = [(0, float(stats["min"]))]
            points.extend((q, float(value)) for q, value in percentiles)
            points.append((100, float(stats["max"])))
            for (p0, q0), (p1, q1) in zip(points[:-1], points[1:]):
                if q1 < q0:
                    raise ValueError(
                        f"{label} row {index} distribution is not monotonic: p{p0}={q0} > p{p1}={q1}"
                    )
            if has_mean and not (float(stats["min"]) <= float(stats["mean"]) <= float(stats["max"])):
                raise ValueError(f"{label} row {index} mean must lie between min and max")


def validate_distribution_bounds(
    df: pd.DataFrame,
    label: str,
    *,
    parameter_col: str,
    bounds: Mapping[str, Tuple[Optional[float], Optional[float]]],
) -> None:
    """Validate fixed/support statistics against parameter-specific bounds.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        label : str
            Context label used in diagnostics and validation errors.
        parameter_col : str
            Column containing canonical parameter names.
        bounds : Mapping[str, Tuple[Optional[float], Optional[float]]]
            Allowed lower and upper bounds keyed by parameter name.

        Raises
        ------
        ValueError
            If a fixed value, mean, endpoint, or percentile lies outside the allowed parameter bounds.
        
    """
    if df is None or df.empty:
        return
    for index, row in df.iterrows():
        parameter = str(row[parameter_col]).strip().lower()
        if parameter not in bounds:
            continue
        low, high = bounds[parameter]
        stats = _row_stats_raw(row)
        for name, value in stats.items():
            if name == "sd":
                continue
            numeric = float(value)
            if low is not None and numeric < low:
                raise ValueError(f"{label} row {index} {parameter}.{name}={numeric} is below minimum {low}")
            if high is not None and numeric > high:
                raise ValueError(f"{label} row {index} {parameter}.{name}={numeric} exceeds maximum {high}")


def validate_stats_table(df: pd.DataFrame, label: str) -> None:
    """Validate a table using the shared numeric input-distribution schema.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        label : str
            Context label used in diagnostics and validation errors.
        
    """
    validate_numeric_distribution_rows(df, label)
    validate_physical_distribution_rows(df, label)


def validate_stats_rows(df: pd.DataFrame, label: str) -> None:
    """Validate every row using the shared numeric input-distribution schema.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        label : str
            Context label used in diagnostics and validation errors.
        
    """
    validate_numeric_distribution_rows(df, label)
    validate_physical_distribution_rows(df, label)


def validate_distribution_catalog(catalog: pd.DataFrame) -> None:
    """Validate reusable distribution identifiers and numeric definitions.

        Parameters
        ----------
        catalog : pd.DataFrame
            Reusable distribution catalog, if configured.

        Raises
        ------
        ValueError
            If ``distribution_id`` values are blank or duplicated.
        
    """
    require_columns(catalog, [DISTRIBUTION_ID], "input_distributions")
    raw_ids = catalog[DISTRIBUTION_ID]
    blank_ids = raw_ids.isna() | raw_ids.astype(str).str.strip().eq("")
    if blank_ids.any():
        rows = catalog.index[blank_ids].tolist()
        raise ValueError(f"input_distributions contains blank distribution_id values at rows {rows}")
    normalized_ids = raw_ids.astype(str).str.strip()
    duplicate = normalized_ids.duplicated(keep=False)
    if duplicate.any():
        ids = sorted(normalized_ids.loc[duplicate].unique().tolist())
        raise ValueError(f"input_distributions contains duplicate distribution_id values: {ids}")
    validate_numeric_distribution_rows(catalog, "input_distributions")


def validate_bmp_selection_table(df: pd.DataFrame, cps: Sequence[int]) -> None:
    """Validate a normalized explicit BMP-selection weight table.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        cps : Sequence[int]
            Conservation Practice Standard (CPS) code or codes.

        Raises
        ------
        ValueError
            If CPS identifiers or selection weights are invalid, duplicated, missing, or do not define positive total weight.
        
    """
    required = {COL_CPS, COL_PROBABILITY}
    if not required.issubset(df.columns):
        raise ValueError(
            f"bmp selection file must contain {sorted(required)} or an accepted selection-weight alias"
        )
    cps_numeric = pd.to_numeric(df[COL_CPS], errors="coerce")
    invalid_cps = (~np.isfinite(cps_numeric)) | (cps_numeric % 1 != 0)
    if invalid_cps.any():
        bad_rows = df.loc[invalid_cps, [COL_CPS, COL_PROBABILITY]].head(5).to_dict(orient="records")
        raise ValueError(
            "bmp selection cps values must be finite integers; "
            f"example bad rows: {bad_rows}"
        )
    configured_cps = {int(value) for value in cps}
    filtered_cps = cps_numeric.astype(int)
    duplicated = filtered_cps.duplicated(keep=False)
    if duplicated.any():
        duplicate_cps = sorted(filtered_cps.loc[duplicated].unique().tolist())
        raise ValueError(
            "bmp selection file contains duplicate selection-weight rows for cps values: "
            f"{duplicate_cps}"
        )
    probs = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")
    invalid = (~np.isfinite(probs)) | (probs < 0.0)
    if invalid.any():
        bad_rows = df.loc[invalid, [COL_CPS, COL_PROBABILITY]].head(5).to_dict(orient="records")
        raise ValueError(
            "bmp selection weights must be finite and nonnegative; "
            f"example bad rows: {bad_rows}"
        )
    found_cps = set(filtered_cps.tolist())
    missing = sorted(configured_cps - found_cps)
    if missing:
        raise ValueError(f"bmp selection file is missing probability rows for cps values: {missing}")
    if float(probs.sum()) <= 0.0:
        raise ValueError("bmp_sel selection weights sum to zero or negative")


def validate_trajectory_table(df: pd.DataFrame) -> None:
    """Validate the canonical outlet-trajectory table schema and required values.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.

        Raises
        ------
        ValueError
            If required trajectory columns are missing, blank, or contain non-finite numeric values.
        
    """
    required = {
        "scenario", "pollutant", "oid", "x_axis", "y_axis", "step", "x_value", "y_value"
    }
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Canonical trajectory table is missing required columns: {missing}")
    for column in ("scenario", "step", "x_value", "y_value"):
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(numeric)
        if invalid.any():
            bad_rows = df.loc[invalid].head(5).to_dict(orient="records")
            raise ValueError(
                f"Canonical trajectory table column {column!r} contains non-finite values; "
                f"example bad rows: {bad_rows}"
            )
    for column in ("pollutant", "oid", "x_axis", "y_axis"):
        blank = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"Canonical trajectory table contains blank values in required column {column!r}"
            )


def _rows_for_pid(table: Optional[pd.DataFrame], pid: str) -> List[pd.Series]:
    """Resolve wildcard parameter rows plus parcel-specific overrides.

        Parameters
        ----------
        table : Optional[pd.DataFrame]
            Input table containing model data.
        pid : str
            Parcel identifier.

        Returns
        -------
        List[pd.Series]
            Rows applicable to the specified parcel.
        
    """
    if table is None or table.empty:
        return []
    from .plet_rusle import canonical_parameter_name
    pids = table[COL_PID].astype(str)
    combined = pd.concat([table[pids == "*"], table[pids == str(pid)]], ignore_index=True)
    if combined.empty:
        return []
    combined = combined.assign(
        _canonical_parameter=combined["parameter"].map(canonical_parameter_name)
    )
    combined = combined.drop_duplicates(subset=["_canonical_parameter"], keep="last")
    return [row for _, row in combined.iterrows()]


def validate_plet_input_table(table: pd.DataFrame, parcel_ids: Sequence[str]) -> pd.DataFrame:
    """Validate and normalize required PLET classifications for every parcel.

        Parameters
        ----------
        table : pd.DataFrame
            Input table containing model data.
        parcel_ids : Sequence[str]
            Parcel identifiers in model order.

        Returns
        -------
        pd.DataFrame
            Validated and normalized PLET input table.

        Raises
        ------
        ValueError
            If PLET classifications or parameter rows violate the PLET input contract.
        
    """
    from .plet_rusle import (
        PLET_CLASSIFICATION_PARAMETERS,
        _PLET_DERIVED_PARAMETERS,
        canonical_parameter_name,
        normalize_plet_hsg,
        normalize_plet_land_cover,
    )
    normalized = table.copy()
    raw_parameter_labels = normalized["parameter"].map(
        lambda value: str(value).strip().lower().replace("-", "_").replace(" ", "_")
    )
    removed_parameters = sorted(set(raw_parameter_labels[raw_parameter_labels.str.startswith("irrigat")]))
    if removed_parameters:
        raise ValueError(f"PLET irrigation parameters are no longer supported: {removed_parameters}")
    normalized["parameter"] = normalized["parameter"].map(canonical_parameter_name)
    supplied_derived = sorted(
        set(normalized.loc[normalized["parameter"].isin(_PLET_DERIVED_PARAMETERS), "parameter"])
    )
    if supplied_derived:
        raise ValueError(
            "PLET parcel inputs may not specify "
            f"{supplied_derived}; define cn and infiltration_fraction in the required "
            "load_generation.hydrology_lookup table instead"
        )
    classification_mask = normalized["parameter"].isin(PLET_CLASSIFICATION_PARAMETERS)
    if classification_mask.any() and "value" not in normalized.columns:
        raise ValueError("PLET land_cover and hsg classifications require a fixed value column")
    for row_index in normalized.index[classification_mask]:
        parameter = str(normalized.at[row_index, "parameter"])
        value = normalized.at[row_index, "value"]
        if pd.isna(value) or str(value).strip() == "":
            raise ValueError(f"PLET classification row {row_index} for {parameter} requires a fixed value")
        if parameter == "land_cover":
            normalized.at[row_index, "value"] = normalize_plet_land_cover(value)
        else:
            normalized.at[row_index, "value"] = normalize_plet_hsg(value)
    for pid in map(str, parcel_ids):
        effective = {canonical_parameter_name(row["parameter"]): row for row in _rows_for_pid(normalized, pid)}
        missing = [parameter for parameter in PLET_CLASSIFICATION_PARAMETERS if parameter not in effective]
        if missing:
            raise ValueError(f"PLET inputs for pid={pid} are missing required classifications: {missing}")
    return normalized


def validate_plet_runtime_inputs(
    plet_inputs: pd.DataFrame,
    rusle_inputs: Optional[pd.DataFrame],
    pollutant_concentrations: Optional[pd.DataFrame],
    groundwater_concentrations: Optional[pd.DataFrame],
    parcel_ids: Sequence[str],
    pollutants: Sequence[str],
) -> None:
    """Validate all PLET/RUSLE coverage before the simulation worker starts.

        Parameters
        ----------
        plet_inputs : pd.DataFrame
            PLET parameter input table.
        rusle_inputs : Optional[pd.DataFrame]
            RUSLE parameter input table, if configured.
        pollutant_concentrations : Optional[pd.DataFrame]
            Pollutant concentration input table, if configured.
        groundwater_concentrations : Optional[pd.DataFrame]
            Groundwater pollutant concentrations or concentration table, if configured.
        parcel_ids : Sequence[str]
            Parcel identifiers in model order.
        pollutants : Sequence[str]
            Pollutant names in model order.

        Raises
        ------
        ValueError
            If any parcel lacks required PLET/RUSLE parameters or pollutant concentrations for runtime load generation.
        
    """
    from .plet_rusle import _REQUIRED_PLET_INPUTS, _REQUIRED_RUSLE, canonical_parameter_name

    if rusle_inputs is not None and not rusle_inputs.empty and "parameter" in rusle_inputs.columns:
        raw_parameter_labels = rusle_inputs["parameter"].map(
            lambda value: str(value).strip().lower().replace("-", "_").replace(" ", "_")
        )
        removed_parameters = sorted(
            set(raw_parameter_labels[raw_parameter_labels.str.startswith("watershed_area")])
        )
        if removed_parameters:
            raise ValueError(
                f"RUSLE watershed-area parameters are no longer supported: {removed_parameters}; "
                "supply 'sdr' instead"
            )

    def effective_parameters(table: Optional[pd.DataFrame], pid: str) -> Dict[str, pd.Series]:
        """Return the effective parameter rows for a parcel.

                Parameters
                ----------
                table : Optional[pd.DataFrame]
                    Input table containing model data.
                pid : str
                    Parcel identifier.

                Returns
                -------
                Dict[str, pd.Series]
                    Effective parameter rows keyed by canonical parameter name.
                
        """
        if table is None:
            return {}
        return {canonical_parameter_name(row["parameter"]): row for row in _rows_for_pid(table, pid)}

    def has_concentration(table: Optional[pd.DataFrame], pid: str, pollutant: str) -> bool:
        """Return whether a parcel has a pollutant concentration input.

                Parameters
                ----------
                table : Optional[pd.DataFrame]
                    Input table containing model data.
                pid : str
                    Parcel identifier.
                pollutant : str
                    Pollutant name.

                Returns
                -------
                bool
                    ``True`` when a matching concentration input exists; otherwise ``False``.
                
        """
        if table is None or table.empty:
            return False
        pids = table[COL_PID].astype(str)
        pols = table[COL_POLLUTANT].astype(str)
        return bool((((pids == "*") | (pids == pid)) & (pols == pollutant)).any())

    for pid in map(str, parcel_ids):
        plet_effective = effective_parameters(plet_inputs, pid)
        missing_plet = [name for name in _REQUIRED_PLET_INPUTS if name not in plet_effective]
        if missing_plet:
            raise ValueError(f"PLET inputs for pid={pid} are missing required parameters: {missing_plet}")
        rusle_effective = effective_parameters(rusle_inputs, pid)
        if rusle_effective:
            missing_rusle = [name for name in _REQUIRED_RUSLE if name not in rusle_effective]
            if missing_rusle:
                raise ValueError(f"RUSLE inputs for pid={pid} are incomplete; missing: {missing_rusle}")
        for pollutant in pollutants:
            pol = str(pollutant).upper()
            if pol in {"TN", "TP"} and not has_concentration(pollutant_concentrations, pid, pol):
                raise ValueError(f"Runoff concentration for pid={pid}, pollutant={pol} is required")
            if pol != "TSS" and not has_concentration(groundwater_concentrations, pid, pol):
                raise ValueError(
                    f"Groundwater concentration for pid={pid}, pollutant={pol} is required in plet_rusle mode"
                )
            if pol == "TSS" and not rusle_effective and not has_concentration(pollutant_concentrations, pid, pol):
                raise ValueError(f"TSS for pid={pid} requires complete RUSLE inputs or a TSS concentration")


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate configuration values after defaults and normalization are applied.

    Parameters
    ----------
    cfg : dict[str, Any]
        Normalized model configuration mapping.

    Raises
    ------
    ValueError
        If a configuration value is non-finite, non-integral where an integer
        is required, or outside its allowed physical range.
    """
    n_scenarios_raw = ci_get(cfg, CFG_N_SCENARIOS)
    n_scenarios = validate_scalar_in_domain(
        n_scenarios_raw, POSITIVE_DOMAIN, CFG_N_SCENARIOS
    )
    if not float(n_scenarios).is_integer():
        raise ValueError(f"{CFG_N_SCENARIOS} must be an integer >= 1")

    validate_scalar_in_domain(
        ci_get(cfg, CFG_BUFFER_DEPTH_FT), POSITIVE_DOMAIN, CFG_BUFFER_DEPTH_FT
    )
    validate_scalar_in_domain(
        ci_get(cfg, CFG_BMP_FAIL_RATE), FRACTION_DOMAIN, CFG_BMP_FAIL_RATE
    )
    validate_scalar_in_domain(
        ci_get(cfg, CFG_BMP_FAIL_REDUCTION), FRACTION_DOMAIN, CFG_BMP_FAIL_REDUCTION
    )

    limit_n = ci_get(cfg, CFG_BMP_LIMIT_N)
    if limit_n is not None:
        numeric_limit_n = validate_scalar_in_domain(
            limit_n, NONNEGATIVE_DOMAIN, CFG_BMP_LIMIT_N
        )
        if not numeric_limit_n.is_integer():
            raise ValueError(f"{CFG_BMP_LIMIT_N} must be an integer >= 0")

    limit_usd = ci_get(cfg, CFG_BMP_LIMIT_USD)
    if limit_usd is not None:
        validate_scalar_in_domain(
            limit_usd, NONNEGATIVE_DOMAIN, CFG_BMP_LIMIT_USD
        )

    parallel = ci_get(cfg, CFG_PARALLEL)
    if not isinstance(parallel, dict):
        raise ValueError(f"{CFG_PARALLEL} must be a mapping")
    if "n_jobs" not in parallel or parallel["n_jobs"] is None:
        raise ValueError("parallel.n_jobs must be specified after configuration normalization")
    n_jobs = validate_scalar_in_domain(
        parallel["n_jobs"], POSITIVE_DOMAIN, "parallel.n_jobs"
    )
    if not n_jobs.is_integer():
        raise ValueError("parallel.n_jobs must be an integer >= 1")


def validate_unique_rows(df: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    """Reject duplicate rows for a logical table key.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        keys : Sequence[str]
            Logical key columns used to identify duplicate rows.
        label : str
            Context label used in diagnostics and validation errors.

        Raises
        ------
        ValueError
            If duplicate rows exist for the requested logical key.
        
    """
    duplicate = df.duplicated(list(keys), keep=False)
    if duplicate.any():
        preview = df.loc[duplicate, list(keys)].head(10).to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate rows for {list(keys)}: {preview}")


def validate_statistical_efficiency_coverage(
    df: pd.DataFrame, cps: Sequence[int], pollutants: Sequence[str], pathways: Sequence[str]
) -> pd.DataFrame:
    """Require complete CPS x pollutant x pathway coverage in statistical mode.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        cps : Sequence[int]
            Conservation Practice Standard (CPS) code or codes.
        pollutants : Sequence[str]
            Pollutant names in model order.
        pathways : Sequence[str]
            Pollutant transport pathway names.

        Returns
        -------
        pd.DataFrame
            Validated BMP-efficiency table with complete pathway coverage.

        Raises
        ------
        ValueError
            If statistical-mode BMP efficiencies do not provide complete CPS/pollutant/pathway coverage.
        
    """
    out = df.copy()
    if COL_PATHWAY not in out.columns:
        if list(pathways) != ["surface"]:
            raise ValueError(
                "statistical mode with multiple pathways requires a pathway column in bmp_efficiency"
            )
        out[COL_PATHWAY] = "surface"
    supplied = set(out[COL_PATHWAY].astype(str))
    expected = set(pathways)
    if supplied != expected:
        raise ValueError(
            "statistical mode requires pollutant_load_rate and bmp_efficiency to use the same pathways; "
            f"expected {sorted(expected)}, found {sorted(supplied)} in bmp_efficiency"
        )
    validate_unique_rows(out, [COL_CPS, COL_POLLUTANT, COL_PATHWAY], CFG_BMP_EFFICIENCY)
    missing = []
    for cps_code in [int(value) for value in cps]:
        for pollutant in [str(value) for value in pollutants]:
            for pathway in pathways:
                mask = (
                    (out[COL_CPS].astype(int) == cps_code)
                    & (out[COL_POLLUTANT] == pollutant)
                    & (out[COL_PATHWAY] == pathway)
                )
                if not mask.any():
                    missing.append((cps_code, pollutant, pathway))
    if missing:
        preview = ", ".join(
            f"cps={cps_code}, pollutant={pollutant}, pathway={pathway}"
            for cps_code, pollutant, pathway in missing[:20]
        )
        raise ValueError(f"bmp_efficiency is missing required statistical-mode coverage: {preview}")
    validate_stats_rows(out, CFG_BMP_EFFICIENCY)
    return out.reset_index(drop=True)


def validate_statistical_load_rates(
    df: pd.DataFrame,
    parcels: pd.DataFrame,
    pollutants: Sequence[str],
) -> Tuple[List[str], bool]:
    """Validate statistical parcel-load-rate coverage and return pathways/mode.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        parcels : pd.DataFrame
            Parcel table or geospatial parcel dataset.
        pollutants : Sequence[str]
            Pollutant names in model order.

        Returns
        -------
        Tuple[List[str], bool]
            Detected pathway names and whether load rates are supplied in aggregate form.

        Raises
        ------
        ValueError
            If statistical pollutant load rates do not provide complete parcel/pollutant/pathway coverage.
        
    """
    parcel_ids = parcels[COL_PID].astype(str).tolist()
    explicit = COL_PATHWAY in df.columns
    pathways = list(dict.fromkeys(df[COL_PATHWAY].astype(str).tolist())) if explicit else []
    keys = [COL_PID, COL_POLLUTANT] + ([COL_PATHWAY] if explicit else [])
    validate_unique_rows(df, keys, CFG_POLLUTANT_LOAD_RATE)
    missing = []
    if explicit:
        for pid in parcel_ids:
            for pollutant in pollutants:
                for pathway in pathways:
                    mask = (
                        (df[COL_PID] == pid)
                        & (df[COL_POLLUTANT] == pollutant)
                        & (df[COL_PATHWAY] == pathway)
                    )
                    if not mask.any():
                        missing.append((pid, pollutant, pathway))
    else:
        for pid in parcel_ids:
            for pollutant in pollutants:
                mask = (df[COL_PID] == pid) & (df[COL_POLLUTANT] == pollutant)
                if not mask.any():
                    missing.append((pid, pollutant, None))
    if missing:
        preview = ", ".join(
            f"pid={pid}, pollutant={pollutant}" + (f", pathway={pathway}" if pathway else "")
            for pid, pollutant, pathway in missing[:20]
        )
        raise ValueError(f"pollutant_load_rate is missing required statistical-mode coverage: {preview}")
    return pathways, not explicit
