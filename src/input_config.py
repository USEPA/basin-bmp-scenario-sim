"""Normalize, default, load, and assemble model inputs.

Filesystem and serialization operations live in :mod:`src.io_utils`, while
validation policy lives in :mod:`src.input_validation`. This module owns
configuration defaults, aliases, type normalization, input preparation, and
construction of the validated data bundle consumed by the simulation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, Tuple
from collections import defaultdict
import geopandas as gpd
import numpy as np
import pandas as pd

from .constants import (
    CFG_BMP_COST,
    CFG_BMP_EFFICIENCY,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BMP_SEL_PROB_VIA_COSTS,
    CFG_BUFFER_DEPTH_FT,
    CFG_OUTPUTS,
    CFG_VERBOSE,
    CFG_CPS,
    CFG_DELIVERY_RATIOS,
    CFG_DOMAIN,
    CFG_N_SCENARIOS,
    CFG_OUTLET_LOC,
    CFG_OUTLET_MEAN,
    CFG_OUTLET_TARGET,
    CFG_PARALLEL,
    CFG_PARCEL_OUT,
    CFG_PARCEL_P,
    CFG_PARCEL_UP,
    CFG_PARCELS,
    CFG_POLLUTANT_LOAD_RATE,
    CFG_POLLUTANTS,
    CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE,
    CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW,
    CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    CFG_RANDOM_SEED,
    CFG_LOAD_GENERATION,
    CFG_INPUT_DISTRIBUTIONS,
    LOAD_MODE_STATISTICAL,
    LOAD_MODE_PLET_RUSLE,
    LOAD_PLET_INPUTS,
    LOAD_HYDROLOGY_LOOKUP,
    LOAD_RUSLE_INPUTS,
    LOAD_CONCENTRATIONS,
    LOAD_GROUNDWATER_CONCENTRATIONS,
    LOAD_GROUNDWATER_LOADS,
    LOAD_TREAT_GROUNDWATER_WITH_BMPS,
    COL_CPS,
    COL_MEAN,
    COL_OID,
    COL_OIDS,
    COL_PID,
    COL_PID_UP,
    COL_POLLUTANT,
    COL_PROBABILITY,
    COL_TARGET,
    COL_UNIT,
    COL_PATHWAY,
    PATHWAY_VALUES,
    PLET_PATHWAY_VALUES,
    DEFAULT_BMP_FAIL_REDUCTION,
    DEFAULT_BUFFER_DEPTH_FT,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
)
from .io_utils import read_csv_table, read_geodataframe, read_parquet_table
from .utils import ci_get, normalize_columns, normalize_pollutant_label
from .logging_utils import log_scope
from .input_distributions import (
    DISTRIBUTION_ID,
    statistic_columns,
    stats_from_row,
)
from .input_validation import (
    FRACTION_DOMAIN,
    NONNEGATIVE_DOMAIN,
    POSITIVE_DOMAIN,
    require_columns,
    validate_config,
    validate_distribution_bounds,
    validate_distribution_catalog,
    validate_numeric_columns_in_domain,
    validate_numeric_distribution_rows,
    validate_plet_input_table,
    validate_plet_runtime_inputs,
    validate_statistical_efficiency_coverage,
    validate_statistical_load_rates,
    validate_stats_rows,
    validate_stats_table,
    validate_unique_rows,
    validate_bmp_selection_table,
    validate_trajectory_table,
)









def _merge_csvs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one or more CSV files and combine them into one table.

        Parameters
        ----------
        paths : str, pathlib.Path, or sequence of str or pathlib.Path
            One or more CSV file paths.
        required_cols : sequence of str
            Columns that must be present in every file.
        label : str
            Human-readable dataset name used in logs and errors.
        logger : Any
            Logger used for progress and duplicate warnings.

        Returns
        -------
        pandas.DataFrame
            Concatenated dataframe with duplicates removed on the required key
            columns.
        
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames: List[pd.DataFrame] = []
    for p in paths:
        logger.verbose(f"Reading {label} from {p}")
        df = read_csv_table(p)
        df = normalize_columns(df)
        require_columns(df, required_cols, f"{label} ({p})", logger)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    dedup_subset = list(required_cols)
    if COL_PATHWAY in out.columns and COL_PATHWAY not in dedup_subset:
        dedup_subset.append(COL_PATHWAY)

    dup = out.duplicated(subset=dedup_subset, keep=False)
    if dup.any():
        logger.warning(f"Duplicate rows detected in {label}; keeping first occurrence")
        out = out.drop_duplicates(subset=dedup_subset, keep="first")
    return out


def load_bmp_selection_probabilities(
    path: Union[str, Path],
    cps: Sequence[int],
    logger: Any,
) -> pd.DataFrame:
    """Read and normalize explicit BMP-selection probabilities.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the BMP-selection probability CSV file.
        cps : Sequence[int]
            Conservation Practice Standard (CPS) code or codes.
        logger : Any
            Logger used for diagnostic and progress messages.

        Returns
        -------
        pd.DataFrame
            Normalized BMP-selection probability table.
        
    """
    logger.verbose(f"Reading BMP selection probabilities from {path}")
    df = normalize_columns(read_csv_table(path))

    if COL_PROBABILITY not in df.columns:
        alias = next((name for name in ("pr", "p") if name in df.columns), None)
        if alias is not None:
            df[COL_PROBABILITY] = df[alias]

    if COL_CPS in df.columns:
        cps_numeric = pd.to_numeric(df[COL_CPS], errors="coerce")
        finite_integer = np.isfinite(cps_numeric) & (cps_numeric % 1 == 0)
        df.loc[finite_integer, COL_CPS] = cps_numeric.loc[finite_integer].astype(int)

    if COL_PROBABILITY in df.columns:
        df[COL_PROBABILITY] = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")

    configured_cps = {int(value) for value in cps}
    if COL_CPS in df.columns:
        df = df[df[COL_CPS].isin(configured_cps)].copy()

    validate_bmp_selection_table(df, cps)
    df[COL_CPS] = df[COL_CPS].astype(int)
    df[COL_PROBABILITY] = df[COL_PROBABILITY].astype(float)
    df[COL_PROBABILITY] = df[COL_PROBABILITY] / float(df[COL_PROBABILITY].sum())

    result = df[[COL_CPS, COL_PROBABILITY]].reset_index(drop=True)
    logger.verbose(
        f"Loaded explicit BMP selection probabilities from {path}: "
        f"{result.to_dict(orient='records')}"
    )
    return result

def load_trajectory_records(
    path: Union[str, Path],
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Read and normalize canonical outlet-trajectory records for plotting.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the canonical trajectory Parquet file.

        Returns
        -------
        Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
            Trajectory records keyed by pollutant, outlet, x-axis, and y-axis definitions.
        
    """
    df = read_parquet_table(path)
    validate_trajectory_table(df)

    for column in ("scenario", "step", "x_value", "y_value"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    out: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
    df = df.sort_values(["scenario", "pollutant", "oid", "x_axis", "y_axis", "step"])
    for _, row in df.iterrows():
        key = (str(row["pollutant"]), str(row["oid"]), str(row["x_axis"]), str(row["y_axis"]))
        out[key].append((int(row["scenario"]), float(row["x_value"]), float(row["y_value"])))
    return out

def _ensure_projected(gdf: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Ensure a geospatial dataframe uses a projected CRS.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geometry table.
    logger : Any
        Logger used to report reprojection activity.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame in a projected coordinate reference system.
    """
    if gdf.crs is None or not gdf.crs.is_projected:
        est = gdf.estimate_utm_crs()
        logger.info(f"Reprojecting to projected CRS: {est}")
        return gdf.to_crs(est)
    return gdf


def _normalize_pollutant_column(df: pd.DataFrame, col: str, label: str, logger: Any) -> pd.DataFrame:
    """Normalize pollutant labels in a dataframe column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    col : str
        Name of the pollutant column.
    label : str
        Dataset label used in error messages.
    logger : Any
        Logger retained for interface consistency.

    Returns
    -------
    pandas.DataFrame
        Dataframe with standardized pollutant labels.

    Raises
    ------
    ValueError
        If the pollutant column is missing or cannot be normalized.
    """
    if col not in df.columns:
        raise ValueError(f"{label} missing required column '{col}'")
    try:
        df[col] = [normalize_pollutant_label(x) for x in df[col]]
    except Exception as ex:  # pylint: disable=broad-except
        raise ValueError(f"Failed to normalize pollutant labels in {label}: {ex}") from ex
    return df


def _normalize_pathway_label(value: Any) -> str:
    """Return a stable, user-extensible pathway label.

        Parameters
        ----------
        value : Any
            Input value to normalize or evaluate.

        Returns
        -------
        str
            Normalized pathway label.
        
    """
    label = str(value).strip().lower().replace("_", " ")
    return " ".join(label.split())


def _normalize_pathway_column(df: pd.DataFrame, label: str, logger: Any) -> pd.DataFrame:
    """Normalize pathway labels without restricting user-defined pathways.

        Statistical mode may use any non-empty pathway labels shared by the parcel
        load-rate and BMP efficiency inputs. PLET/RUSLE-specific pathway restrictions
        are applied later, after the load-generation mode is known.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        label : str
            Context label used in diagnostics and validation errors.
        logger : Any
            Logger used for diagnostic and progress messages.

        Returns
        -------
        pd.DataFrame
            Copy of the table with normalized pathway labels.

        Raises
        ------
        ValueError
            If any pathway label is blank after normalization.
        
    """
    del logger
    if COL_PATHWAY not in df.columns:
        return df
    df[COL_PATHWAY] = df[COL_PATHWAY].map(_normalize_pathway_label)
    bad = df[COL_PATHWAY].eq("")
    if bad.any():
        raise ValueError(f"{label} contains blank pathway labels")
    return df


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






def load_distribution_catalog(path: Any, logger: Any = None) -> Optional[pd.DataFrame]:
    """Read and validate the optional reusable input-distribution catalog.

        Parameters
        ----------
        path : Any
            Path or paths to reusable distribution-catalog CSV files.
        logger : Any
            Logger used for diagnostic and progress messages.

        Returns
        -------
        Optional[pd.DataFrame]
            Validated distribution catalog, or ``None`` when no catalog is configured.

        Raises
        ------
        ValueError
            If a catalog file is missing the required ``distribution_id`` column.
        
    """
    if path is None:
        return None
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    frames: List[pd.DataFrame] = []
    for item in paths:
        if logger is not None:
            logger.verbose(f"Reading reusable input distributions from {item}")
        frame = read_csv_table(item)
        frame = normalize_columns(frame)
        if DISTRIBUTION_ID not in frame.columns:
            raise ValueError(f"input_distributions ({item}) is missing required column '{DISTRIBUTION_ID}'")
        frames.append(frame)
    catalog = pd.concat(frames, ignore_index=True)
    validate_distribution_catalog(catalog)
    catalog[DISTRIBUTION_ID] = catalog[DISTRIBUTION_ID].astype(str).str.strip()
    return catalog.reset_index(drop=True)


def resolve_distribution_references(
    df: pd.DataFrame,
    catalog: Optional[pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    """Expand distribution references into inline statistics during input loading.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        catalog : Optional[pd.DataFrame]
            Reusable distribution catalog, if configured.
        label : str
            Context label used in diagnostics and validation errors.

        Returns
        -------
        pd.DataFrame
            Table with distribution references expanded to inline statistics.

        Raises
        ------
        ValueError
            If a distribution reference conflicts with inline statistics, is unknown, or cannot be resolved because no catalog is configured.
        
    """
    out = df.copy()
    if DISTRIBUTION_ID not in out.columns:
        return out
    catalog_map: Dict[str, pd.Series] = {}
    if catalog is not None:
        catalog_map = {str(row[DISTRIBUTION_ID]).strip(): row for _, row in catalog.iterrows()}
    all_stat_cols = set(statistic_columns(out.columns))
    if catalog is not None:
        all_stat_cols.update(statistic_columns(catalog.columns))
    for column in all_stat_cols:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for index, row in out.iterrows():
        ref = row.get(DISTRIBUTION_ID)
        if not _nonblank(ref):
            continue
        ref_id = str(ref).strip()
        inline = _row_stats_raw(row)
        if inline:
            raise ValueError(
                f"{label} row {index} specifies distribution_id={ref_id!r} and inline statistics; use one or the other"
            )
        if not catalog_map:
            raise ValueError(
                f"{label} row {index} references distribution_id={ref_id!r}, but no input_distributions catalog is configured"
            )
        if ref_id not in catalog_map:
            raise ValueError(f"{label} row {index} references unknown distribution_id={ref_id!r}")
        source = catalog_map[ref_id]
        for source_col in statistic_columns(source.index):
            value = source.get(source_col)
            if not pd.isna(value):
                out.at[index, source_col] = float(value)
        if "units" in source.index and ("units" not in out.columns or not _nonblank(row.get("units"))):
            if "units" not in out.columns:
                out["units"] = np.nan
            source_units = source.get("units")
            if _nonblank(source_units):
                out.at[index, "units"] = source_units
    return out







def _rows_for_pid(table: Optional[pd.DataFrame], pid: str) -> List[pd.Series]:
    """Resolve wildcard input rows plus parcel-specific overrides for one parcel.

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
    wildcard_rows = table[pids == "*"]
    exact_rows = table[pids == str(pid)]
    combined = pd.concat([wildcard_rows, exact_rows], ignore_index=True)
    if combined.empty:
        return []
    combined = combined.assign(
        _canonical_parameter=combined["parameter"].map(canonical_parameter_name)
    )
    combined = combined.drop_duplicates(subset=["_canonical_parameter"], keep="last")
    return [row for _, row in combined.iterrows()]




def _load_plet_hydrology_records(
    lookup_path: Optional[Union[str, Path]],
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Read and validate fixed PLET CN/infiltration records for deterministic helpers.

        Parameters
        ----------
        lookup_path : Optional[Union[str, Path]]
            Optional path to the PLET hydrology lookup table.

        Returns
        -------
        Dict[Tuple[str, str], Tuple[float, float]]
            Hydrology values keyed by normalized land-cover and HSG classes.

        Raises
        ------
        FileNotFoundError
            If the configured PLET hydrology lookup file does not exist.
        ValueError
            If the deterministic lookup is missing required columns or values, contains duplicate rows, or contains stochastic definitions.
        
    """
    from .plet_rusle import (
        PLET_HYDROLOGY_LOOKUP_PATH,
        _PLET_DERIVED_PARAMETERS,
        canonical_parameter_name,
        normalize_plet_hsg,
        normalize_plet_land_cover,
    )
    path = PLET_HYDROLOGY_LOOKUP_PATH if lookup_path is None else Path(lookup_path)
    if not path.exists():
        raise FileNotFoundError(f"PLET hydrology lookup table not found: {path}")
    table = read_csv_table(path)
    table = normalize_columns(table)
    records: Dict[Tuple[str, str], Dict[str, float]] = {}
    if {"land_cover", "hsg", "parameter"} <= set(table.columns):
        for row_index, row in table.iterrows():
            land_cover = normalize_plet_land_cover(row["land_cover"])
            hsg = normalize_plet_hsg(row["hsg"])
            parameter = canonical_parameter_name(row["parameter"])
            if parameter not in _PLET_DERIVED_PARAMETERS:
                continue
            stats = stats_from_row(row)
            if set(stats) != {"value"}:
                raise ValueError(
                    "The deterministic PLET hydrology helper requires fixed lookup values; "
                    f"row {row_index} for {land_cover}/{hsg}/{parameter} is stochastic"
                )
            records.setdefault((land_cover, hsg), {})[parameter] = float(stats["value"])
        output: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for key, values in records.items():
            if set(values) != set(_PLET_DERIVED_PARAMETERS):
                raise ValueError(f"PLET hydrology lookup is incomplete for {key}")
            output[key] = (float(values["cn"]), float(values["infiltration_fraction"]))
        return output
    required = {"land_cover", "hsg", "cn", "infiltration_fraction"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"PLET hydrology lookup is missing required columns: {missing}")
    output = {}
    for _, row in table.iterrows():
        key = (normalize_plet_land_cover(row["land_cover"]), normalize_plet_hsg(row["hsg"]))
        if key in output:
            raise ValueError(f"Duplicate PLET hydrology lookup row for {key}")
        output[key] = (float(row["cn"]), float(row["infiltration_fraction"]))
    return output


def _plet_parameter_defaults(pollutants: Sequence[str]) -> Dict[str, float]:
    """Return the centralized PLET/RUSLE parameter defaults.

        Parameters
        ----------
        pollutants : Sequence[str]
            Pollutant names in model order.

        Returns
        -------
        Dict[str, float]
            Centralized default PLET/RUSLE parameter values.
        
    """
    defaults: Dict[str, float] = {
        "annual_precip_in": 0.0,
        "rain_correction_fraction": 1.0,
        "ia_ratio": 0.0,
        "runoff_multiplier": 1.0,
        "groundwater_multiplier": 1.0,
        "sediment_multiplier": 1.0,
        "sediment_delivery_multiplier": 1.0,
        "enrichment_ratio": 2.0,
        "sediment_n_pct": 0.0,
        "sediment_p_pct": 0.0,
    }
    for pollutant in pollutants:
        defaults[f"load_multiplier_{str(pollutant).lower()}"] = 1.0
    return defaults


def apply_plet_parameter_defaults(
    parameters: Mapping[str, Any],
    pollutants: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return a parameter mapping with centralized PLET/RUSLE defaults applied.

        Parameters
        ----------
        parameters : Mapping[str, Any]
            Model parameter values keyed by canonical parameter name.
        pollutants : Sequence[str]
            Pollutant names in model order.

        Returns
        -------
        Dict[str, Any]
            Parameter mapping with centralized defaults applied.
        
    """
    out = dict(parameters)
    for parameter, value in _plet_parameter_defaults(pollutants).items():
        if parameter not in out or out[parameter] is None:
            out[parameter] = value
    return out


def _append_parameter_defaults(
    table: pd.DataFrame,
    pollutants: Sequence[str],
) -> pd.DataFrame:
    """Append wildcard PLET parameter defaults before scenario sampling.

        Parameters
        ----------
        table : pd.DataFrame
            Input table containing model data.
        pollutants : Sequence[str]
            Pollutant names in model order.

        Returns
        -------
        pd.DataFrame
            Table containing explicit wildcard default rows.
        
    """
    defaults = _plet_parameter_defaults(pollutants)
    out = table.copy()
    if "value" not in out.columns:
        out["value"] = np.nan
    existing_wildcards = set(
        out.loc[out[COL_PID].astype(str) == "*", "parameter"].astype(str).tolist()
    )
    rows: List[Dict[str, Any]] = []
    for parameter, value in defaults.items():
        if parameter in existing_wildcards:
            continue
        row = {column: np.nan for column in out.columns}
        row[COL_PID] = "*"
        row["parameter"] = parameter
        row["value"] = value
        rows.append(row)
    if rows:
        out = pd.concat([out, pd.DataFrame(rows, columns=out.columns)], ignore_index=True)
    return out




def _set_case_insensitive_default(mapping: Dict[str, Any], key: str, value: Any) -> None:
    """Set one input default while respecting existing case-insensitive keys.

        Parameters
        ----------
        mapping : Dict[str, Any]
            Input mapping.
        key : str
            Configuration or mapping key.
        value : Any
            Input value to normalize or evaluate.
        
    """
    matching_key = next((existing for existing in mapping if str(existing).lower() == key.lower()), None)
    if matching_key is None:
        mapping[key] = value
    elif mapping[matching_key] is None:
        mapping[matching_key] = value


def apply_config_defaults(cfg: Dict[str, Any]) -> None:
    """Apply all supported top-level configuration defaults in one place.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Normalized model configuration mapping.
        
    """
    _set_case_insensitive_default(cfg, CFG_N_SCENARIOS, 1)
    _set_case_insensitive_default(cfg, CFG_OUTPUTS, "./outputs")
    _set_case_insensitive_default(cfg, CFG_VERBOSE, False)
    _set_case_insensitive_default(cfg, CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT)
    _set_case_insensitive_default(cfg, CFG_BMP_SEL_PROB_VIA_COSTS, False)
    _set_case_insensitive_default(cfg, CFG_BMP_FAIL_RATE, 0.0)
    _set_case_insensitive_default(cfg, CFG_BMP_FAIL_REDUCTION, DEFAULT_BMP_FAIL_REDUCTION)
    _set_case_insensitive_default(cfg, CFG_PARALLEL, {"n_jobs": 1})
    _set_case_insensitive_default(cfg, CFG_LOAD_GENERATION, {})




def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configuration keys and apply centralized defaults in place.

    Parameters
    ----------
    cfg : dict[str, Any]
        Raw configuration mapping, typically returned by
        :func:`src.io_utils.read_config`.

    Returns
    -------
    dict[str, Any]
        The same mapping object after top-level key normalization and default
        application.
    """
    normalized = {str(key).lower(): value for key, value in cfg.items()}
    cfg.clear()
    cfg.update(normalized)
    apply_config_defaults(cfg)
    parallel = ci_get(cfg, CFG_PARALLEL)
    if isinstance(parallel, dict):
        normalized_parallel = {str(key).lower(): value for key, value in parallel.items()}
        parallel.clear()
        parallel.update(normalized_parallel)
        if "n_jobs" not in parallel or parallel["n_jobs"] is None:
            parallel["n_jobs"] = 1
    return cfg



def _load_parameter_stats_table(path: Any, label: str, logger: Any, distribution_catalog: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Load a parcel parameter statistics table.

        Parameters
        ----------
        path : Any
            CSV file path or sequence of paths.
        label : str
            Dataset label used in logs and errors.
        logger : Any
            Logger used for progress reporting.
        distribution_catalog : Optional[pd.DataFrame]
            Reusable distribution catalog used to resolve referenced statistics.

        Returns
        -------
        pandas.DataFrame or None
            Loaded parameter statistics table, or ``None`` when no path is
            provided.
        
    """
    if path is None:
        return None
    from .plet_rusle import canonical_parameter_name
    df = _merge_csvs(path, [COL_PID, "parameter"], label, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df["parameter"] = df["parameter"].map(canonical_parameter_name)
    validate_unique_rows(df, [COL_PID, "parameter"], label)
    df = resolve_distribution_references(df, distribution_catalog, label)
    validate_stats_rows(df, label)
    return df


def _load_plet_parameter_table(
    path: Any,
    parcel_ids: Sequence[str],
    logger: Any,
    distribution_catalog: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Load PLET numeric parameters and required categorical classifications.

        Unlike the general parameter loader, this function permits fixed string
        values for ``land_cover`` and ``hsg`` while retaining the existing
        distribution-statistics behavior for all numeric parameters.

        Parameters
        ----------
        path : Any
            CSV file path or sequence of paths.
        parcel_ids : sequence of str
            Parcel identifiers that may be selected by the model.
        logger : Any
            Logger used for progress reporting.
        distribution_catalog : Optional[pd.DataFrame]
            Reusable distribution catalog used to resolve referenced statistics.

        Returns
        -------
        pandas.DataFrame or None
            Validated PLET parameter table, or ``None`` when no path is provided.

        Raises
        ------
        ValueError
            If fixed PLET land-cover or HSG classifications are supplied as distributions.
        
    """

    if path is None:
        return None
    from .plet_rusle import canonical_parameter_name, PLET_CLASSIFICATION_PARAMETERS
    df = _merge_csvs(path, [COL_PID, "parameter"], LOAD_PLET_INPUTS, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df["parameter"] = df["parameter"].map(canonical_parameter_name)
    validate_unique_rows(df, [COL_PID, "parameter"], LOAD_PLET_INPUTS)

    categorical_mask = df["parameter"].isin(PLET_CLASSIFICATION_PARAMETERS)
    categorical_rows = df.loc[categorical_mask].copy()
    if DISTRIBUTION_ID in categorical_rows.columns:
        bad = categorical_rows[DISTRIBUTION_ID].notna() & categorical_rows[DISTRIBUTION_ID].astype(str).str.strip().ne("")
        if bad.any():
            raise ValueError("PLET land_cover and hsg are classifications and must use fixed value, not distribution_id")
    # Classification rows are intentionally deterministic. Reject numeric
    # distribution columns rather than silently ignoring them.
    categorical_stat_cols = [
        col for col in statistic_columns(categorical_rows.columns) if str(col).strip().lower() != "value"
    ]
    if categorical_stat_cols and not categorical_rows.empty:
        bad_stats = categorical_rows[categorical_stat_cols].notna().any(axis=1)
        if bad_stats.any():
            rows = categorical_rows.index[bad_stats].tolist()
            raise ValueError(
                "PLET land_cover and hsg are classifications and must use only a fixed value; "
                f"distribution statistics were supplied at rows {rows}"
            )
    numeric_rows = df.loc[~categorical_mask].copy()
    if not numeric_rows.empty:
        numeric_rows = resolve_distribution_references(
            numeric_rows, distribution_catalog, LOAD_PLET_INPUTS
        )
        validate_stats_rows(numeric_rows, LOAD_PLET_INPUTS)
    df = pd.concat([categorical_rows, numeric_rows], axis=0).sort_index()
    return validate_plet_input_table(df, parcel_ids)


def _load_plet_hydrology_lookup(
    path: Any,
    logger: Any,
    distribution_catalog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load required land-cover/HSG hydrology distributions for PLET mode.

        The table is long-form with one row per ``land_cover`` x ``hsg`` x
        ``parameter``. Exactly two parameters are required for every supported
        pairing: ``cn`` and ``infiltration_fraction``. Each row follows the same
        fixed-value/distribution schema as other numeric model inputs.

        Parameters
        ----------
        path : Any
            Path to the required PLET hydrology lookup CSV file.
        logger : Any
            Logger used for diagnostic and progress messages.
        distribution_catalog : Optional[pd.DataFrame]
            Reusable distribution catalog used to resolve referenced statistics.

        Returns
        -------
        pd.DataFrame
            Validated PLET hydrology lookup table.

        Raises
        ------
        ValueError
            If the required hydrology lookup is absent, contains unsupported parameters, or lacks required land-cover/HSG coverage.
        
    """
    if path is None:
        raise ValueError(
            "load_generation.hydrology_lookup is required for mode='plet_rusle'"
        )
    from .plet_rusle import (
        PLET_HSG_VALUES,
        PLET_LAND_COVERS,
        canonical_parameter_name,
        normalize_plet_hsg,
        normalize_plet_land_cover,
    )
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    frames: List[pd.DataFrame] = []
    for item in paths:
        logger.verbose(f"Reading {LOAD_HYDROLOGY_LOOKUP} from {item}")
        frame = read_csv_table(item)
        frame = normalize_columns(frame)
        require_columns(
            frame,
            ["land_cover", "hsg", "parameter"],
            f"{LOAD_HYDROLOGY_LOOKUP} ({item})",
            logger,
        )
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["land_cover"] = table["land_cover"].map(normalize_plet_land_cover)
    table["hsg"] = table["hsg"].map(normalize_plet_hsg)
    table["parameter"] = table["parameter"].map(canonical_parameter_name)

    allowed_parameters = {"cn", "infiltration_fraction"}
    unexpected_parameters = sorted(set(table["parameter"]) - allowed_parameters)
    if unexpected_parameters:
        raise ValueError(
            f"{LOAD_HYDROLOGY_LOOKUP} contains unsupported parameters: "
            f"{unexpected_parameters}; expected only cn and infiltration_fraction"
        )

    table = resolve_distribution_references(
        table, distribution_catalog, LOAD_HYDROLOGY_LOOKUP
    )
    validate_stats_rows(table, LOAD_HYDROLOGY_LOOKUP)
    validate_unique_rows(
        table, ["land_cover", "hsg", "parameter"], LOAD_HYDROLOGY_LOOKUP
    )

    expected = {
        (land_cover, hsg, parameter)
        for land_cover in PLET_LAND_COVERS
        for hsg in PLET_HSG_VALUES
        for parameter in ("cn", "infiltration_fraction")
    }
    supplied = set(
        zip(table["land_cover"], table["hsg"], table["parameter"])
    )
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise ValueError(
            f"{LOAD_HYDROLOGY_LOOKUP} must define cn and infiltration_fraction "
            "for every supported land_cover x hsg pairing; "
            f"missing={missing}, unexpected={extra}"
        )

    validate_distribution_bounds(
        table,
        LOAD_HYDROLOGY_LOOKUP,
        parameter_col="parameter",
        bounds={
            "cn": (1.0e-9, 100.0),
            "infiltration_fraction": (0.0, 1.0),
        },
    )
    return table.reset_index(drop=True)


def _load_pollutant_concentrations(path: Any, pollutants: List[str], logger: Any, distribution_catalog: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Load parcel pollutant concentration inputs.

        Parameters
        ----------
        path : Any
            CSV file path or sequence of paths.
        pollutants : list[str]
            Pollutant names to retain.
        logger : Any
            Logger used for progress reporting.
        distribution_catalog : Optional[pd.DataFrame]
            Reusable distribution catalog used to resolve referenced statistics.

        Returns
        -------
        pandas.DataFrame or None
            Filtered pollutant concentration table, or ``None`` when no path is
            provided.
        
    """
    if path is None:
        return None
    df = _merge_csvs(path, [COL_PID, COL_POLLUTANT], LOAD_CONCENTRATIONS, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, LOAD_CONCENTRATIONS, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df = df[df[COL_POLLUTANT].isin(pollutants)].copy()
    validate_unique_rows(df, [COL_PID, COL_POLLUTANT], LOAD_CONCENTRATIONS)
    df = resolve_distribution_references(df, distribution_catalog, LOAD_CONCENTRATIONS)
    validate_stats_rows(df, LOAD_CONCENTRATIONS)
    return df


def _load_groundwater_concentrations(path: Any, pollutants: List[str], logger: Any, distribution_catalog: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Load optional parcel groundwater concentration inputs.

        Parameters
        ----------
        path : Any
            CSV file path or sequence of paths.
        pollutants : list[str]
            Pollutant names to retain.
        logger : Any
            Logger used for progress reporting.
        distribution_catalog : Optional[pd.DataFrame]
            Reusable distribution catalog used to resolve referenced statistics.

        Returns
        -------
        pandas.DataFrame or None
            Filtered groundwater concentration table, or ``None`` when no path is
            provided.
        
    """
    if path is None:
        return None
    df = _merge_csvs(path, [COL_PID, COL_POLLUTANT], LOAD_GROUNDWATER_CONCENTRATIONS, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, LOAD_GROUNDWATER_CONCENTRATIONS, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df = df[df[COL_POLLUTANT].isin(pollutants)].copy()
    validate_unique_rows(df, [COL_PID, COL_POLLUTANT], LOAD_GROUNDWATER_CONCENTRATIONS)
    df = resolve_distribution_references(df, distribution_catalog, LOAD_GROUNDWATER_CONCENTRATIONS)
    validate_stats_rows(df, LOAD_GROUNDWATER_CONCENTRATIONS)
    return df



def _load_pollutants(cfg: Dict[str, Any]) -> List[str]:
    """Load pollutant names from configuration.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.

    Returns
    -------
    list[str]
        Normalized pollutant names.
    """
    pols = ci_get(cfg, CFG_POLLUTANTS)
    if isinstance(pols, str):
        pols = [pols]
    if not pols:
        raise ValueError(f"At least one {CFG_POLLUTANTS} value must be specified")
    return [normalize_pollutant_label(p) for p in pols]


def _load_cps(cfg: Dict[str, Any]) -> List[int]:
    """Load BMP CPS codes from configuration.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.

    Returns
    -------
    list[int]
        BMP CPS codes as integers.
    """
    cps = ci_get(cfg, CFG_CPS)
    if isinstance(cps, int):
        cps = [cps]
    if not cps:
        raise ValueError("At least one cps code must be specified")
    return [int(c) for c in cps]


def _load_domain(cfg: Dict[str, Any], logger: Any) -> gpd.GeoDataFrame:
    """Load and normalize the model domain boundary.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    geopandas.GeoDataFrame
        Domain boundary in a projected CRS with lowercase columns.
    """
    domain_path = Path(ci_get(cfg, CFG_DOMAIN))
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    domain = read_geodataframe(domain_path)
    domain = _ensure_projected(domain, logger)
    return domain.rename(columns={c: c.lower() for c in domain.columns})


def _load_parcels(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load parcels, clip them to the domain, and compute geometry metrics.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    domain : geopandas.GeoDataFrame
        Domain boundary used for clipping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    geopandas.GeoDataFrame
        Parcel dataframe with area and perimeter columns.

    Raises
    ------
    ValueError
        If the parcel file is missing required columns, becomes empty after
        clipping, or contains duplicate parcel IDs.
    """
    parcels_path = Path(ci_get(cfg, CFG_PARCELS))
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path}")
    parcels = read_geodataframe(parcels_path)
    parcels = _ensure_projected(parcels, logger)
    parcels = gpd.overlay(parcels, domain, how="intersection")
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if "pid" not in parcels.columns:
        raise ValueError("Parcels must include a 'pid' column")
    if parcels.empty:
        raise ValueError("No parcels remain after clipping to the domain")
    if parcels["pid"].astype(str).duplicated().any():
        dup_pids = sorted(parcels.loc[parcels["pid"].astype(str).duplicated(), "pid"].astype(str).unique().tolist())
        raise ValueError(f"Parcel IDs must be unique after clipping; duplicates found: {dup_pids}")
    parcels["area_m2"] = parcels.geometry.area
    parcels["perim_m"] = parcels.geometry.length
    parcels["area_ha"] = parcels["area_m2"] / 10000.0
    validate_numeric_columns_in_domain(
        parcels, ["area_m2", "area_ha", "perim_m"], POSITIVE_DOMAIN, CFG_PARCELS
    )
    return parcels


def _load_parcel_graph(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load parcel-to-parcel upstream relationships.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame
        Table describing which parcels flow into which others.
    """
    up_path = Path(ci_get(cfg, CFG_PARCEL_UP))
    if not up_path.exists():
        raise FileNotFoundError(f"{CFG_PARCEL_UP} not found: {up_path}")
    df = _merge_csvs(up_path, [COL_PID, COL_PID_UP], CFG_PARCEL_UP, logger)
    return df


def _build_parcel_up_map(
    upstream_rows: pd.DataFrame,
    parcel_ids: Sequence[str],
) -> Dict[str, List[str]]:
    """Build a validated mapping of parcels to upstream parcel IDs.

    Each ``pid_up`` cell may contain one ID, a comma-separated list of IDs, or
    no value. A single ``pid='*'`` row with a blank ``pid_up`` may be used to
    declare the watershed-wide default of no upstream parcels; explicit parcel
    rows then provide exceptions. IDs are stripped of surrounding whitespace,
    deduplicated while preserving their input order, and checked against the
    loaded parcel set.

    Parameters
    ----------
    upstream_rows : pandas.DataFrame
        Parcel graph table containing ``pid`` and ``pid_up`` columns.
    parcel_ids : sequence of str
        Valid parcel IDs after the parcel layer has been clipped to the model
        domain.

    Returns
    -------
    dict[str, list[str]]
        Upstream parcel IDs keyed by receiving parcel ID.

    Raises
    ------
    ValueError
        If a receiving or upstream parcel ID is missing from the loaded parcel
        set, a graph row has a blank receiving parcel ID, or a wildcard row is
        malformed.
    """
    ordered_pids = [str(pid).strip() for pid in parcel_ids]
    valid_pids = set(ordered_pids)
    parcel_up_map: Dict[str, List[str]] = {pid: [] for pid in ordered_pids}
    seen_by_pid = {pid: set() for pid in ordered_pids}
    unknown_pids = set()
    wildcard_default_seen = False

    def resolve_pid(value: Any) -> str:
        """Match numeric CSV values such as ``4.0`` to parcel ID ``4``.

                Parameters
                ----------
                value : Any
                    Input value to normalize or evaluate.

                Returns
                -------
                str
                    Normalized parcel identifier.
                
        """
        pid = str(value).strip()
        if pid in valid_pids:
            return pid
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
            integer_pid = str(int(value))
            if integer_pid in valid_pids:
                return integer_pid
        return pid

    for row_idx, row in upstream_rows.iterrows():
        raw_pid = row[COL_PID]
        if pd.isna(raw_pid) or not str(raw_pid).strip():
            raise ValueError(f"{CFG_PARCEL_UP} row {row_idx} has a blank {COL_PID}")

        raw_upstream = row[COL_PID_UP]
        pid_text = str(raw_pid).strip()
        if pid_text == "*":
            # A wildcard is intentionally limited to the unambiguous default
            # case: parcels have no upstream parcels unless an exact pid row
            # says otherwise. The map is initialized to [] for every parcel,
            # so no physical expansion of the wildcard row is required.
            upstream_is_blank = pd.isna(raw_upstream) or not str(raw_upstream).strip()
            if not upstream_is_blank:
                raise ValueError(
                    f"{CFG_PARCEL_UP} row {row_idx} uses pid='*' but {COL_PID_UP} is not blank. "
                    "The parcel_up wildcard may only declare the default of no upstream parcels; "
                    "actual upstream relationships must use explicit parcel IDs."
                )
            if wildcard_default_seen:
                raise ValueError(
                    f"{CFG_PARCEL_UP} may contain at most one pid='*' row with a blank {COL_PID_UP}"
                )
            wildcard_default_seen = True
            continue

        pid = resolve_pid(raw_pid)
        if pid not in valid_pids:
            unknown_pids.add(pid)
            continue

        if pd.isna(raw_upstream):
            continue

        if isinstance(raw_upstream, (int, float, np.integer, np.floating)):
            raw_upstream_pids = [raw_upstream]
        else:
            raw_upstream_pids = str(raw_upstream).split(",")

        for raw_upstream_pid in raw_upstream_pids:
            upstream_pid = resolve_pid(raw_upstream_pid)
            if not upstream_pid:
                continue
            if upstream_pid not in valid_pids:
                unknown_pids.add(upstream_pid)
                continue
            if upstream_pid not in seen_by_pid[pid]:
                parcel_up_map[pid].append(upstream_pid)
                seen_by_pid[pid].add(upstream_pid)

    if unknown_pids:
        unknown = sorted(unknown_pids)
        preview = unknown[:10]
        suffix = f" (and {len(unknown) - len(preview)} more)" if len(unknown) > len(preview) else ""
        raise ValueError(
            f"{CFG_PARCEL_UP} references parcel IDs not found in parcels after clipping: "
            f"{preview}{suffix}"
        )

    return parcel_up_map


def _expand_pid_defaults(
    df: pd.DataFrame,
    parcel_ids: Sequence[str],
    *,
    label: str,
    logger: Any,
    key_columns: Optional[Sequence[str]] = (),
) -> pd.DataFrame:
    """Expand ``pid='*'`` defaults to modeled parcels.

        Exact parcel rows override wildcard defaults. When ``key_columns`` is an
        empty sequence, at most one row may be defined for the wildcard and for
        each exact parcel (used by ``parcel_p``). When one or more key columns are
        supplied, overrides are resolved independently for each key combination
        (used by ``delivery_ratios`` with ``oid``). When ``key_columns`` is
        ``None``, rows are treated as a parcel-level group: if a parcel has any
        explicit rows, those rows replace the wildcard group entirely (used by
        ``parcel_out``).

        Rows whose exact parcel IDs are not present in the clipped parcel layer are
        removed with a warning. If no wildcard row is present, the function simply
        returns the valid exact rows, preserving subset behavior.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        parcel_ids : Sequence[str]
            Parcel identifiers in model order.
        label : str
            Context label used in diagnostics and validation errors.
        logger : Any
            Logger used for diagnostic and progress messages.
        key_columns : Optional[Sequence[str]]
            Columns that, together with parcel ID, identify an input row.

        Returns
        -------
        pd.DataFrame
            Table with wildcard parcel defaults expanded to modeled parcels.

        Raises
        ------
        ValueError
            If wildcard defaults are ambiguous or parcel-specific rows are duplicated after expansion.
        
    """
    out = df.copy()
    out[COL_PID] = out[COL_PID].astype(str).str.strip()
    ordered_pids = [str(pid).strip() for pid in parcel_ids]
    valid_pids = set(ordered_pids)
    valid_mask = out[COL_PID].isin(valid_pids | {"*"})
    removed = out.loc[~valid_mask]
    if not removed.empty:
        preview = removed[COL_PID].astype(str).drop_duplicates().head(10).tolist()
        logger.warning(
            f"{label}: some PIDs not found in parcels after clipping; they were removed. "
            f"Example PIDs: {preview}"
        )
    out = out.loc[valid_mask].copy()
    if out.empty:
        return out.reset_index(drop=True)

    defaults = out[out[COL_PID] == "*"].copy()
    exact = out[out[COL_PID] != "*"].copy()

    # Validate exact/default row uniqueness even when no wildcard row exists.
    if key_columns is not None:
        keys = list(key_columns)
        if keys:
            if not defaults.empty:
                validate_unique_rows(defaults, keys, label)
            validate_unique_rows(exact, [COL_PID, *keys], label)
        else:
            if len(defaults) > 1:
                raise ValueError(f"{label} may contain at most one pid='*' default row")
            if exact[COL_PID].duplicated().any():
                dup_pids = sorted(
                    exact.loc[exact[COL_PID].duplicated(keep=False), COL_PID]
                    .astype(str)
                    .unique()
                    .tolist()
                )
                raise ValueError(
                    f"{label} must contain one row per parcel; duplicates found: {dup_pids}"
                )

    if defaults.empty:
        return exact.reset_index(drop=True)

    expanded: List[pd.Series] = []
    if key_columns is None:
        for pid in ordered_pids:
            pid_rows = exact[exact[COL_PID] == pid]
            source = pid_rows if not pid_rows.empty else defaults
            for _, row in source.iterrows():
                copied = row.copy()
                copied[COL_PID] = pid
                expanded.append(copied)
    else:
        keys = list(key_columns)

        def _key(row: pd.Series) -> Tuple[Any, ...]:
            """Return a stable composite key for an input row.

                        Parameters
                        ----------
                        row : pd.Series
                            Input table row.

                        Returns
                        -------
                        Tuple[Any, ...]
                            Composite key used to identify an input row.
                        
            """
            return tuple(row[column] for column in keys)

        for pid in ordered_pids:
            pid_rows = exact[exact[COL_PID] == pid]
            exact_keys = {_key(row) for _, row in pid_rows.iterrows()}
            for _, row in defaults.iterrows():
                if _key(row) in exact_keys:
                    continue
                copied = row.copy()
                copied[COL_PID] = pid
                expanded.append(copied)
            for _, row in pid_rows.iterrows():
                expanded.append(row.copy())

    if not expanded:
        return out.iloc[0:0].copy().reset_index(drop=True)
    return pd.DataFrame(expanded, columns=out.columns).reset_index(drop=True)

def _load_parcel_outlets(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load parcel-to-outlet relationships.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame
        Table describing which outlets each parcel drains to.
    """
    out_path = Path(ci_get(cfg, CFG_PARCEL_OUT))
    if not out_path.exists():
        raise FileNotFoundError(f"{CFG_PARCEL_OUT} not found: {out_path}")
    df = _merge_csvs(out_path, [COL_PID, COL_OIDS], CFG_PARCEL_OUT, logger)
    return df


def _load_parcel_selection(cfg: Dict[str, Any], parcels: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Load or synthesize parcel selection probabilities.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    parcels : pandas.DataFrame
        Parcel table used to determine available parcel IDs.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame
        Parcel IDs and normalized selection probabilities.
    """
    if parcels.empty:
        raise ValueError("No parcels available for selection")
    p_cfg = ci_get(cfg, CFG_PARCEL_P)
    if p_cfg is not None:
        df = _merge_csvs(p_cfg, [COL_PID, COL_PROBABILITY], CFG_PARCEL_P, logger)
        probs = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")
        invalid = (~np.isfinite(probs)) | (probs < 0.0)
        if invalid.any():
            bad_rows = df.loc[invalid, [COL_PID, COL_PROBABILITY]]
            preview = bad_rows.head(5).to_dict(orient="records")
            raise ValueError(
                f"{CFG_PARCEL_P} contains invalid probability values (must be finite and >= 0). "
                f"Example rows: {preview}"
            )
        df[COL_PROBABILITY] = probs.astype(float)
        df = _expand_pid_defaults(
            df,
            parcels[COL_PID].astype(str).tolist(),
            label=CFG_PARCEL_P,
            logger=logger,
            key_columns=[],
        )
        if df.empty:
            raise ValueError(f"{CFG_PARCEL_P} has no {COL_PID}s that exist in parcels after filtering")
        total_prob = df[COL_PROBABILITY].sum()
        if total_prob <= 0:
            raise ValueError(f"{CFG_PARCEL_P} selection weights sum to zero or negative")
        df[COL_PROBABILITY] /= total_prob
        return df[[COL_PID, COL_PROBABILITY]].copy()
    # synthesize uniform
    return pd.DataFrame({COL_PID: parcels[COL_PID].values, COL_PROBABILITY: np.full(len(parcels), 1 / len(parcels))})


def _load_outlet_loc(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load outlet locations and align them to the domain CRS.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    domain : geopandas.GeoDataFrame
        Domain boundary whose CRS is used for alignment.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    geopandas.GeoDataFrame
        Outlet location dataframe in the domain CRS.
    """
    outlet_path = Path(ci_get(cfg, CFG_OUTLET_LOC))
    if not outlet_path.exists():
        raise FileNotFoundError(f"Outlet location not found: {outlet_path}")
    outlet_loc = read_geodataframe(outlet_path).to_crs(domain.crs)
    outlet_loc = outlet_loc.rename(columns={c: c.lower() for c in outlet_loc.columns})
    require_columns(outlet_loc, [COL_OID], CFG_OUTLET_LOC, logger)
    return outlet_loc


def _load_optional_outlet_stats(
    cfg: Dict[str, Any],
    key: str,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Optionally load an outlet summary table.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    key : str
        Configuration key for the table path.
    required_cols : sequence of str
        Columns that must be present in the table.
    label : str
        Dataset label used in logs and errors.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame or None
        Loaded table with normalized pollutant names, or ``None`` when the
        configuration key is absent.
    """
    if ci_get(cfg, key) is None:
        logger.verbose(f"Optional key {key} not provided; skipping {label}")
        return None
    df = _merge_csvs(ci_get(cfg, key), required_cols, label, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, label, logger)
    numeric_columns = [
        column for column in (COL_TARGET, COL_MEAN) if column in df.columns
    ]
    validate_numeric_columns_in_domain(
        df, numeric_columns, NONNEGATIVE_DOMAIN, label
    )
    return df


def _load_delivery_ratios(cfg: Dict[str, Any], logger: Any) -> Optional[pd.DataFrame]:
    """Optionally load parcel-to-outlet delivery ratios.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress and warning messages.

    Returns
    -------
    pandas.DataFrame or None
        Delivery ratio table, or ``None`` when not configured or missing.
    """
    dr_cfg = ci_get(cfg, CFG_DELIVERY_RATIOS)
    if dr_cfg is None:
        logger.verbose("No delivery ratios configured; using default delivery coefficients")
        return None
    dr_path = Path(dr_cfg)
    if not dr_path.exists():
        logger.warning(f"{CFG_DELIVERY_RATIOS} specified but file not found: {dr_cfg}; skipping delivery ratios")
        return None
    return _merge_csvs(
        dr_cfg,
        [COL_PID, COL_OID, "sdr_f_to_s", "sdr_s_to_o", "ndr_f_to_s", "ndr_s_to_o"],
        CFG_DELIVERY_RATIOS,
        logger,
    )


def _efficiency_stat_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that can define an efficiency distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        BMP efficiency input table.

    Returns
    -------
    list[str]
        Statistic columns present in the table.
    """
    named_stats = {
        "value",
        "mean",
        "average",
        "avg",
        "sd",
        "std",
        "min",
        "minimum",
        "max",
        "maximum",
        "p0",
        "p100",
    }
    return [
        col
        for col in df.columns
        if str(col).lower() in named_stats
        or (
            str(col).lower().startswith("p")
            and str(col).lower()[1:].isdigit()
        )
    ]


def _complete_bmp_efficiency_coverage(
    df: pd.DataFrame,
    cps: Sequence[int],
    pollutants: Sequence[str],
    logger: Any,
) -> pd.DataFrame:
    """Legacy three-path completion used by the public loader API.

        Surface is required. Missing shallow/deep subsurface values are completed
        as fixed zero distributions with verbose logging. Production mode-specific
        validation bypasses this compatibility layer.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        cps : Sequence[int]
            Conservation Practice Standard (CPS) code or codes.
        pollutants : Sequence[str]
            Pollutant names in model order.
        logger : Any
            Logger used for diagnostic and progress messages.

        Returns
        -------
        pd.DataFrame
            BMP efficiency table with complete legacy pathway coverage.

        Raises
        ------
        ValueError
            If required surface-efficiency coverage is missing for a configured CPS/pollutant combination.
        
    """
    completed = df.copy()
    completed[COL_CPS] = completed[COL_CPS].astype(int)
    if COL_PATHWAY not in completed.columns:
        completed[COL_PATHWAY] = PATHWAY_VALUES[0]
        logger.verbose(
            "bmp_efficiency has no pathway column; treating supplied "
            "CPS x pollutant efficiencies as surface efficiencies"
        )

    configured_cps = [int(cps_code) for cps_code in cps]
    configured_pollutants = [str(pollutant) for pollutant in pollutants]
    surface_pathway = PATHWAY_VALUES[0]
    subsurface_pathways = PATHWAY_VALUES[1:]
    missing_surface: List[Tuple[int, str]] = []
    for cps_code in configured_cps:
        for pollutant in configured_pollutants:
            mask = (
                (completed[COL_CPS] == cps_code)
                & (completed[COL_POLLUTANT] == pollutant)
                & (completed[COL_PATHWAY] == surface_pathway)
            )
            if not mask.any():
                missing_surface.append((cps_code, pollutant))
    if missing_surface:
        details = ", ".join(
            f"cps={cps_code}, pollutant={pollutant}"
            for cps_code, pollutant in missing_surface
        )
        raise ValueError(
            "bmp_efficiency is missing required surface efficiency coverage "
            f"for configured CPS x pollutant combinations: {details}"
        )

    stat_columns = _efficiency_stat_columns(completed)
    added_rows: List[pd.Series] = []
    for cps_code in configured_cps:
        for pollutant in configured_pollutants:
            pair_mask = (
                (completed[COL_CPS] == cps_code)
                & (completed[COL_POLLUTANT] == pollutant)
            )
            surface_row = completed[
                pair_mask & (completed[COL_PATHWAY] == surface_pathway)
            ].iloc[0]
            for pathway in subsurface_pathways:
                pathway_mask = pair_mask & (completed[COL_PATHWAY] == pathway)
                if pathway_mask.any():
                    row_index = completed[pathway_mask].index[0]
                    has_values = any(
                        not pd.isna(completed.at[row_index, column])
                        for column in stat_columns
                    )
                    if has_values:
                        continue
                    for column in stat_columns:
                        completed.at[row_index, column] = 0.0
                else:
                    default_row = surface_row.copy()
                    default_row[COL_PATHWAY] = pathway
                    for column in stat_columns:
                        default_row[column] = 0.0
                    added_rows.append(default_row)
                logger.verbose(
                    "No bmp_efficiency value specified for "
                    f"cps={cps_code}, pollutant={pollutant}, pathway='{pathway}'; "
                    "assuming efficiency=0"
                )

    if added_rows:
        completed = pd.concat(
            [completed, pd.DataFrame(added_rows, columns=completed.columns)],
            ignore_index=True,
        )

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)
    pathway_order = {pathway: idx for idx, pathway in enumerate(PATHWAY_VALUES)}
    completed["_pathway_order"] = completed[COL_PATHWAY].map(pathway_order)
    completed = completed.sort_values(
        [COL_CPS, COL_POLLUTANT, "_pathway_order"], kind="stable"
    ).drop(columns="_pathway_order")
    return completed.reset_index(drop=True)




def _complete_plet_bmp_efficiency_coverage(
    df: pd.DataFrame, cps: Sequence[int], pollutants: Sequence[str], logger: Any
) -> pd.DataFrame:
    """Require PLET surface efficiencies and default missing subsurface to zero.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        cps : Sequence[int]
            Conservation Practice Standard (CPS) code or codes.
        pollutants : Sequence[str]
            Pollutant names in model order.
        logger : Any
            Logger used for diagnostic and progress messages.

        Returns
        -------
        pd.DataFrame
            BMP efficiency table with required PLET pathway coverage.

        Raises
        ------
        ValueError
            If PLET/RUSLE surface-efficiency coverage is missing for a configured CPS/pollutant combination.
        
    """
    completed = df.copy()
    completed[COL_CPS] = completed[COL_CPS].astype(int)
    if COL_PATHWAY not in completed.columns:
        completed[COL_PATHWAY] = "surface"
        logger.verbose(
            "plet_rusle bmp_efficiency has no pathway column; treating supplied "
            "CPS x pollutant rows as surface efficiencies"
        )
    else:
        unexpected = sorted(
            set(completed[COL_PATHWAY].astype(str)) - set(PLET_PATHWAY_VALUES)
        )
        if unexpected:
            logger.warning(
                "plet_rusle recognizes only pathway labels 'surface' and "
                f"'subsurface'. Ignoring unexpected bmp_efficiency pathway labels: {unexpected}. "
                "If no correctly labeled subsurface efficiency remains for a CPS/pollutant, "
                "subsurface efficiency will be assumed to be 0."
            )
        completed = completed[completed[COL_PATHWAY].isin(PLET_PATHWAY_VALUES)].copy()

    validate_unique_rows(
        completed, [COL_CPS, COL_POLLUTANT, COL_PATHWAY], CFG_BMP_EFFICIENCY
    )
    stat_columns = _efficiency_stat_columns(completed)
    added_rows: List[pd.Series] = []
    missing_surface: List[Tuple[int, str]] = []
    for cps_code in [int(x) for x in cps]:
        for pollutant in [str(x) for x in pollutants]:
            pair = (completed[COL_CPS] == cps_code) & (completed[COL_POLLUTANT] == pollutant)
            surf = completed[pair & (completed[COL_PATHWAY] == "surface")]
            if surf.empty:
                missing_surface.append((cps_code, pollutant))
                continue
            sub = completed[pair & (completed[COL_PATHWAY] == "subsurface")]
            sub_has_stats = False
            if not sub.empty:
                row = sub.iloc[0]
                sub_has_stats = any(not pd.isna(row.get(c)) for c in stat_columns)
            if not sub_has_stats:
                template = surf.iloc[0].copy()
                template[COL_PATHWAY] = "subsurface"
                for c in stat_columns:
                    template[c] = 0.0
                if not sub.empty:
                    completed = completed.drop(index=sub.index)
                added_rows.append(template)
                logger.warning(
                    "plet_rusle: no correctly labeled subsurface BMP efficiency was "
                    f"defined for cps={cps_code}, pollutant={pollutant}; assuming efficiency=0"
                )
    if missing_surface:
        details = ", ".join(f"cps={c}, pollutant={p}" for c, p in missing_surface)
        raise ValueError(
            "plet_rusle requires a surface bmp_efficiency for every configured "
            f"CPS x pollutant combination; missing: {details}"
        )
    if added_rows:
        completed = pd.concat([completed, pd.DataFrame(added_rows)], ignore_index=True)
    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)
    order = {"surface": 0, "subsurface": 1}
    completed["_pathway_order"] = completed[COL_PATHWAY].map(order)
    return completed.sort_values(
        [COL_CPS, COL_POLLUTANT, "_pathway_order"], kind="stable"
    ).drop(columns="_pathway_order").reset_index(drop=True)




def _load_bmp_efficiency(
    cfg: Dict[str, Any],
    cps: List[int],
    pollutants: List[str],
    logger: Any,
    *,
    complete_legacy: bool = True,
    distribution_catalog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load and normalize BMP effectiveness inputs.

        ``complete_legacy=True`` preserves the public three-path loader behavior
        used by existing callers/tests. The main model loader passes ``False`` and
        then performs mode-specific validation for PLET/RUSLE or statistical mode.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Normalized model configuration mapping.
        cps : List[int]
            Conservation Practice Standard (CPS) code or codes.
        pollutants : List[str]
            Pollutant names in model order.
        logger : Any
            Logger used for diagnostic and progress messages.
        complete_legacy : bool
            Whether to complete legacy three-pathway BMP efficiency coverage.
        distribution_catalog : Optional[pd.DataFrame]
            Reusable distribution catalog used to resolve referenced statistics.

        Returns
        -------
        pd.DataFrame
            Normalized BMP efficiency table.

        Raises
        ------
        ValueError
            If no BMP-efficiency records remain for the configured CPS codes and pollutants.
        
    """
    df = _merge_csvs(ci_get(cfg, CFG_BMP_EFFICIENCY), [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pathway_column(df, CFG_BMP_EFFICIENCY, logger)
    df = resolve_distribution_references(df, distribution_catalog, CFG_BMP_EFFICIENCY)
    df = df[df[COL_CPS].astype(int).isin(cps) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")

    # Delay statistic validation until pathway coverage is resolved. PLET/RUSLE
    # and the legacy public loader intentionally allow a blank subsurface row,
    # which is converted to a fixed zero efficiency by their completion logic.
    # Statistical mode performs its own strict validation after coverage checks,
    # so blank statistical-mode efficiencies still fail as intended.
    if complete_legacy:
        return _complete_bmp_efficiency_coverage(df, cps, pollutants, logger)
    return df

def _load_bmp_cost(cfg: Dict[str, Any], cps: List[int], logger: Any, distribution_catalog: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Optionally load BMP cost inputs.

        Parameters
        ----------
        cfg : dict[str, Any]
            Configuration mapping.
        cps : list[int]
            BMP CPS codes to retain.
        logger : Any
            Logger used for progress and warning messages.
        distribution_catalog : Optional[pd.DataFrame]
            Reusable distribution catalog used to resolve referenced statistics.

        Returns
        -------
        pandas.DataFrame or None
            BMP cost table filtered to the requested BMPs, or ``None`` when no
            usable cost table is configured.
        
    """
    path = ci_get(cfg, CFG_BMP_COST)
    if path is None:
        return None
    df = _merge_csvs(path, [COL_CPS, COL_UNIT], CFG_BMP_COST, logger)
    df = resolve_distribution_references(df, distribution_catalog, CFG_BMP_COST)
    validate_stats_table(df, CFG_BMP_COST)
    df = df[df[COL_CPS].astype(int).isin(cps)].copy()
    if df.empty:
        logger.warning("bmp_cost has no records for specified cps; proceeding without costing")
        return None
    return df


def _expand_pollutant_load_rate_defaults(
    df: pd.DataFrame,
    parcel_ids: Sequence[str],
    pollutants: Sequence[str],
) -> pd.DataFrame:
    """Expand ``pid='*'`` load-rate defaults while preserving exact overrides.

        This lets large statistical-mode applications define one distribution for
        many or all parcels and add only the parcel-specific exceptions. Exact
        parcel rows override wildcard rows for the same pollutant/pathway.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        parcel_ids : Sequence[str]
            Parcel identifiers in model order.
        pollutants : Sequence[str]
            Pollutant names in model order.

        Returns
        -------
        pd.DataFrame
            Load-rate table with wildcard parcel defaults expanded.
        
    """
    out = df.copy()
    out[COL_PID] = out[COL_PID].astype(str)
    valid_pids = {str(pid) for pid in parcel_ids}
    out = out[
        out[COL_PID].isin(valid_pids | {"*"})
        & out[COL_POLLUTANT].isin(list(pollutants))
    ].copy()
    if out.empty or not (out[COL_PID] == "*").any():
        return out[out[COL_PID].isin(valid_pids)].reset_index(drop=True)

    explicit = COL_PATHWAY in out.columns
    keys = [COL_PID, COL_POLLUTANT] + ([COL_PATHWAY] if explicit else [])
    validate_unique_rows(out, keys, CFG_POLLUTANT_LOAD_RATE)
    pathways: List[Optional[str]] = (
        list(dict.fromkeys(out[COL_PATHWAY].astype(str).tolist()))
        if explicit else [None]
    )

    defaults: Dict[Tuple[str, Optional[str]], pd.Series] = {}
    exact: Dict[Tuple[str, str, Optional[str]], pd.Series] = {}
    for _, row in out.iterrows():
        path = str(row[COL_PATHWAY]) if explicit else None
        pollutant = str(row[COL_POLLUTANT])
        pid = str(row[COL_PID])
        if pid == "*":
            defaults[(pollutant, path)] = row
        else:
            exact[(pid, pollutant, path)] = row

    expanded: List[pd.Series] = []
    for pid in map(str, parcel_ids):
        for pollutant in map(str, pollutants):
            for path in pathways:
                row = exact.get((pid, pollutant, path))
                if row is None:
                    row = defaults.get((pollutant, path))
                if row is None:
                    continue
                copied = row.copy()
                copied[COL_PID] = pid
                expanded.append(copied)
    if not expanded:
        return out.iloc[0:0].copy()
    return pd.DataFrame(expanded).reset_index(drop=True)


def _load_pollutant_load_rate(cfg: Dict[str, Any], parcels: pd.DataFrame, pollutants: List[str], logger: Any, distribution_catalog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load parcel pollutant load rates for non-PLET mode.

        Parameters
        ----------
        cfg : dict[str, Any]
            Configuration mapping.
        parcels : pandas.DataFrame
            Parcel table used to filter valid IDs.
        pollutants : list[str]
            Pollutants to retain.
        logger : Any
            Logger used for progress reporting.
        distribution_catalog : Optional[pd.DataFrame]
            Reusable distribution catalog used to resolve referenced statistics.

        Returns
        -------
        pandas.DataFrame
            Parcel pollutant load rate table filtered to valid parcels and pollutants.

        Raises
        ------
        ValueError
            If no pollutant load-rate records remain for the modeled parcels and pollutants.
        
    """
    df = _merge_csvs(ci_get(cfg, CFG_POLLUTANT_LOAD_RATE), [COL_PID, COL_POLLUTANT], CFG_POLLUTANT_LOAD_RATE, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_POLLUTANT_LOAD_RATE, logger)
    df = _normalize_pathway_column(df, CFG_POLLUTANT_LOAD_RATE, logger)
    df = resolve_distribution_references(df, distribution_catalog, CFG_POLLUTANT_LOAD_RATE)
    validate_stats_table(df, CFG_POLLUTANT_LOAD_RATE)
    df[COL_PID] = df[COL_PID].astype(str)
    df = _expand_pollutant_load_rate_defaults(
        df, parcels[COL_PID].astype(str).tolist(), pollutants
    )
    if df.empty:
        raise ValueError("pollutant_load_rate has no records for specified parcels+pollutants")
    validate_stats_rows(df, CFG_POLLUTANT_LOAD_RATE)
    return df





def _resolve_aggregate_pathway_fractions(
    cfg: Dict[str, Any], load_generation: Dict[str, Any], pathways: Sequence[str]
) -> Dict[str, float]:
    """Resolve fractions used to split one sampled aggregate parcel load rate.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Normalized model configuration mapping.
        load_generation : Dict[str, Any]
            Load-generation configuration mapping.
        pathways : Sequence[str]
            Pollutant transport pathway names.

        Returns
        -------
        Dict[str, float]
            Normalized pathway fractions keyed by pathway name.

        Raises
        ------
        ValueError
            If pathway fractions are malformed, reference unknown pathways, fall outside ``[0, 1]``, or do not sum to one.
        
    """
    pathways = list(pathways)
    if len(pathways) == 1:
        return {pathways[0]: 1.0}
    raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS)
    if raw is None:
        raw = load_generation.get(CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS)
    fractions: Dict[str, float] = {}
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError(f"{CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS} must be a mapping")
        fractions = {_normalize_pathway_label(k): float(v) for k, v in raw.items()}
    else:
        # Shorthand pathway fractions. A shallow fraction alone means the
        # remaining aggregate load rate is surface load.
        surf_raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE)
        shallow_raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW)
        if surf_raw is not None:
            fractions["surface"] = float(surf_raw)
        if shallow_raw is not None:
            fractions["shallow subsurface"] = float(shallow_raw)
        if shallow_raw is not None and surf_raw is None and "surface" in pathways:
            fractions["surface"] = 1.0 - float(shallow_raw)
        elif surf_raw is not None and shallow_raw is not None and "deep subsurface" in pathways:
            fractions["deep subsurface"] = 1.0 - float(surf_raw) - float(shallow_raw)
    unknown = set(fractions) - set(pathways)
    if unknown:
        raise ValueError(
            f"Pathway fractions refer to pathways not defined by bmp_efficiency: {sorted(unknown)}"
        )
    if not fractions:
        raise ValueError(
            "Statistical mode uses one aggregate pollutant_load_rate per parcel but multiple BMP pathways. "
            f"Define {CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS}, e.g. {{'shallow subsurface': 0.2, 'surface': 0.8}}."
        )
    for path in pathways:
        fractions.setdefault(path, 0.0)
    vals = np.asarray(list(fractions.values()), dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("pollutant load rate pathway fractions must be finite")
    if (vals < 0.0).any() or (vals > 1.0).any():
        raise ValueError("pollutant load rate pathway fractions must each be in [0,1]")
    total = float(vals.sum())
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError(f"pollutant load rate pathway fractions must sum to 1.0; got {total:.12g}")
    return {path: float(fractions[path]) for path in pathways}




def _complete_delivery_ratio_defaults(
    delivery_ratios: Optional[pd.DataFrame],
    parcel_out_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Return a complete parcel/outlet delivery table with neutral defaults.

        Parameters
        ----------
        delivery_ratios : Optional[pd.DataFrame]
            Parcel-to-outlet delivery-ratio table, if configured.
        parcel_out_map : Mapping[str, Sequence[str]]
            Mapping from parcel IDs to connected outlet IDs.

        Returns
        -------
        pd.DataFrame
            Complete parcel-to-outlet delivery-ratio table.

        Raises
        ------
        ValueError
            If any configured delivery-ratio value lies outside ``[0, 1]``.
        
    """
    columns = [COL_PID, COL_OID, COL_SDR_F_TO_S, COL_SDR_S_TO_O, COL_NDR_F_TO_S, COL_NDR_S_TO_O]
    if delivery_ratios is None:
        out = pd.DataFrame(columns=columns)
    else:
        out = delivery_ratios.copy()
    existing = {(str(row[COL_PID]), str(row[COL_OID])) for _, row in out.iterrows()}
    rows: List[Dict[str, Any]] = []
    for pid, outlet_ids in parcel_out_map.items():
        for oid in outlet_ids:
            key = (str(pid), str(oid))
            if key in existing:
                continue
            rows.append({
                COL_PID: key[0],
                COL_OID: key[1],
                COL_SDR_F_TO_S: 1.0,
                COL_SDR_S_TO_O: 1.0,
                COL_NDR_F_TO_S: 1.0,
                COL_NDR_S_TO_O: 1.0,
            })
    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
    for column in (COL_SDR_F_TO_S, COL_SDR_S_TO_O, COL_NDR_F_TO_S, COL_NDR_S_TO_O):
        if column not in out.columns:
            out[column] = pd.Series(dtype=float)
        out[column] = pd.to_numeric(out[column], errors="raise")
        validate_numeric_columns_in_domain(
            out, [column], FRACTION_DOMAIN, CFG_DELIVERY_RATIOS
        )
    return out.reset_index(drop=True)

def load_and_validate_all(cfg: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    """Load, validate, and assemble all scenario inputs.

    Parameters
    ----------
    cfg : dict[str, Any]
        Scenario configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    dict[str, Any]
        Data bundle containing the validated inputs and derived lookup
        structures required by the model.

    Raises
    ------
    ValueError
        If configuration values are invalid or required inputs are missing.
    FileNotFoundError
        If a configured input file does not exist.
    """
    normalize_config(cfg)
    validate_config(cfg)
    logger.info("Loading and validating input datasets")
    with log_scope(logger=logger):
        domain = _load_domain(cfg, logger)
        parcels = _load_parcels(cfg, domain, logger)

        up = _load_parcel_graph(cfg, logger)
        out = _load_parcel_outlets(cfg, logger)
        out = _expand_pid_defaults(
            out,
            parcels[COL_PID].astype(str).tolist(),
            label=CFG_PARCEL_OUT,
            logger=logger,
            key_columns=None,
        )
        sel = _load_parcel_selection(cfg, parcels, logger)

        # Upstream list mapping. A pid_up cell may contain multiple
        # comma-separated IDs, matching the format written by
        # examples/utils/create_parcel_up.py.
        parcel_up_map = _build_parcel_up_map(
            up,
            parcels[COL_PID].astype(str).tolist(),
        )

        # Parcel->outlet mapping
        parcel_out_map: Dict[str, List[str]] = {}
        for pid in parcels[COL_PID].astype(str):
            oids: List[str] = []
            rows = out[out[COL_PID].astype(str) == str(pid)]
            if not rows.empty:
                for value in rows[COL_OIDS].tolist():
                    oids.extend([str(x).strip() for x in str(value).split(",") if str(x).strip()])
            parcel_out_map[str(pid)] = list(dict.fromkeys(oids))

        pollutants = _load_pollutants(cfg)
        cps = _load_cps(cfg)

        load_generation = ci_get(cfg, CFG_LOAD_GENERATION) or {}
        if not isinstance(load_generation, dict):
            raise ValueError("load_generation must be a mapping")
        load_generation = {str(k).lower(): v for k, v in load_generation.items()}
        if "mode" not in load_generation or load_generation["mode"] is None:
            load_generation["mode"] = LOAD_MODE_STATISTICAL
        load_mode = str(load_generation["mode"]).strip().lower()
        if load_mode not in {LOAD_MODE_STATISTICAL, LOAD_MODE_PLET_RUSLE}:
            raise ValueError(f"Unsupported load_generation mode: {load_mode}")
        load_generation["mode"] = load_mode

        # One optional catalog can define reusable numeric distributions for
        # any model input table. References are expanded during input loading.
        distribution_catalog = load_distribution_catalog(
            ci_get(cfg, CFG_INPUT_DISTRIBUTIONS), logger
        )

        outlet_loc = _load_outlet_loc(cfg, domain, logger)
        outlet_target = _load_optional_outlet_stats(cfg, CFG_OUTLET_TARGET, [COL_OID, COL_POLLUTANT, COL_TARGET], CFG_OUTLET_TARGET, logger)
        outlet_mean = _load_optional_outlet_stats(cfg, CFG_OUTLET_MEAN, [COL_OID, COL_POLLUTANT, COL_MEAN], CFG_OUTLET_MEAN, logger)

        if "pathway_mode" in load_generation:
            raise ValueError(
                "load_generation.pathway_mode has been removed; "
                "plet_rusle mode always derives pathway loads from PLET/RUSLE inputs"
            )
        supplied_legacy_groundwater_keys = (
            LOAD_GROUNDWATER_LOADS in load_generation
            or LOAD_TREAT_GROUNDWATER_WITH_BMPS in load_generation
        )
        if LOAD_GROUNDWATER_LOADS not in load_generation or load_generation[LOAD_GROUNDWATER_LOADS] is None:
            load_generation[LOAD_GROUNDWATER_LOADS] = False
        groundwater_loads = bool(load_generation[LOAD_GROUNDWATER_LOADS])
        if LOAD_TREAT_GROUNDWATER_WITH_BMPS not in load_generation or load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] is None:
            load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] = False
        treat_groundwater_with_bmps = bool(load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS])
        load_generation[LOAD_GROUNDWATER_LOADS] = groundwater_loads
        load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] = treat_groundwater_with_bmps
        if load_mode == LOAD_MODE_PLET_RUSLE and supplied_legacy_groundwater_keys:
            logger.verbose(
                "plet_rusle now always estimates lookup-derived subsurface loads; "
                "groundwater_loads/treat_groundwater_with_bmps do not alter pathway generation. "
                "Subsurface BMP treatment is controlled by the subsurface efficiency."
            )

        if ci_get(cfg, CFG_BMP_EFFICIENCY) is None:
            raise ValueError("bmp_efficiency is required")
        bmp_eff = _load_bmp_efficiency(
            cfg, cps, pollutants, logger, complete_legacy=False,
            distribution_catalog=distribution_catalog,
        )
        bmp_cost = _load_bmp_cost(cfg, cps, logger, distribution_catalog)

        plet_inputs = None
        rusle_inputs = None
        pollutant_concentrations = None
        groundwater_concentrations = None
        if load_mode == LOAD_MODE_PLET_RUSLE:
            plet_inputs = _load_plet_parameter_table(
                load_generation.get(LOAD_PLET_INPUTS),
                sel[COL_PID].astype(str).tolist(),
                logger,
                distribution_catalog,
            )
            if plet_inputs is None:
                raise ValueError("load_generation.plet_inputs is required for mode='plet_rusle'")
            plet_hydrology_lookup = _load_plet_hydrology_lookup(
                load_generation.get(LOAD_HYDROLOGY_LOOKUP),
                logger,
                distribution_catalog,
            )
            # Keep the normalized/resolved table with the validated load-generation
            # settings so workers receive it without requiring Model changes or
            # repeatedly reading the CSV. The leading underscore marks it as an
            # internal resolved input rather than a user-facing config key.
            load_generation["_hydrology_lookup_table"] = plet_hydrology_lookup
            rusle_inputs = _load_parameter_stats_table(
                load_generation.get(LOAD_RUSLE_INPUTS), LOAD_RUSLE_INPUTS, logger,
                distribution_catalog,
            )
            pollutant_concentrations = _load_pollutant_concentrations(
                load_generation.get(LOAD_CONCENTRATIONS), pollutants, logger,
                distribution_catalog,
            )
            groundwater_concentrations = _load_groundwater_concentrations(
                load_generation.get(LOAD_GROUNDWATER_CONCENTRATIONS), pollutants, logger,
                distribution_catalog,
            )
            plet_inputs = _append_parameter_defaults(plet_inputs, pollutants)
            validate_plet_runtime_inputs(
                plet_inputs,
                rusle_inputs,
                pollutant_concentrations,
                groundwater_concentrations,
                parcels[COL_PID].astype(str).tolist(),
                pollutants,
            )
            if any(p in {"TN", "TP"} for p in pollutants) and pollutant_concentrations is None:
                raise ValueError(
                    "load_generation.pollutant_concentrations is required for TN or TP in plet_rusle mode"
                )
            if any(p != "TSS" for p in pollutants) and groundwater_concentrations is None:
                raise ValueError(
                    "load_generation.groundwater_concentrations is required for non-TSS pollutants in plet_rusle mode"
                )
            pollutant_load_rate = None
            pathways = list(PLET_PATHWAY_VALUES)
            pollutant_load_rate_is_aggregate = False
            pollutant_load_rate_pathway_fractions: Dict[str, float] = {}
            bmp_eff = _complete_plet_bmp_efficiency_coverage(bmp_eff, cps, pollutants, logger)
        else:
            pollutant_load_rate = _load_pollutant_load_rate(
                cfg, parcels, pollutants, logger, distribution_catalog
            )
            load_rate_pathways, pollutant_load_rate_is_aggregate = validate_statistical_load_rates(
                pollutant_load_rate, parcels, pollutants
            )
            if pollutant_load_rate_is_aggregate:
                if COL_PATHWAY in bmp_eff.columns:
                    pathways = list(dict.fromkeys(bmp_eff[COL_PATHWAY].astype(str).tolist()))
                else:
                    pathways = ["surface"]
                pollutant_load_rate_pathway_fractions = _resolve_aggregate_pathway_fractions(
                    cfg, load_generation, pathways
                )
            else:
                pathways = load_rate_pathways
                pollutant_load_rate_pathway_fractions = {}
            bmp_eff = validate_statistical_efficiency_coverage(
                bmp_eff, cps, pollutants, pathways
            )

        delivery_ratios = _load_delivery_ratios(cfg, logger)
        if delivery_ratios is not None:
            delivery_ratios = _expand_pid_defaults(
                delivery_ratios,
                parcels[COL_PID].astype(str).tolist(),
                label=CFG_DELIVERY_RATIOS,
                logger=logger,
                key_columns=[COL_OID],
            )
        delivery_ratios = _complete_delivery_ratio_defaults(delivery_ratios, parcel_out_map)

        # Precompute averages for selection heuristics and reporting
        avg_area_ha = float(parcels["area_ha"].mean())
        avg_perim_m = float(parcels["perim_m"].mean())

        logger.info("Input validation complete; assembling data payload")

    return dict(
        parcels=parcels,
        parcel_p=sel,
        parcel_up_map=parcel_up_map,
        parcel_out_map=parcel_out_map,
        pollutants=pollutants,
        cps=cps,
        outlet_loc=outlet_loc,
        outlet_target=outlet_target,
        outlet_mean=outlet_mean,
        bmp_eff=bmp_eff,
        bmp_cost=bmp_cost,
        pollutant_load_rate=pollutant_load_rate,
        delivery_ratios=delivery_ratios,
        load_generation=load_generation,
        plet_inputs=plet_inputs,
        rusle_inputs=rusle_inputs,
        pollutant_concentrations=pollutant_concentrations,
        groundwater_concentrations=groundwater_concentrations,
        pathways=pathways,
        pollutant_load_rate_pathway_fractions=pollutant_load_rate_pathway_fractions,
        pollutant_load_rate_is_aggregate=pollutant_load_rate_is_aggregate,
        bmp_limit_n=ci_get(cfg, CFG_BMP_LIMIT_N),
        bmp_limit_usd=ci_get(cfg, CFG_BMP_LIMIT_USD),
        n_scenarios=int(ci_get(cfg, CFG_N_SCENARIOS)),
        random_seed=ci_get(cfg, CFG_RANDOM_SEED),
        avg_area_ha=avg_area_ha,
        avg_perim_m=avg_perim_m,
        parallel=ci_get(cfg, CFG_PARALLEL),
    )
