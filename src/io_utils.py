"""Filesystem and serialization helpers for model inputs and outputs.

This module deliberately avoids model-specific defaults, aliases, and domain
validation. Defaults and normalization live in :mod:`src.input_config`;
validation policy lives in :mod:`src.input_validation`.
"""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd
import yaml


def read_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Deserialize a YAML configuration file without applying model semantics.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the YAML configuration file.

        Returns
        -------
        Dict[str, Any]
            Deserialized configuration mapping.

        Raises
        ------
        FileNotFoundError
            If the YAML configuration file does not exist.
        ValueError
            If the YAML document root is not a mapping.
        
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Input YAML file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Configuration YAML root must be a mapping")
    return dict(loaded)


def read_csv_table(path: Union[str, Path]) -> pd.DataFrame:
    """Deserialize one CSV file without model-specific normalization.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Deserialized CSV table.
        
    """
    return pd.read_csv(path)


def read_csv_tables(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
) -> List[pd.DataFrame]:
    """Deserialize one or more CSV files, preserving one frame per file.

        Parameters
        ----------
        paths : Union[str, Path, Sequence[Union[str, Path]]]
            One or more input file paths.

        Returns
        -------
        List[pd.DataFrame]
            Deserialized CSV tables in input order.
        
    """
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    return [read_csv_table(item) for item in items]


def read_geodataframe(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """Deserialize a geospatial vector dataset without domain normalization.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the geospatial vector dataset.

        Returns
        -------
        gpd.GeoDataFrame
            Deserialized geospatial dataset.
        
    """
    return gpd.read_file(path)


def read_parquet_table(path: Union[str, Path]) -> pd.DataFrame:
    """Deserialize a Parquet table without model-specific validation.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the Parquet file.

        Returns
        -------
        pd.DataFrame
            Deserialized Parquet table.
        
    """
    return pd.read_parquet(Path(path))


def _write_parquet_atomic(
    df: pd.DataFrame, path: Path, *, logger: logging.Logger
) -> None:
    """Write a parquet file atomically.

        Parameters
        ----------
        df : pd.DataFrame
            Table to serialize.
        path : Path
            Destination path for the Parquet file.
        logger : logging.Logger
            Logger used for diagnostic and progress messages.

        Raises
        ------
        RuntimeError
            If no supported Parquet engine is installed.
        
    """
    del logger
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    except (ImportError, ModuleNotFoundError) as ex:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            "Parquet output is required but no parquet engine is installed. "
            "Install pyarrow or fastparquet in the active environment."
        ) from ex


def _flatten_plot_records(
    merged: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
) -> pd.DataFrame:
    """Convert plot records into the normalized serialized trajectory table.

        Parameters
        ----------
        merged : Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
            Merged trajectory records keyed by pollutant, outlet, and axis definitions.

        Returns
        -------
        pd.DataFrame
            Normalized trajectory table suitable for serialization.
        
    """
    rows: List[Dict[str, Any]] = []
    step_counters: Dict[Tuple[int, str, str, str, str], int] = defaultdict(int)
    for (pol, oid, xax, yax), points in merged.items():
        for sid, xval, yval in points:
            counter_key = (int(sid), str(pol), str(oid), str(xax), str(yax))
            step_counters[counter_key] += 1
            rows.append(
                {
                    "scenario": int(sid),
                    "pollutant": str(pol),
                    "oid": str(oid),
                    "x_axis": str(xax),
                    "y_axis": str(yax),
                    "step": int(step_counters[counter_key]),
                    "x_value": float(xval),
                    "y_value": float(yval),
                }
            )
    columns = [
        "scenario", "pollutant", "oid", "x_axis", "y_axis",
        "step", "x_value", "y_value",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["scenario", "pollutant", "oid", "x_axis", "y_axis", "step"]
    ).reset_index(drop=True)
