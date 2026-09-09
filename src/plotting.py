"""Create cross-scenario summary plots.

This module turns scenario trajectory data into line plots comparing BMP
sequences across pollutants, outlets, and x/y-axis combinations.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from .constants import (
    CFG_BMP_COST,
    CFG_OUTLET_MEAN,
    CFG_OUTLET_TARGET,
    COL_MEAN,
    COL_OID,
    COL_POLLUTANT,
    COL_TARGET,
    DATA_OUTLET_LOC,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_POLLUTANTS,
    XAXIS_COUNT,
    XAXIS_COST,
    YAXIS_MEAN,
    YAXIS_TARGET,
    YAXIS_TOTAL,
)
from .logging_utils import log_scope
from .input_config import load_trajectory_records
from .constants import DIR_OUTLET_TRAJECTORIES, FILE_ALL_SCENARIOS_PARQUET


def _build_denominator_maps(
    data: Dict[str, Any],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Build denominator lookup maps for outlet plots.

        Parameters
        ----------
        data : Dict[str, Any]
            Loaded model data mapping.

        Returns
        -------
        Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]
            Denominator lookup mappings used for outlet plots.
        
    """
    target_map: Dict[Tuple[str, str], float] = {}
    mean_map: Dict[Tuple[str, str], float] = {}
    tgt_df = data.get(DATA_OUTLET_TARGET)
    if tgt_df is not None and not tgt_df.empty:
        for _, row in tgt_df.iterrows():
            key = (str(row[COL_OID]), str(row[COL_POLLUTANT]))
            try:
                target_map[key] = float(row[COL_TARGET])
            except (TypeError, ValueError):
                continue
    mean_df = data.get(DATA_OUTLET_MEAN)
    if mean_df is not None and not mean_df.empty:
        for _, row in mean_df.iterrows():
            key = (str(row[COL_OID]), str(row[COL_POLLUTANT]))
            try:
                mean_map[key] = float(row[COL_MEAN])
            except (TypeError, ValueError):
                continue

    return target_map, mean_map


def _build_line_segments(
    by_scenario: Dict[int, List[Tuple[float, float]]],
    *,
    pollutant: str,
    oid: str,
    x_axis: str,
    y_axis: str,
) -> List[List[Tuple[float, float]]]:
    """Build plot line segments while preserving simulation record order.

        Non-finite trajectory coordinates are treated as invalid model output and
        raise an actionable error instead of silently producing an empty or
        malformed plot.

        Parameters
        ----------
        by_scenario : Dict[int, List[Tuple[float, float]]]
            Plot coordinates grouped by scenario identifier.
        pollutant : str
            Pollutant name.
        oid : str
            Outlet identifier.
        x_axis : str
            Name of the x-axis metric.
        y_axis : str
            Name of the y-axis metric.

        Returns
        -------
        List[List[Tuple[float, float]]]
            Ordered line segments for plotting scenario trajectories.

        Raises
        ------
        ValueError
            If any trajectory coordinate is non-finite.
        
    """
    lines: List[List[Tuple[float, float]]] = []
    for sid, points in sorted(by_scenario.items()):
        xs = [0.0]
        ys = [0.0]
        for point_index, (raw_x, raw_y) in enumerate(points, start=1):
            x_value = float(raw_x)
            y_value = float(raw_y)
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                raise ValueError(
                    "Non-finite trajectory value while plotting: "
                    f"scenario={sid}, pollutant={pollutant}, outlet={oid}, "
                    f"x_axis={x_axis}, y_axis={y_axis}, point={point_index}, "
                    f"x_value={x_value}, y_value={y_value}"
                )
            xs.append(x_value)
            ys.append(y_value)

        lines.extend(
            [[(xs[i], ys[i]), (xs[i + 1], ys[i + 1])] for i in range(len(xs) - 1)]
        )
    return lines


def make_summary_plots(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    scenario_records: Optional[Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]],
    outputs_dir: Path,
    logger,
) -> None:
    """Write one plot image per pollutant/outlet/axis combination.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Normalized model configuration mapping.
        data : Dict[str, Any]
            Loaded model data mapping.
        scenario_records : Optional[Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]]
            Scenario trajectory records, if already available in memory.
        outputs_dir : Path
            Directory where model outputs are written.
        logger : Any
            Logger used for diagnostic and progress messages.
        
    """
    scenario_records = scenario_records or {}
    canonical_traj = Path(outputs_dir) / DIR_OUTLET_TRAJECTORIES / FILE_ALL_SCENARIOS_PARQUET
    # Prefer in-memory records produced by the current run; loading/parsing the
    # canonical parquet can be very expensive for large runs.
    if not scenario_records and canonical_traj.exists():
        try:
            scenario_records = load_trajectory_records(canonical_traj)
            logger.verbose(f"Loaded canonical trajectory table for plotting: {canonical_traj}")
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                f"Failed to load canonical trajectory table ({canonical_traj}); "
                f"falling back to in-memory records: {ex}"
            )
    pollutants = data[DATA_POLLUTANTS]
    oids = [str(x) for x in data[DATA_OUTLET_LOC][COL_OID].astype(str).tolist()]
    logger.verbose(f"Generating summary plots for pollutants={pollutants} outlets={oids}")

    x_axes = [XAXIS_COUNT]
    if cfg.get(CFG_BMP_COST):
        x_axes.append(XAXIS_COST)

    y_axes = [YAXIS_TOTAL]
    if cfg.get(CFG_OUTLET_TARGET):
        y_axes.append(YAXIS_TARGET)
    if cfg.get(CFG_OUTLET_MEAN):
        y_axes.append(YAXIS_MEAN)
    target_map, mean_map = _build_denominator_maps(data)
    warned_missing_denominator = set()

    plots_dir = Path(outputs_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    with log_scope(logger=logger):
        for pol in pollutants:
            for oid in oids:
                for xax in x_axes:
                    for yax in y_axes:
                        with log_scope(label=f"plot pol={pol} oid={oid} x={xax} y={yax}", logger=logger):
                            if yax == YAXIS_TARGET and cfg.get(CFG_OUTLET_TARGET):
                                tgt = target_map.get((str(oid), str(pol)))
                                if tgt is None or not np.isfinite(tgt) or tgt <= 0.0:
                                    key = (str(oid), str(pol), YAXIS_TARGET)
                                    if key not in warned_missing_denominator:
                                        logger.warning(
                                            f"Skipping plot for pol={pol} oid={oid} y={YAXIS_TARGET}: "
                                            "missing or nonpositive outlet target denominator"
                                        )
                                        warned_missing_denominator.add(key)
                                    continue
                            if yax == YAXIS_MEAN and cfg.get(CFG_OUTLET_MEAN):
                                mu = mean_map.get((str(oid), str(pol)))
                                if mu is None or not np.isfinite(mu) or mu <= 0.0:
                                    key = (str(oid), str(pol), YAXIS_MEAN)
                                    if key not in warned_missing_denominator:
                                        logger.warning(
                                            f"Skipping plot for pol={pol} oid={oid} y={YAXIS_MEAN}: "
                                            "missing or nonpositive outlet mean denominator"
                                        )
                                        warned_missing_denominator.add(key)
                                    continue
                            by_scenario: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
                            for (p, o, xa, ya), trip in scenario_records.items():
                                if p == pol and o == oid and xa == xax and ya == yax:
                                    for sid, xx, yy in trip:
                                        by_scenario[sid].append((xx, yy))
                            if not by_scenario:
                                continue

                            lines = _build_line_segments(
                                by_scenario,
                                pollutant=str(pol),
                                oid=str(oid),
                                x_axis=str(xax),
                                y_axis=str(yax),
                            )
                            if not lines:
                                logger.warning(
                                    f"Skipping empty plot for pol={pol} oid={oid} x={xax} y={yax}: "
                                    "no trajectory segments"
                                )
                                continue

                            plt.figure(figsize=(7, 5), dpi=200)
                            ax = plt.gca()
                            lc = LineCollection(lines, colors="steelblue", linewidths=1.25, alpha=0.5)
                            ax.add_collection(lc)
                            ax.autoscale()
                            plt.xlabel("total cost (USD)" if xax == XAXIS_COST else "total bmp count")
                            if yax == YAXIS_TOTAL:
                                plt.ylabel(f"total {pol} load reduction (delivered)")
                            elif yax == YAXIS_TARGET:
                                plt.ylabel(f"{pol} reduction (% of target)")
                            else:
                                plt.ylabel(f"{pol} reduction (% of mean load)")
                            plt.title(f"{pol} | outlet {oid} | x={xax} | y={yax}")
                            plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
                            fname = plots_dir / f"plot_{pol}_oid{oid}_x{xax}_y{yax}.jpg"
                            plt.tight_layout()
                            logger.verbose(
                                f"Saving plot file={fname} xax={xax} yax={yax} pollutant={pol} oid={oid}"
                            )
                            plt.savefig(fname, format="jpg", dpi=300)
                            plt.close()
                            logger.info(f"Saved plot: {fname}")
