"""
Plotting utilities for cross-scenario comparisons.

Generates line plots that compare cumulative delivered reductions by pollutant
and outlet, using either BMP count or cost on x-axis, and absolute or normalized
reductions on y-axis.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .constants import (
    CFG_BMP_COST,
    CFG_OUTLET_MEAN,
    CFG_OUTLET_TARGET,
    COL_OID,
    DATA_OUTLET_LOC,
    DATA_POLLUTANTS,
    XAXIS_COUNT,
    XAXIS_COST,
    YAXIS_MEAN,
    YAXIS_TARGET,
    YAXIS_TOTAL,
)


def make_summary_plots(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    scenario_records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]],
    outputs_dir: Path,
    logger,
) -> None:
    """Generate summary plots for scenario outcomes.

    Parameters
    ----------
    cfg : Dict[str, Any]
        User configuration dict.
    data : Dict[str, Any]
        Validated input payload (pollutants, outlet locations, etc.).
    scenario_records : Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
        Mapping (pollutant, outlet_oid, x_axis, y_axis) -> list of (sid, x, y).
    outputs_dir : Path
        Root outputs directory. Plots are saved to outputs/plots/.
    logger : logging.Logger
        Logger for messages.
    """
    pollutants = data[DATA_POLLUTANTS]
    oids = [str(x) for x in data[DATA_OUTLET_LOC][COL_OID].astype(str).tolist()]
    logger.debug(f"Generating summary plots for pollutants={pollutants} outlets={oids}")

    x_axes = [XAXIS_COUNT]
    if cfg.get(CFG_BMP_COST):
        x_axes.append(XAXIS_COST)

    y_axes = [YAXIS_TOTAL]
    if cfg.get(CFG_OUTLET_TARGET):
        y_axes.append(YAXIS_TARGET)
    if cfg.get(CFG_OUTLET_MEAN):
        y_axes.append(YAXIS_MEAN)

    plots_dir = Path(outputs_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for pol in pollutants:
        for oid in oids:
            for xax in x_axes:
                for yax in y_axes:
                    by_scenario = defaultdict(list)
                    for (p, o, xa, ya), trip in scenario_records.items():
                        if p == pol and o == oid and xa == xax and ya == yax:
                            for (sid, xx, yy) in trip:
                                by_scenario[sid].append((xx, yy))
                    if not by_scenario:
                        continue

                    plt.figure(figsize=(7, 5), dpi=200)
                    ax = plt.gca()

                    # Draw multi-segment lines scenario-by-scenario
                    lines = []
                    for sid, pts in sorted(by_scenario.items()):
                        pts = sorted(pts, key=lambda t: t[0])
                        xs = [0] + [x for x, _ in pts]
                        ys = [0] + [y for _, y in pts]
                        segments = [[(xs[i], ys[i]), (xs[i + 1], ys[i + 1])] for i in range(len(xs) - 1)]
                        lines.extend(segments)

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
                    logger.debug(f"Saving plot file={fname} xax={xax} yax={yax} pollutant={pol} oid={oid}")
                    plt.savefig(fname, format="jpg", dpi=300)
                    plt.close()
                    logger.info(f"Saved plot: {fname}")