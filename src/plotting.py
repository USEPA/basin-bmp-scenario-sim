import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    logger: logging.Logger,
) -> None:
    """Generate summary plots for scenario outcomes.

    The plots compare cumulative pollutant reduction across scenarios and outlets,
    using cost or BMP count on the x-axis and absolute/target/mean reductions on the y-axis.
    """
    pollutants = data[DATA_POLLUTANTS]
    oids = [str(x) for x in data[DATA_OUTLET_LOC][COL_OID].astype(str).tolist()]
    logger.debug(f"Generating summary plots for pollutants={pollutants} outlets={oids}")

    x_axes = []
    if cfg.get(CFG_BMP_COST):
        x_axes.append(XAXIS_COST)
    x_axes.append(XAXIS_COUNT)

    y_axes = [YAXIS_TOTAL]
    if cfg.get(CFG_OUTLET_TARGET):
        y_axes.append(YAXIS_TARGET)
    if cfg.get(CFG_OUTLET_MEAN):
        y_axes.append(YAXIS_MEAN)

    for pol in pollutants:
        for oid in oids:
            for xax in x_axes:
                for yax in y_axes:
                    by_scenario = defaultdict(list)
                    for (p, o, xa, ya), trip in scenario_records.items():
                        if p==pol and o==oid and xa==xax and ya==yax:
                            for (sid, xx, yy) in trip:
                                by_scenario[sid].append((xx, yy))
                    if not by_scenario:
                        continue
                    plt.figure(figsize=(7,5), dpi=200)
                    for sid, pts in sorted(by_scenario.items()):
                        pts = sorted(pts, key=lambda t: t[0])
                        xs = [x for x,y in pts]
                        ys = [y for x,y in pts]
                        lbl = f"scenario {sid}"
                        plt.plot(xs, ys, marker="o", markersize=2, linewidth=1.25, label=lbl, alpha=0.9)
                    plt.xlabel("total cost (USD)" if xax == XAXIS_COST else "total bmp count")
                    if yax == YAXIS_TOTAL:
                        plt.ylabel(f"total {pol} load reduction (delivered)")
                    elif yax == YAXIS_TARGET:
                        plt.ylabel(f"{pol} reduction (% of target)")
                    else:
                        plt.ylabel(f"{pol} reduction (% of mean load)")
                    plt.title(f"{pol} | outlet {oid} | x={xax} | y={yax}")
                    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
                    plt.legend(fontsize=6)
                    fname = outputs_dir / f"plot_{pol}_oid{oid}_x{xax}_y{yax}.jpg"
                    plt.tight_layout()
                    logger.debug(f"Saving plot file={fname} xax={xax} yax={yax} pollutant={pol} oid={oid}")
                    plt.savefig(fname, format="jpg", dpi=300)
                    plt.close()
                    logger.info(f"Saved plot: {fname}")
