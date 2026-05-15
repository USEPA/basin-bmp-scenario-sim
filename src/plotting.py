import matplotlib.pyplot as plt
from collections import defaultdict

def make_summary_plots(cfg, data, scenario_records, outputs_dir, logger):
    pollutants = data["pollutants"]
    oids = [str(x) for x in data["outlet_loc"]["oid"].astype(str).tolist()]

    x_axes = []
    if cfg.get("bmp_cost"):
        x_axes.append("cost")
    x_axes.append("count")

    y_axes = ["total"]
    if cfg.get("outlet_target"):
        y_axes.append("target")
    if cfg.get("outlet_mean"):
        y_axes.append("mean")

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
                    plt.xlabel("total cost (USD)" if xax=="cost" else "total bmp count")
                    if yax=="total":
                        plt.ylabel(f"total {pol} load reduction (delivered)")
                    elif yax=="target":
                        plt.ylabel(f"{pol} reduction (% of target)")
                    else:
                        plt.ylabel(f"{pol} reduction (% of mean load)")
                    plt.title(f"{pol} | outlet {oid} | x={xax} | y={yax}")
                    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
                    plt.legend(fontsize=6)
                    fname = outputs_dir / f"plot_{pol}_oid{oid}_x{xax}_y{yax}.jpg"
                    plt.tight_layout()
                    plt.savefig(fname, format="jpg", dpi=300)
                    plt.close()
                    logger.info(f"Saved plot: {fname}")
