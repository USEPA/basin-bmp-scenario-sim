import logging
import numpy as np
from numpy.random import Generator
from typing import Any, Dict, Optional, Sequence

def _trunc_normal(
    rng: Generator,
    mean: float,
    sd: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
    size: Optional[int] = None,
) -> np.ndarray:
    """Sample from a truncated normal distribution using rejection sampling.

    This helper generates values centered on mean with standard deviation sd,
    while enforcing optional lower and upper bounds.
    """
    # simple rejection sampler for truncation
    out: list[float] = []
    n = size or 1
    if sd <= 0:
        return np.full(n, mean)
    while len(out) < n:
        x = rng.normal(mean, sd, size=n)
        if low is not None:
            x = x[x >= low]
        if high is not None:
            x = x[x <= high]
        out.extend(x.tolist())
    return np.array(out[:n])

def _piecewise_quantile_sample(
    rng: Generator,
    stats: Dict[str, float],
    size: int = 1,
    verbose_logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Sample from a piecewise linear distribution defined by percentiles."""

    if verbose_logger:
        verbose_logger.debug(f"calling _piecewise_quantile_sample with stats={stats} size={size}")
    # stats includes min/max and possibly percentiles like p5, p50, p90
    cols = {k.lower(): v for k, v in stats.items()}
    # Collect points (p, q)
    pts = []
    if any(k in cols for k in ("min","minimum","p0")):
        qmin = cols.get("min", cols.get("minimum", cols.get("p0")))
        pts.append((0.0, float(qmin)))
    else:
        raise ValueError("Piecewise sampler requires min")

    percs = {}
    for k, v in list(cols.items()):
        if k.startswith("p") and k[1:].isdigit():
            percs[int(k[1:])] = float(v)
    for p in sorted(percs.keys()):
        if p>0 and p<100:
            pts.append((p/100.0, percs[p]))

    if any(k in cols for k in ("max","maximum","p100")):
        qmax = cols.get("max", cols.get("maximum", cols.get("p100")))
        pts.append((1.0, float(qmax)))
    else:
        raise ValueError("Piecewise sampler requires max")

    pts = sorted(pts, key=lambda t: t[0])

    u = rng.uniform(0.0, 1.0, size=size)
    samples = np.empty(size, dtype=float)
    for i, ui in enumerate(u):
        for (p0, q0), (p1, q1) in zip(pts[:-1], pts[1:]):
            if ui >= p0 and ui <= p1:
                if p1 == p0:
                    samples[i] = q0
                else:
                    t = (ui - p0) / (p1 - p0)
                    samples[i] = q0 + t * (q1 - q0)
                break
    if verbose_logger:
        verbose_logger.debug(f"Sampled values {samples} from piecewise quantiles with stats={stats}")
    return samples

def sample_from_stats(
    rng: Generator,
    stats: Dict[str, float],
    kind: Optional[str] = None,
    verbose_logger: Optional[logging.Logger] = None,
) -> float:
    """Sample a value from distribution statistics provided by the input data.

    The algorithm chooses the strongest available representation: piecewise
    quantiles if percentiles exist, otherwise truncated normal when mean/sd are
    available, or uniform sampling when only min/max are provided.
    """
    if verbose_logger:
        verbose_logger.debug(f"calling sample_from_stats with stats={stats} kind={kind}")

    cols = {k.lower(): v for k, v in stats.items()}
    # Prefer piecewise if min/max with any percentiles are provided
    has_min = any(k in cols for k in ("min","minimum","p0"))
    has_max = any(k in cols for k in ("max","maximum","p100"))
    has_sd = any(k in cols for k in ("sd","std"))
    has_mean = any(k in cols for k in ("mean","average","avg"))
    has_percentiles = any(k.startswith("p") and k[1:].isdigit() for k in cols.keys())

    low, high = None, None
    if kind == "efficiency":
        min_candidates = [cols.get(k) for k in ("min","minimum","p0") if k in cols]
        if not any((m is not None and float(m) < 0) for m in min_candidates):
            low = 0.0
        high = 1.0
    elif kind == "yield":
        low = 0.0

    if has_min and has_max and has_percentiles:
        s = _piecewise_quantile_sample(rng, cols, size=1, verbose_logger=verbose_logger)[0]
    elif has_min and has_max and has_mean and not has_sd:
        mn = float(cols.get("mean", cols.get("average", cols.get("avg"))))
        lo = float(cols.get("min", cols.get("minimum", cols.get("p0"))))
        hi = float(cols.get("max", cols.get("maximum", cols.get("p100"))))
        sd = (hi - lo) / 4.0
        s = _trunc_normal(rng, mn, sd, low=lo if low is None else max(low, lo), high=hi if high is None else min(high, hi))[0]
    elif has_min and has_max and not has_mean and not has_sd and not has_percentiles:
        lo = float(cols.get("min", cols.get("minimum", cols.get("p0"))))
        hi = float(cols.get("max", cols.get("maximum", cols.get("p100"))))
        lo = max(lo, low) if low is not None else lo
        hi = min(hi, high) if high is not None else hi
        s = rng.uniform(lo, hi)
    elif has_mean and has_sd:
        mn = float(cols.get("mean", cols.get("average", cols.get("avg"))))
        sd = float(cols.get("sd", cols.get("std")))
        s = _trunc_normal(rng, mn, sd, low=low, high=high)[0]
    else:
        raise ValueError("Insufficient distribution statistics to sample")
    if verbose_logger:
        verbose_logger.debug(f"Sampled value {s:.6g} from stats={stats} kind={kind}")

    if low is not None and s < low:
        if verbose_logger:
            verbose_logger.debug(f"Sampled value {s:.6g} less than bound minimum ({low}), reducing to {low}")
        s = low
    if high is not None and s > high:
        if verbose_logger:
            verbose_logger.debug(f"Sampled value {s:.6g} greater than bound maximum ({high}), reducing to {high}")
        s = high
    return float(s)
