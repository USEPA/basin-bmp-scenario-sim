"""Random sampling helpers for model inputs.

This module provides the sampling primitives used to draw values from fixed
inputs, summary statistics, and percentile-based distributions while
respecting optional bounds for loads and preserving signed BMP effects.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from .input_units import convert_sampling_mapping
from .input_validation import (
    PhysicalDomain,
    physical_domain_for_sampling_kind,
    validate_scalar_in_domain,
)

if TYPE_CHECKING:
    from .model import Model

def _first_present(mapping: Dict[str, float], names: Tuple[str, ...]) -> float:
    """Return the first present alias from a validated statistics mapping.

        Parameters
        ----------
        mapping : Dict[str, float]
            Input mapping.
        names : Tuple[str, ...]
            Candidate aliases in precedence order.

        Returns
        -------
        float
            Value associated with the first matching alias.
        
    """
    return next(mapping[name] for name in names if name in mapping)


def _trunc_normal(
    self: "Model",
    mean: float,
    sd: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
    size: Optional[int] = None,
) -> np.ndarray:
    """Draw samples from an explicitly truncated normal distribution.

    Bounds define the distribution support; they are not used to repair an
    invalid draw after sampling. A deterministic ``sd == 0`` value outside
    the support is therefore rejected.

    Parameters
    ----------
    self : Model
        Active simulation model instance providing the RNG.
    mean : float
        Mean of the underlying normal distribution.
    sd : float
        Standard deviation of the underlying normal distribution.
    low : float or None, optional
        Lower truncation bound.
    high : float or None, optional
        Upper truncation bound.
    size : int or None, optional
        Number of values to sample.

    Returns
    -------
    numpy.ndarray
        Samples lying within the requested support.

    Raises
    ------
    ValueError
        If the distribution definition is non-finite, has negative spread,
        has reversed bounds, or a deterministic value is outside the support.
    """
    mean = float(mean)
    sd = float(sd)
    if not np.isfinite(mean) or not np.isfinite(sd):
        raise ValueError("Normal mean and sd must be finite")
    if sd < 0.0:
        raise ValueError("Normal sd must be >= 0")
    if low is not None:
        low = float(low)
        if not np.isfinite(low):
            raise ValueError("Normal lower bound must be finite")
    if high is not None:
        high = float(high)
        if not np.isfinite(high):
            raise ValueError("Normal upper bound must be finite")
    if low is not None and high is not None and low > high:
        raise ValueError("Normal lower bound cannot exceed upper bound")

    n = int(size or 1)
    if n < 1:
        raise ValueError("Normal sample size must be >= 1")

    def in_support(value: float) -> bool:
        return (low is None or value >= low) and (high is None or value <= high)

    if sd == 0.0:
        if not in_support(mean):
            raise ValueError(
                f"Deterministic normal value {mean:g} is outside truncation support "
                f"[{low}, {high}]"
            )
        return np.full(n, mean, dtype=float)

    out = np.empty(n, dtype=float)
    filled = 0
    batch = max(4, n)
    max_tries = 20
    tries = 0
    while filled < n and tries < max_tries:
        x = self.rng.normal(mean, sd, size=batch)
        if low is not None:
            x = x[x >= low]
        if high is not None:
            x = x[x <= high]
        k = min(len(x), n - filled)
        if k > 0:
            out[filled : filled + k] = x[:k]
            filled += k
        tries += 1
        batch = min(max(batch * 2, n - filled), (n - filled) * 8 + 1024)

    if filled < n:
        # A validated mean lies inside every explicit physical/support bound.
        # Reusing it here is a numerical fallback for very low acceptance, not
        # a correction of an invalid user value.
        if not in_support(mean):
            raise RuntimeError(
                "Could not draw enough truncated-normal samples and the mean "
                "is outside the requested support"
            )
        out[filled:] = mean

    return out


def _piecewise_quantile_sample(
    self: "Model",
    stats: Dict[str, float],
    size: int = 1,
) -> np.ndarray:
    """Sample values from piecewise percentile statistics.
    Parameters
    ----------
    self : Model
        Active simulation model instance providing the RNG.
    stats : dict[str, float]
        Mapping containing percentile statistics such as ``min``, ``p50``,
        and ``max``.
    size : int, optional
        Number of values to sample. Default is ``1``.

    Returns
    -------
    numpy.ndarray
        Sampled values interpolated between the supplied percentile points.
    Raises
    ------
    ValueError
        If either a minimum or maximum statistic is missing.
    """
    cols = {str(k).lower(): v for k, v in stats.items()}

    pts = []
    if any(k in cols for k in ("min", "minimum", "p0")):
        qmin = float(_first_present(cols, ("min", "minimum", "p0")))
        pts.append((0.0, qmin))
    else:
        raise ValueError("Piecewise sampler requires min")
    percs = {}
    for k, v in list(cols.items()):
        if k.startswith("p") and k[1:].isdigit():
            percs[int(k[1:])] = float(v)
    for p in sorted(percs.keys()):
        if 0 < p < 100:
            pts.append((p / 100.0, percs[p]))

    if any(k in cols for k in ("max", "maximum", "p100")):
        qmax = float(_first_present(cols, ("max", "maximum", "p100")))
        pts.append((1.0, qmax))
    else:
        raise ValueError("Piecewise sampler requires max")
    pts = sorted(pts, key=lambda t: t[0])

    u = self.rng.uniform(0.0, 1.0, size=size)
    samples = np.empty(size, dtype=float)
    for i, ui in enumerate(u):
        for (p0, q0), (p1, q1) in zip(pts[:-1], pts[1:]):
            if p0 <= ui <= p1:
                if p1 == p0:
                    samples[i] = q0
                else:
                    t = (ui - p0) / (p1 - p0)
                    samples[i] = q0 + t * (q1 - q0)
                break
    return samples

def _sample_from_stats(
    self: "Model",
    stats: Dict[str, float],
    kind: Optional[str] = None,
) -> float:
    """Sample one value from validated summary statistics.

    ``kind`` supplies a physical domain. For Normal inputs that domain becomes
    explicit truncation support. Input distributions are assumed to have been
    physically validated once during input loading; this hot-path sampler does
    not revalidate every supplied statistic on every Monte Carlo draw. The
    sampled result is still checked against the requested domain as an internal
    invariant and is never clipped into compliance. Supplied unit metadata is
    converted to canonical units before the physical-domain rules are applied.

    Parameters
    ----------
    self : Model
        Active simulation model instance providing the RNG.
    stats : dict[str, float]
        Summary statistics for one sampled value.
    kind : str or None, optional
        Semantic domain. Supported values are ``efficiency``, ``load_rate``,
        ``nonnegative``, ``fraction``, ``cn``, ``ia_ratio``, ``percent``, and
        ``positive``. Negative efficiencies remain valid; efficiencies above
        one are rejected.

    Returns
    -------
    float
        Sampled numeric value.

    Raises
    ------
    ValueError
        If statistics are insufficient or violate the requested physical
        domain.
    """
    unit_kind = {
        "efficiency": "fraction",
        "load_rate": "load_rate",
        "fraction": "fraction",
        "cn": "dimensionless",
        "ia_ratio": "fraction",
        "percent": "percent",
    }.get(kind)
    cols = convert_sampling_mapping(stats, expected_kind=unit_kind)
    domain: Optional[PhysicalDomain] = physical_domain_for_sampling_kind(kind)
    if kind is not None and domain is None:
        raise ValueError(f"Unknown sampling kind: {kind!r}")

    # Physical/statistical input validation is deliberately performed once in
    # the input-loading layer. Repeating those dictionary scans here would put
    # validation work inside the Monte Carlo hot path.

    has_min = any(k in cols for k in ("min", "minimum", "p0"))
    has_max = any(k in cols for k in ("max", "maximum", "p100"))
    has_sd = any(k in cols for k in ("sd", "std"))
    has_mean = any(k in cols for k in ("mean", "average", "avg"))
    has_percentiles = any(
        str(k).startswith("p") and str(k)[1:].isdigit() for k in cols
    )

    physical_low = domain.low if domain is not None else None
    physical_high = domain.high if domain is not None else None

    if "value" in cols:
        sample = float(cols["value"])
    elif has_min and has_max and has_percentiles:
        sample = float(self._piecewise_quantile_sample(cols, size=1)[0])
    elif has_min and has_max and has_mean and not has_sd:
        mean = float(_first_present(cols, ("mean", "average", "avg")))
        low = float(_first_present(cols, ("min", "minimum", "p0")))
        high = float(_first_present(cols, ("max", "maximum", "p100")))
        sd = max((high - low) / 4.0, 1e-12)
        sample = float(
            self._trunc_normal(mean, sd, low=low, high=high, size=1)[0]
        )
    elif has_min and has_max and not has_mean and not has_sd and not has_percentiles:
        low = float(_first_present(cols, ("min", "minimum", "p0")))
        high = float(_first_present(cols, ("max", "maximum", "p100")))
        sample = float(self.rng.uniform(low, high))
    elif has_mean and has_sd:
        mean = float(_first_present(cols, ("mean", "average", "avg")))
        sd = float(_first_present(cols, ("sd", "std")))
        low = (
            float(_first_present(cols, ("min", "minimum", "p0")))
            if has_min
            else physical_low
        )
        high = (
            float(_first_present(cols, ("max", "maximum", "p100")))
            if has_max
            else physical_high
        )
        sample = float(
            self._trunc_normal(mean, sd, low=low, high=high, size=1)[0]
        )
    else:
        raise ValueError("Insufficient distribution statistics to sample")

    if not np.isfinite(sample):
        raise ValueError("Sampled value must be finite")
    if domain is not None:
        validate_scalar_in_domain(sample, domain, f"sampled {kind}")
    return float(sample)

