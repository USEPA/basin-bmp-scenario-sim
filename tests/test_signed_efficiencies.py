from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.bmp import _apply_pathway_reduction
from src.sampling import (
    _piecewise_quantile_sample,
    _sample_from_stats,
    _trunc_normal,
)


class _Sampler:
    """Minimal sampling context with the helpers used by the model."""

    _trunc_normal = _trunc_normal
    _piecewise_quantile_sample = _piecewise_quantile_sample

    def __init__(self, seed: int = 1234) -> None:
        self.rng = np.random.default_rng(seed)


@pytest.mark.parametrize(
    "stats",
    [
        {"value": -0.25},
        {"mean": -0.25, "sd": 0.0},
        {"min": -0.40, "max": -0.20},
        {"min": -0.40, "mean": -0.30, "max": -0.20},
        {"min": -0.40, "p50": -0.30, "max": -0.20},
    ],
)
def test_efficiency_sampler_preserves_negative_values(stats: dict[str, float]) -> None:
    sampled = _sample_from_stats(_Sampler(), stats, kind="efficiency")
    assert sampled < 0.0


def test_efficiency_sampler_rejects_value_above_upper_bound() -> None:
    with pytest.raises(ValueError, match="allowed physical domain"):
        _sample_from_stats(_Sampler(), {"value": 1.25}, kind="efficiency")


def test_cross_zero_efficiency_distribution_retains_adverse_outcomes() -> None:
    sampler = _Sampler()
    stats = {"min": -0.30, "mean": -0.05, "max": 0.15}
    sampled = np.array(
        [_sample_from_stats(sampler, stats, kind="efficiency") for _ in range(250)]
    )
    assert np.all(sampled >= stats["min"])
    assert np.all(sampled <= stats["max"])
    assert np.any(sampled < 0.0)
    assert np.any(sampled > 0.0)


def test_negative_efficiency_increases_pathway_load() -> None:
    model = SimpleNamespace(
        current_pathway_load_rates=np.array([[[10.0, 20.0, 30.0]]], dtype=float)
    )

    removed_load_rate = _apply_pathway_reduction(
        model,
        parcel_idx=0,
        pol_idx=0,
        treatment_fraction=0.5,
        eff_map={
            "surface": -0.40,
            "shallow subsurface": 0.0,
            "deep subsurface": 0.0,
        },
    )

    np.testing.assert_allclose(
        model.current_pathway_load_rates[0, 0, :],
        np.array([12.0, 20.0, 30.0]),
    )
    assert removed_load_rate == pytest.approx(-2.0)
