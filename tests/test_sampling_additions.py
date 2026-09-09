from __future__ import annotations

import numpy as np
import pytest

from src.sampling import (
    _piecewise_quantile_sample,
    _sample_from_stats,
    _trunc_normal,
)


class DummySampler:
    _trunc_normal = _trunc_normal
    _piecewise_quantile_sample = _piecewise_quantile_sample
    _sample_from_stats = _sample_from_stats

    def __init__(self, seed: int = 123) -> None:
        self.rng = np.random.default_rng(seed)


def test_trunc_normal_sd_zero_returns_mean_within_support() -> None:
    sampler = DummySampler(seed=1)

    out = sampler._trunc_normal(mean=2.5, sd=0.0, low=2.0, high=3.0, size=4)

    assert out.shape == (4,)
    assert np.allclose(out, 2.5)


def test_trunc_normal_sd_zero_outside_support_raises() -> None:
    sampler = DummySampler(seed=1)

    with pytest.raises(ValueError, match="support|bound|outside"):
        sampler._trunc_normal(mean=5.0, sd=0.0, low=0.0, high=4.0, size=1)


def test_trunc_normal_respects_explicit_bounds() -> None:
    sampler = DummySampler(seed=7)

    samples = sampler._trunc_normal(mean=0.5, sd=0.75, low=0.0, high=1.0, size=500)

    assert samples.shape == (500,)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    assert np.unique(np.round(samples, 6)).size > 50


def test_piecewise_quantile_sample_with_min_p50_max_stays_in_support() -> None:
    sampler = DummySampler(seed=4)

    cols = {"min": 10.0, "p50": 20.0, "max": 30.0}
    samples = sampler._piecewise_quantile_sample(cols, size=400)

    assert samples.shape == (400,)
    assert np.all(samples >= 10.0)
    assert np.all(samples <= 30.0)
    assert np.any(samples < 20.0)
    assert np.any(samples > 20.0)


def test_piecewise_quantile_sample_with_named_aliases_stays_in_support() -> None:
    sampler = DummySampler(seed=9)

    cols = {"minimum": -4.0, "p50": -1.0, "maximum": 2.0}
    samples = sampler._piecewise_quantile_sample(cols, size=200)

    assert np.all(samples >= -4.0)
    assert np.all(samples <= 2.0)
    assert np.any(samples < -1.0)
    assert np.any(samples > -1.0)


def test_sample_from_stats_uses_uniform_when_only_min_and_max_are_present() -> None:
    sampler = DummySampler(seed=5)

    samples = np.array(
        [sampler._sample_from_stats({"min": 2.0, "max": 5.0}, kind="nonnegative") for _ in range(200)]
    )

    assert np.all(samples >= 2.0)
    assert np.all(samples <= 5.0)
    assert len(set(np.round(samples, 8))) > 20


def test_sample_from_stats_min_mean_max_without_sd_uses_bounded_sampling() -> None:
    sampler = DummySampler(seed=11)

    samples = np.array(
        [
            sampler._sample_from_stats(
                {"min": 0.2, "mean": 0.5, "max": 0.8},
                kind="fraction",
            )
            for _ in range(200)
        ]
    )

    assert np.all(samples >= 0.2)
    assert np.all(samples <= 0.8)
    assert samples.mean() == pytest.approx(0.5, abs=0.1)


def test_sample_from_stats_accepts_mean_average_avg_aliases() -> None:
    sampler = DummySampler(seed=8)

    a = sampler._sample_from_stats({"average": 5.0, "sd": 0.0}, kind="nonnegative")
    b = sampler._sample_from_stats({"avg": 7.0, "std": 0.0}, kind="nonnegative")

    assert a == pytest.approx(5.0)
    assert b == pytest.approx(7.0)


def test_sample_from_stats_accepts_p0_and_p100_aliases() -> None:
    sampler = DummySampler(seed=3)

    samples = np.array(
        [
            sampler._sample_from_stats(
                {"p0": -2.0, "mean": -1.0, "p100": 1.0},
                kind="efficiency",
            )
            for _ in range(250)
        ]
    )

    assert np.all(samples >= -2.0)
    assert np.all(samples <= 1.0)
    assert np.any(samples < 0.0)
    assert np.any(samples > 0.0)


def test_sample_from_stats_invalid_fixed_value_outside_domain_is_rejected() -> None:
    sampler = DummySampler(seed=1)

    with pytest.raises(ValueError, match="allowed physical domain"):
        sampler._sample_from_stats({"value": -3.0}, kind="nonnegative")

    with pytest.raises(ValueError, match="allowed physical domain"):
        sampler._sample_from_stats({"value": 1.2}, kind="efficiency")


def test_sample_from_stats_rejects_insufficient_statistics() -> None:
    sampler = DummySampler(seed=1)

    with pytest.raises(ValueError, match="Insufficient distribution statistics"):
        sampler._sample_from_stats({"mean": 4.0}, kind="nonnegative")

    with pytest.raises(ValueError, match="Insufficient distribution statistics"):
        sampler._sample_from_stats({"sd": 2.0}, kind="nonnegative")


def test_sample_from_stats_fraction_uniform_support_within_domain_stays_in_bounds() -> None:
    sampler = DummySampler(seed=12)

    draws = np.array(
        [sampler._sample_from_stats({"min": 0.1, "max": 0.3}, kind="fraction") for _ in range(200)]
    )

    assert np.all(draws >= 0.1)
    assert np.all(draws <= 0.3)
    assert np.any(draws > 0.1)
    assert np.any(draws < 0.3)


def test_sample_from_stats_efficiency_allows_negative_values_but_not_above_one() -> None:
    sampler = DummySampler(seed=2)

    negative_value = sampler._sample_from_stats({"value": -0.4}, kind="efficiency")
    assert negative_value == pytest.approx(-0.4)

    with pytest.raises(ValueError, match="allowed physical domain"):
        sampler._sample_from_stats({"value": 1.01}, kind="efficiency")


def test_sample_from_stats_seeded_sampling_is_reproducible() -> None:
    s1 = DummySampler(seed=99)
    s2 = DummySampler(seed=99)

    draws1 = [
        s1._sample_from_stats({"mean": 10.0, "sd": 2.5, "min": 5.0, "max": 15.0}, kind="nonnegative")
        for _ in range(25)
    ]
    draws2 = [
        s2._sample_from_stats({"mean": 10.0, "sd": 2.5, "min": 5.0, "max": 15.0}, kind="nonnegative")
        for _ in range(25)
    ]

    assert draws1 == pytest.approx(draws2)


def test_sample_from_stats_respects_explicit_bounds_over_broader_physical_domain() -> None:
    sampler = DummySampler(seed=23)

    samples = np.array(
        [
            sampler._sample_from_stats(
                {"mean": 50.0, "sd": 20.0, "min": 40.0, "max": 45.0},
                kind="nonnegative",
            )
            for _ in range(100)
        ]
    )

    assert np.all(samples >= 40.0)
    assert np.all(samples <= 45.0)


def test_sample_from_stats_with_cross_zero_efficiency_distribution_keeps_both_signs() -> None:
    sampler = DummySampler(seed=44)
    stats = {"min": -0.30, "mean": -0.05, "max": 0.20}

    draws = np.array([sampler._sample_from_stats(stats, kind="efficiency") for _ in range(300)])

    assert np.all(draws >= -0.30)
    assert np.all(draws <= 0.20)
    assert np.any(draws < 0.0)
    assert np.any(draws > 0.0)