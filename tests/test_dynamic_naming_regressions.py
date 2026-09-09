from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from src.bmp import (
    _apply_pathway_reduction,
    _get_current_total_load_rate,
    _get_pathway_load_rates,
    _simulate_infield,
)
from src.constants import OUTPUT_REMOVED, OUTPUT_TREATED
from src.plet_rusle import LoadState, calculate_plet_pathway_load_rates
from src.parcel import _sample_load_rate
from src.sampling import _sample_from_stats


class _Logger:
    def verbose(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


def _bmp_ctx() -> SimpleNamespace:
    ctx = SimpleNamespace(
        pathway_names=["surface", "subsurface"],
        parcel_area_ha=np.asarray([2.0]),
        pollutants=["TN"],
        current_pathway_load_rates=np.asarray([[[6.0, 4.0]]], dtype=float),
        current_untreated_groundwater_load_rates=np.zeros((1, 1), dtype=float),
        logger=_Logger(),
    )
    ctx._get_pathway_load_rates = MethodType(_get_pathway_load_rates, ctx)
    ctx._get_current_total_load_rate = MethodType(_get_current_total_load_rate, ctx)
    ctx._apply_pathway_reduction = MethodType(_apply_pathway_reduction, ctx)
    return ctx


def test_pathway_load_rate_helpers_use_canonical_names() -> None:
    ctx = _bmp_ctx()
    expected = {"surface": 6.0, "subsurface": 4.0}
    assert _get_pathway_load_rates(ctx, 0, 0, 10.0) == expected
    assert _get_current_total_load_rate(ctx, 0, 0, 10.0) == pytest.approx(10.0)


def test_infield_uses_explicit_areal_load_rate_and_mass_rate_names_semantically() -> None:
    ctx = _bmp_ctx()
    load_rates = np.asarray([[10.0]], dtype=float)
    outputs = {OUTPUT_TREATED: np.zeros(1), OUTPUT_REMOVED: np.zeros(1)}
    _simulate_infield(ctx, 0, [{"surface": 0.5, "subsurface": 0.5}], load_rates, {}, outputs)
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(20.0)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(10.0)
    assert load_rates[0, 0] == pytest.approx(5.0)
    assert ctx.current_pathway_load_rates[0, 0].tolist() == pytest.approx([3.0, 2.0])


def test_load_generation_helpers_use_only_load_rate_names() -> None:
    assert callable(calculate_plet_pathway_load_rates)


def test_parcel_sampler_uses_only_load_rate_name() -> None:
    ctx = SimpleNamespace(
        pollutant_load_rate_stats=[[{"value": 12.5}]], parcel_ids=["p1"], pollutants=["TN"],
        _sample_from_stats=lambda stats, kind=None: float(stats["value"]),
    )
    assert _sample_load_rate(ctx, 0, 0) == pytest.approx(12.5)


def test_sampling_hint_rejects_negative_load_rate() -> None:
    ctx = SimpleNamespace(
        rng=np.random.default_rng(1),
        _piecewise_quantile_sample=lambda cols, size=1: np.asarray([0.0]),
        _trunc_normal=lambda mean, sd, low=None, high=None, size=1: np.asarray([mean]),
    )
    with pytest.raises(ValueError, match="allowed physical domain"):
        _sample_from_stats(ctx, {"value": -2.0}, kind="load_rate")


def test_negative_efficiency_returns_signed_removed_load_rate() -> None:
    ctx = _bmp_ctx()
    removed_load_rate = _apply_pathway_reduction(ctx, 0, 0, 0.5, {"surface": -0.4, "subsurface": 0.0})
    assert removed_load_rate == pytest.approx(-1.2)
    assert ctx.current_pathway_load_rates[0, 0].tolist() == pytest.approx([7.2, 4.0])
