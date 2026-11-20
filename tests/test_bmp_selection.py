"""Regression tests for the shared BMP eligibility and placement selectors."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.bmp_selection import BMPEligibilityPool
from src.constants import CFG_DETERMINISTIC_PARCEL_ORDER
from src.parcel import _sample_parcel_index


class _Logger:
    def verbose(self, message: str) -> None:
        pass


def _context(pool: BMPEligibilityPool, seed: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        cfg={CFG_DETERMINISTIC_PARCEL_ORDER: pool.deterministic_parcels},
        parcel_selection_ids=pool.selection_pids,
        parcel_selection_probs=pool.parcel_probabilities,
        rng=np.random.default_rng(seed),
        logger=_Logger(),
    )


def _pool(*, pids=("1", "2"), cps=(313, 590), bmp_probs=(0.3, 0.7),
          parcel_probs=(0.5, 0.5), hosts=("1",), deterministic_parcels=False,
          deterministic_bmps=False) -> BMPEligibilityPool:
    return BMPEligibilityPool(
        pids, cps, bmp_probs, {313}, hosts, parcel_probs,
        deterministic_parcels=deterministic_parcels,
        deterministic_bmps=deterministic_bmps,
    )


def test_sparse_afo_host_is_selected_without_retry_exhaustion() -> None:
    """One AFO host among 1000 parcels must be reachable with certainty."""
    pids = tuple(str(i) for i in range(1000))
    pool = _pool(pids=pids, cps=(313,), bmp_probs=(1.0,),
                 parcel_probs=(0.001,) * 1000, hosts=("999",))
    ctx = _context(pool)
    assert pool.active_selection_indices().tolist() == [999]
    selection_idx = _sample_parcel_index(ctx, pool.active_selection_indices())
    assert selection_idx == 999
    cps, _ = pool.select_bmp(pids[selection_idx], ctx.rng, 0)
    assert cps == 313
    pool.record_placement(pids[selection_idx], cps)
    assert pool.active_selection_indices().size == 0


@pytest.mark.parametrize("deterministic_parcels", [False, True])
@pytest.mark.parametrize("deterministic_bmps", [False, True])
def test_selection_modes_share_host_and_nonduplicate_rules(
    deterministic_parcels: bool, deterministic_bmps: bool,
) -> None:
    pool = _pool(deterministic_parcels=deterministic_parcels,
                 deterministic_bmps=deterministic_bmps)
    ctx = _context(pool)
    assert pool.eligible_indices("1") == [0, 1]
    assert pool.eligible_indices("2") == [1]
    # Even if the global deterministic CPS cursor points at 313, the
    # non-host must receive 590.
    cps, cursor = pool.select_bmp("2", ctx.rng, 0)
    assert cps == 590
    pool.record_placement("2", cps)
    assert pool.eligible_indices("2") == []
    assert pool.active_selection_indices().tolist() == [0]
    assert _sample_parcel_index(ctx, pool.active_selection_indices()) == 0
    cps, cursor = pool.select_bmp("1", ctx.rng, cursor)
    pool.record_placement("1", cps)
    assert cps not in {pool.bmp_cps[i] for i in pool.eligible_indices("1")}
    with pytest.raises(ValueError, match="not selectable"):
        pool.record_placement("1", cps)


def test_stochastic_parcel_probabilities_are_conditional_on_active_pool() -> None:
    pool = _pool(pids=("1", "2", "3"), cps=(590,), bmp_probs=(1.0,),
                 hosts=(), parcel_probs=(0.5, 0.3, 0.2))
    pool.record_placement("1", 590)
    ctx = _context(pool, seed=5)
    draws = [_sample_parcel_index(ctx, pool.active_selection_indices()) for _ in range(8000)]
    assert set(draws) == {1, 2}
    assert abs(draws.count(1) / len(draws) - 0.6) < 0.03


def test_stochastic_bmp_probabilities_are_conditional_on_eligibility() -> None:
    pool = _pool(pids=("1",), cps=(313, 590, 340),
                 bmp_probs=(0.6, 0.1, 0.3), parcel_probs=(1.0,), hosts=())
    ctx = _context(pool, seed=9)
    assert pool.eligible_indices("1") == [1, 2]
    draws = [pool.select_bmp("1", ctx.rng, 0)[0] for _ in range(8000)]
    assert set(draws) == {340, 590}
    assert abs(draws.count(590) / len(draws) - 0.25) < 0.03


def test_deterministic_parcel_cursor_skips_exhausted_without_skipping_next() -> None:
    pool = _pool(pids=("10", "2", "3"), cps=(590,), bmp_probs=(1.0,),
                 parcel_probs=(0.4, 0.3, 0.3), deterministic_parcels=True)
    ctx = _context(pool)
    first = _sample_parcel_index(ctx, pool.active_selection_indices())
    assert first == 1  # pid 2
    pool.record_placement("2", 590)
    second = _sample_parcel_index(ctx, pool.active_selection_indices())
    assert second == 2  # pid 3, not pid 10
    pool.record_placement("3", 590)
    assert _sample_parcel_index(ctx, pool.active_selection_indices()) == 0


def test_zero_weight_bmp_excluded_only_from_stochastic_selection() -> None:
    stochastic = _pool(pids=("1",), cps=(590, 313), bmp_probs=(0.0, 1.0),
                       parcel_probs=(1.0,), hosts=())
    assert stochastic.eligible_indices("1") == [0]
    assert stochastic.active_selection_indices().size == 0
    deterministic = _pool(pids=("1",), cps=(590, 313), bmp_probs=(0.0, 1.0),
                          parcel_probs=(1.0,), hosts=(), deterministic_bmps=True)
    assert deterministic.active_selection_indices().tolist() == [0]
    assert deterministic.select_bmp("1", np.random.default_rng(3), 0)[0] == 590


def test_zero_weight_parcel_ignored_stochastically_but_not_deterministically() -> None:
    pool = _pool(pids=("1", "2"), cps=(590,), bmp_probs=(1.0,),
                 hosts=(), parcel_probs=(0.0, 1.0))
    assert pool.active_selection_indices().tolist() == [1]
    deterministic = _pool(pids=("1", "2"), cps=(590,), bmp_probs=(1.0,),
                          hosts=(), parcel_probs=(0.0, 1.0), deterministic_parcels=True)
    assert deterministic.active_selection_indices().tolist() == [0, 1]


def test_same_seed_reproduces_selection_sequence() -> None:
    def simulate(seed: int) -> list[tuple[str, int]]:
        pool = _pool(pids=("1", "2", "3"), cps=(313, 590),
                     bmp_probs=(0.5, 0.5), parcel_probs=(0.2, 0.3, 0.5))
        ctx = _context(pool, seed=seed)
        results = []
        cursor = 0
        while pool.active_selection_indices().size:
            selection_idx = _sample_parcel_index(ctx, pool.active_selection_indices())
            pid = pool.selection_pids[selection_idx]
            cps, cursor = pool.select_bmp(pid, ctx.rng, cursor)
            results.append((pid, cps))
            pool.record_placement(pid, cps)
        return results
    assert simulate(11) == simulate(11)
    assert len(simulate(11)) == 4  # 2 practices on host 1; one each on 2 and 3


def test_no_repeated_or_nonhost_feedlot_application() -> None:
    pool = _pool(deterministic_parcels=True, deterministic_bmps=True)
    ctx = _context(pool)
    cursor = 0
    placements = []
    while pool.active_selection_indices().size:
        idx = _sample_parcel_index(ctx, pool.active_selection_indices())
        pid = pool.selection_pids[idx]
        cps, cursor = pool.select_bmp(pid, ctx.rng, cursor)
        placements.append((pid, cps))
        pool.record_placement(pid, cps)
    assert len(placements) == len(set(placements)) == 3
    assert ("2", 313) not in placements


def test_deterministic_bmp_order_is_numerically_sorted() -> None:
    """Input CPS order must not change ascending deterministic selection."""
    pool = _pool(pids=("1",), cps=(590, 340, 313),
                 bmp_probs=(0.4, 0.3, 0.3), parcel_probs=(1.0,),
                 hosts=("1",), deterministic_bmps=True)
    rng = np.random.default_rng(1)
    cursor = 0
    selected = []
    for _ in range(3):
        cps, cursor = pool.select_bmp("1", rng, cursor)
        selected.append(cps)
        pool.record_placement("1", cps)
    assert selected == [313, 340, 590]
