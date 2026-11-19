"""Regression coverage for complete wetland upstream traversal and cycle validation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
)
from src.input_config import _build_parcel_up_map
from src.model import _ScenarioContext
from src.parcel_graph import iter_full_upstream_by_distance
from test_hydrologic_parcel_universe import _model_with_nonselectable_upstream_parcels


def _wetland_run(model, *, wet_area=1.0, catchment_ratio=2.0):
    ctx = _ScenarioContext(
        model.cfg,
        model._shared_payload(),
        logging.getLogger("wetland.graph.test"),
        seed=123,
    )
    draws = iter((wet_area, catchment_ratio))
    ctx._sample_from_stats = lambda stats, kind=None: next(draws)
    ctx.current_pathway_load_rates = np.full((3, 1, 1), 10.0)
    load_rates = np.full((3, 1), 10.0)
    record = {}
    outputs = {
        OUTPUT_TREATED: np.zeros(1),
        OUTPUT_REMOVED: np.zeros(1),
    }
    ctx._simulate_wetland(2, [{"surface": .5}], load_rates, record, outputs)
    return load_rates, record, outputs


def test_three_parcel_chain_reaches_indirectly_upstream_parcel():
    # A -> B -> C (wetland). A is two graph edges from C, not directly adjacent.
    model = _model_with_nonselectable_upstream_parcels()
    model.parcel_up_idxs = [[], [0], [1]]
    loads, record, outputs = _wetland_run(model)
    assert record[OUTPUT_IMPACTED_PIDS] == "C,B,A"
    np.testing.assert_allclose(loads[:, 0], [5., 5., 5.])
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(30.)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(15.)


def test_wetland_partial_last_indirect_upstream_parcel():
    model = _model_with_nonselectable_upstream_parcels()
    model.parcel_up_idxs = [[], [0], [1]]
    loads, record, outputs = _wetland_run(model, wet_area=1, catchment_ratio=1.5)
    assert record[OUTPUT_IMPACTED_PIDS] == "C,B,A"
    assert record[OUTPUT_CATCHMENT_RATIO] == pytest.approx(1.5)
    np.testing.assert_allclose(loads[:, 0], [7.5, 5., 5.])
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(25.)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(12.5)


def test_wetland_shortage_uses_all_upstream_network_and_updates_ratio():
    model = _model_with_nonselectable_upstream_parcels()
    model.parcel_up_idxs = [[], [0], [1]]
    loads, record, outputs = _wetland_run(model, wet_area=1., catchment_ratio=100.)
    assert record[OUTPUT_IMPACTED_PIDS] == "C,B,A"
    assert record[OUTPUT_CATCHMENT_RATIO] == pytest.approx(2.)
    np.testing.assert_allclose(loads[:, 0], [5., 5., 5.])
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(30.)


def test_stochastic_wetland_needs_no_upstream_when_host_area_covers_target():
    model = _model_with_nonselectable_upstream_parcels()
    model.parcel_up_idxs = [[], [0], [1]]
    model.parcel_area_ha[2] = 5.0
    loads, record, outputs = _wetland_run(model, wet_area=1., catchment_ratio=2.)
    # Although the graph contains upstream parcels, the host alone supplies
    # 3 ha of treatment capacity, so neither B nor A may be marked impacted.
    assert record[OUTPUT_IMPACTED_PIDS] == ""
    np.testing.assert_allclose(loads[:, 0], [10., 10., 7.])
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(30.)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(15.)


def test_tied_direct_upstream_parcels_ordered_by_pid_not_input_order():
    model = _model_with_nonselectable_upstream_parcels()
    model.parcel_up_idxs = [[], [], [1, 0]]
    _, record, _ = _wetland_run(model)
    assert record[OUTPUT_IMPACTED_PIDS] == "C,A,B"


def test_branched_graph_nearest_first_then_numeric_pid_and_deduplicated():
    # 100 -> [20,3]; 20 -> [1,2]; 3 -> [2,1]. Both routes
    # reach 1 and 2; each ancestor may be treated only once.
    parcel_ids = ["100", "20", "3", "1", "2"]
    parents = [[1, 2], [4, 3], [3, 4], [], []]
    got = [parcel_ids[i] for i in iter_full_upstream_by_distance(0, parents, parcel_ids)]
    assert got == ["3", "20", "1", "2"]
    assert len(got) == len(set(got))


def test_shortest_path_wins_when_ancestor_reachable_by_multiple_routes():
    # 1 is immediately upstream of root, and also upstream of 2 via 3.
    ids = ["100", "3", "2", "1"]
    parents = [[1, 3], [2], [3], []]
    assert [ids[i] for i in iter_full_upstream_by_distance(0, parents, ids)] == ["1", "3", "2"]


def test_no_upstream_returns_no_impacted_ancestors():
    assert list(iter_full_upstream_by_distance(0, [[], [], []], ["A", "B", "C"])) == []


@pytest.mark.parametrize(
    ("edges", "expected_cycle"),
    [
        ([("1", "1")], "1 -> 1"),
        ([("1", "2"), ("2", "1")], "1 -> 2 -> 1"),
        ([("1", "2"), ("2", "3"), ("3", "1")], "1 -> 2 -> 3 -> 1"),
        # Even an isolated cycle that is unrelated to a wetland is invalid.
        ([("1", "2"), ("2", "1"), ("3", "4")], "1 -> 2 -> 1"),
    ],
)
def test_cycle_detection_prevents_simulation(edges, expected_cycle):
    rows = pd.DataFrame(edges, columns=["pid", "pid_up"])
    modeled = sorted({pid for edge in edges for pid in edge})
    with pytest.raises(ValueError, match="parcel_up.*directed cycle") as exc:
        _build_parcel_up_map(rows, modeled)
    assert expected_cycle in str(exc.value)


def test_valid_diverging_and_merging_network_is_not_rejected():
    edges = pd.DataFrame([
        {"pid": "100", "pid_up": "10"},
        {"pid": "100", "pid_up": "20"},
        {"pid": "10", "pid_up": "1"},
        {"pid": "20", "pid_up": "1"},
        {"pid": "20", "pid_up": "1"},  # duplicate allowed and deduplicated
    ])
    upstream = _build_parcel_up_map(edges, ["1", "10", "20", "100"])
    assert upstream["100"] == ["10", "20"]
    assert upstream["20"] == ["1"]


def test_long_chain_is_validated_iteratively_without_recursion_error():
    size = 1500
    rows = pd.DataFrame({
        "pid": [str(i) for i in range(1, size)],
        "pid_up": [str(i-1) for i in range(1, size)],
    })
    upstream = _build_parcel_up_map(rows, [str(i) for i in range(size)])
    assert len(upstream) == size
    assert upstream[str(size-1)] == [str(size-2)]
    ids = [str(i) for i in range(size)]
    indices = [[]] + [[i-1] for i in range(1, size)]
    ordered = list(iter_full_upstream_by_distance(size-1, indices, ids))
    assert len(ordered) == size - 1
    assert ordered[:3] == [size-2, size-3, size-4]
    assert ordered[-1] == 0


def test_domain_clipping_preserves_modeled_valid_subgraph():
    rows = pd.DataFrame([
        {"pid": "1", "pid_up": "2"},
        {"pid": "2", "pid_up": "3"},
        {"pid": "3", "pid_up": "4"},  # 4 is outside modeled domain
    ])
    result = _build_parcel_up_map(rows, ["1", "2", "3"],
                                  source_parcel_ids=["1", "2", "3", "4"])
    assert result == {"1": ["2"], "2": ["3"], "3": []}
