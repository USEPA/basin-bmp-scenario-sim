from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from src.constants import (
    CFG_BUFFER_DEPTH_FT,
    CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_PORTION_TREATED,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
    OUTPUT_WETLAND_AREA,
)
from src.plet_rusle import initialize_plet_rusle_state
from src.model import Model, _ScenarioContext


def _model_with_nonselectable_upstream_parcels() -> Model:
    """Build a minimal Model whose hydrologic universe is larger than parcel_p."""
    model = Model.__new__(Model)
    model.cfg = {}
    model.data = {DATA_AVG_AREA_HA: 1.0, DATA_AVG_PERIM_M: 100.0}

    model.parcel_ids = ["A", "B", "C"]
    model.pid_to_index = {"A": 0, "B": 1, "C": 2}
    model.pollutants = ["TN"]
    model.parcel_area_ha = [1.0, 1.0, 1.0]
    model.parcel_perim_m = [100.0, 100.0, 100.0]
    model.parcel_out_oids = [["1"], ["1"], ["1"]]
    model.parcel_up_idxs = [[], [], [0, 1]]

    # Only C may receive a BMP. A and B still exist hydrologically.
    model.parcel_selection_ids = ["C"]
    model.parcel_selection_probs = np.asarray([1.0], dtype=float)
    model.selection_source_idxs = [2]

    model.outlet_oids = ["1"]
    model.outlet_target_map = {}
    model.outlet_mean_map = {}
    model.delivery_coeffs = {}
    model.bmp_efficiency_stats = {656: [{"surface": {"value": 0.5}}]}
    model.pollutant_load_rate_stats = [
        [{"value": 10.0}],
        [{"value": 10.0}],
        [{"value": 10.0}],
    ]
    model.load_generation = {"mode": "statistical"}
    model.load_generation_mode = "statistical"
    model.plet_inputs = None
    model.rusle_inputs = None
    model.pollutant_concentrations = None
    model.groundwater_concentrations = None
    model.pathway_names = ["surface"]
    model.pollutant_load_rate_pathway_fractions = {"surface": 1.0}
    model.pollutant_load_rate_is_aggregate = True
    model.groundwater_loads = False
    model.bmp_cps = [656]
    model.bmp_selection_probs = np.asarray([1.0], dtype=float)
    return model


def test_shared_payload_separates_hydrologic_and_selection_universes() -> None:
    model = _model_with_nonselectable_upstream_parcels()

    shared = model._shared_payload()

    assert shared["parcel_ids"] == ["A", "B", "C"]
    assert shared["pid_to_index"] == {"A": 0, "B": 1, "C": 2}
    assert shared["parcel_up_idxs"][2] == [0, 1]
    assert shared["parcel_selection_ids"] == ["C"]
    assert shared["selection_source_idxs"].tolist() == [2]
    assert len(shared["pollutant_load_rate_stats"]) == 3
    # These lightweight test models do not define optional AFO attributes.
    assert shared["afo_inputs"] is None
    assert shared["feedlot_bmp_cps"] == set()



def test_shared_payload_preserves_configured_afo_state() -> None:
    """The optional defaults must not erase real AFO inputs."""
    model = _model_with_nonselectable_upstream_parcels()
    afo_inputs = {"operations": [{"afo_id": "farm-1", "pid": "A"}]}
    model.afo_inputs = afo_inputs
    model.feedlot_bmp_cps = {341, 342}

    shared = model._shared_payload()

    assert shared["afo_inputs"] is afo_inputs
    assert shared["feedlot_bmp_cps"] == {341, 342}


def test_wetland_treats_nonselectable_upstream_parcels() -> None:
    model = _model_with_nonselectable_upstream_parcels()
    shared = model._shared_payload()
    logger = logging.getLogger("test_element18_wetland")
    ctx = _ScenarioContext({}, shared, logger, seed=1)

    # Force a 1-ha wetland with a 2:1 catchment ratio, requiring all three
    # 1-ha hydrologic parcels (C plus nonselectable A and B).
    draws = iter([1.0, 2.0])
    ctx._sample_from_stats = lambda stats, kind=None: next(draws)

    load_rates = np.full((3, 1), 10.0, dtype=float)
    ctx.current_pathway_load_rates = np.full((3, 1, 1), 10.0, dtype=float)
    bmp_rec = {}
    bmp_outputs = {
        OUTPUT_TREATED: np.zeros(1, dtype=float),
        OUTPUT_REMOVED: np.zeros(1, dtype=float),
    }

    ctx._simulate_wetland(
        2,
        [{"surface": 0.5}],
        load_rates,
        bmp_rec,
        bmp_outputs,
    )

    assert bmp_rec[OUTPUT_IMPACTED_PIDS] == "C,A,B"
    assert np.allclose(load_rates[:, 0], [5.0, 5.0, 5.0])
    assert bmp_outputs[OUTPUT_TREATED][0] == 30.0
    assert bmp_outputs[OUTPUT_REMOVED][0] == 15.0


def test_deterministic_wetland_design_uses_only_selected_parcel() -> None:
    model = _model_with_nonselectable_upstream_parcels()
    model.cfg = {CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS: True}
    model.parcel_area_ha[2] = 2.0
    ctx = _ScenarioContext(model.cfg, model._shared_payload(), logging.getLogger("test_fixed_wetland"), seed=1)
    ctx._sample_from_stats = lambda *args, **kwargs: pytest.fail("design must not be sampled")
    ctx.current_pathway_load_rates = np.full((3, 1, 1), 10.0, dtype=float)
    load_rates = np.full((3, 1), 10.0, dtype=float)
    record = {}
    outputs = {
        OUTPUT_TREATED: np.zeros(1, dtype=float),
        OUTPUT_REMOVED: np.zeros(1, dtype=float),
    }

    ctx._simulate_wetland(2, [{"surface": 0.5}], load_rates, record, outputs)

    assert record[OUTPUT_WETLAND_AREA] == pytest.approx(0.4)
    assert record[OUTPUT_CATCHMENT_RATIO] == pytest.approx(4.0)
    assert record[OUTPUT_IMPACTED_PIDS] == ""
    np.testing.assert_allclose(load_rates[:, 0], [10.0, 10.0, 5.0])
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(20.0)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("deterministic", "draws", "length_fraction", "treated_fraction"),
    [
        (True, [], 0.25, 0.25),
        (False, [0.1, 0.3], 0.1, 0.3),
    ],
)
def test_grassed_design_respects_config(
    deterministic, draws, length_fraction, treated_fraction
) -> None:
    model = _model_with_nonselectable_upstream_parcels()
    model.cfg = {
        CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS: deterministic,
        CFG_BUFFER_DEPTH_FT: 35.0,
    }
    model.parcel_area_ha[2] = 2.0
    model.parcel_perim_m[2] = 100.0
    ctx = _ScenarioContext(model.cfg, model._shared_payload(), logging.getLogger("test_grassed_design"), seed=1)
    draw_iter = iter(draws)
    sampled = []

    def sample(*args, **kwargs):
        sampled.append(1)
        return next(draw_iter)

    ctx._sample_from_stats = sample
    ctx.current_pathway_load_rates = np.full((3, 1, 1), 10.0, dtype=float)
    load_rates = np.full((3, 1), 10.0, dtype=float)
    record = {}
    outputs = {
        OUTPUT_TREATED: np.zeros(1, dtype=float),
        OUTPUT_REMOVED: np.zeros(1, dtype=float),
    }

    ctx._simulate_grassed(2, [{"surface": 0.5}], load_rates, record, outputs)

    assert len(sampled) == len(draws)
    assert record[OUTPUT_LINEAR_LENGTH] == pytest.approx(100.0 * length_fraction)
    assert record[OUTPUT_BUFFER_AREA] == pytest.approx(100.0 * length_fraction * 35.0 * 0.3048 / 10000.0)
    assert record[OUTPUT_PORTION_TREATED] == pytest.approx(treated_fraction)
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(20.0 * treated_fraction)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(10.0 * treated_fraction)


def test_plet_initialization_uses_full_hydrologic_parcel_universe(monkeypatch) -> None:
    import src.plet_rusle as lg

    ctx = SimpleNamespace(
        parcel_ids=["A", "B", "C"],
        parcel_selection_ids=["C"],
        plet_inputs=object(),
        rusle_inputs=None,
        pollutant_concentrations=object(),
        groundwater_concentrations=None,
        pollutants=["TSS"],
    )

    seen_parameter_calls = []

    def fake_sample_parameter_table(_ctx, _table, parcel_ids, *, cache_prefix):
        ids = list(parcel_ids)
        seen_parameter_calls.append((cache_prefix, ids))
        if cache_prefix == "plet":
            return [
                {
                    "annual_precip_in": 40.0,
                    "rain_days": 100.0,
                    "rain_correction_fraction": 0.9,
                    "runoff_day_fraction": 0.3,
                    "land_cover": "cropland",
                    "hsg": "B",
                }
                for _ in ids
            ]
        return [{} for _ in ids]

    def fake_sample_concentrations(_ctx, table, parcel_ids):
        ids = list(parcel_ids)
        if table is ctx.pollutant_concentrations:
            return [{"TSS": 1.0} for _ in ids]
        return [{} for _ in ids]

    monkeypatch.setattr(lg, "_sample_parameter_table", fake_sample_parameter_table)
    monkeypatch.setattr(lg, "_sample_concentrations", fake_sample_concentrations)
    monkeypatch.setattr(
        lg,
        "_sample_plet_hydrology",
        lambda *_args, **_kwargs: {"cn": 75.0, "infiltration_fraction": 0.2},
    )
    monkeypatch.setattr(
        lg,
        "calculate_plet_pathway_load_rates",
        lambda *_args, **_kwargs: np.asarray([[1.0, 0.0]], dtype=float),
    )

    baseline, state = initialize_plet_rusle_state(ctx)

    assert seen_parameter_calls[0] == ("plet", ["A", "B", "C"])
    assert state.parcel_ids == ["A", "B", "C"]
    assert baseline.shape == (3, 1)
    assert np.allclose(baseline[:, 0], 1.0)
