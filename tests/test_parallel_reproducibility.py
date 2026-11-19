from __future__ import annotations

import logging

import pandas as pd

from model_test_cases import make_plet_rusle_case, make_statistical_case
from test_reproducibility_helpers import (
    assert_cross_output_consistency,
    assert_expected_scenarios_present,
    assert_outputs_equal,
    read_canonical_outputs,
)
from src.model import Model
from src.constants import (
    CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
    CFG_DETERMINISTIC_BMP_TYPE_ORDER,
    CFG_DETERMINISTIC_PARCEL_ORDER,
    DATA_BMP_EFFICIENCY,
    DATA_CPS,
    DATA_PARCELS,
    DATA_PARCEL_P,
    DATA_PARCEL_DOWN_MAP,
    DATA_PARCEL_OUT_MAP,
    DATA_POLLUTANT_LOAD_RATE,
)


def _run_case(builder, outputs_dir, *, n_jobs: int, random_seed: int) -> None:
    cfg, data = builder(
        outputs_dir,
        n_jobs=n_jobs,
        n_scenarios=2,
        random_seed=random_seed,
        bmp_limit_n=3,
    )
    logger = logging.getLogger(f"reproducibility.{outputs_dir.name}")
    Model(cfg, data, logger).run_all_scenarios()


def test_statistical_mode_repeat_run_reproducible_serial(tmp_path) -> None:
    outputs_a = tmp_path / "serial_a"
    outputs_b = tmp_path / "serial_b"

    _run_case(make_statistical_case, outputs_a, n_jobs=1, random_seed=202501)
    _run_case(make_statistical_case, outputs_b, n_jobs=1, random_seed=202501)

    a = read_canonical_outputs(outputs_a, expect_load_parameters=False)
    b = read_canonical_outputs(outputs_b, expect_load_parameters=False)

    assert_expected_scenarios_present(a, n_scenarios=2)
    assert_expected_scenarios_present(b, n_scenarios=2)
    assert_cross_output_consistency(a, expect_load_parameters=False)
    assert_cross_output_consistency(b, expect_load_parameters=False)
    assert_outputs_equal(a, b)


def test_statistical_mode_serial_equals_parallel(tmp_path) -> None:
    outputs_serial = tmp_path / "serial"
    outputs_parallel = tmp_path / "parallel"

    _run_case(make_statistical_case, outputs_serial, n_jobs=1, random_seed=202502)
    _run_case(make_statistical_case, outputs_parallel, n_jobs=2, random_seed=202502)

    serial = read_canonical_outputs(outputs_serial, expect_load_parameters=False)
    parallel = read_canonical_outputs(outputs_parallel, expect_load_parameters=False)

    assert_expected_scenarios_present(serial, n_scenarios=2)
    assert_expected_scenarios_present(parallel, n_scenarios=2)
    assert_cross_output_consistency(serial, expect_load_parameters=False)
    assert_cross_output_consistency(parallel, expect_load_parameters=False)
    assert_outputs_equal(serial, parallel)


def test_plet_mode_repeat_run_reproducible_serial(tmp_path) -> None:
    outputs_a = tmp_path / "plet_serial_a"
    outputs_b = tmp_path / "plet_serial_b"

    _run_case(make_plet_rusle_case, outputs_a, n_jobs=1, random_seed=202503)
    _run_case(make_plet_rusle_case, outputs_b, n_jobs=1, random_seed=202503)

    a = read_canonical_outputs(outputs_a, expect_load_parameters=True)
    b = read_canonical_outputs(outputs_b, expect_load_parameters=True)

    assert_expected_scenarios_present(a, n_scenarios=2)
    assert_expected_scenarios_present(b, n_scenarios=2)
    assert_cross_output_consistency(a, expect_load_parameters=True)
    assert_cross_output_consistency(b, expect_load_parameters=True)
    assert_outputs_equal(a, b)


def test_plet_mode_serial_equals_parallel(tmp_path) -> None:
    outputs_serial = tmp_path / "plet_serial"
    outputs_parallel = tmp_path / "plet_parallel"

    _run_case(make_plet_rusle_case, outputs_serial, n_jobs=1, random_seed=202504)
    _run_case(make_plet_rusle_case, outputs_parallel, n_jobs=2, random_seed=202504)

    serial = read_canonical_outputs(outputs_serial, expect_load_parameters=True)
    parallel = read_canonical_outputs(outputs_parallel, expect_load_parameters=True)

    assert_expected_scenarios_present(serial, n_scenarios=2)
    assert_expected_scenarios_present(parallel, n_scenarios=2)
    assert_cross_output_consistency(serial, expect_load_parameters=True)
    assert_cross_output_consistency(parallel, expect_load_parameters=True)
    assert_outputs_equal(serial, parallel)


def test_deterministic_design_produces_identical_scenario_trajectories(tmp_path) -> None:
    cfg, data = make_statistical_case(
        tmp_path, n_jobs=2, n_scenarios=2, random_seed=17, bmp_limit_n=6
    )
    cfg.update({
        CFG_DETERMINISTIC_PARCEL_ORDER: True,
        CFG_DETERMINISTIC_BMP_TYPE_ORDER: True,
        CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS: True,
    })
    pid_map = {"P1": "1", "P2": "2", "P3": "3"}
    for key in (DATA_PARCELS, DATA_PARCEL_P, DATA_POLLUTANT_LOAD_RATE, "delivery_ratios"):
        data[key]["pid"] = data[key]["pid"].map(pid_map)
    for key in (DATA_PARCEL_DOWN_MAP, DATA_PARCEL_OUT_MAP):
        data[key] = {pid_map[pid]: values for pid, values in data[key].items()}
    data[DATA_CPS] = [412, 656]
    data[DATA_BMP_EFFICIENCY]["cps"] = data[DATA_BMP_EFFICIENCY]["cps"].replace({340: 412, 345: 656})

    Model(cfg, data, logging.getLogger("test_deterministic_design")).run_all_scenarios()

    a = pd.read_parquet(tmp_path / "bmps" / "s1.parquet")
    b = pd.read_parquet(tmp_path / "bmps" / "s2.parquet")
    pd.testing.assert_frame_equal(a.drop(columns="scenario"), b.drop(columns="scenario"))
    assert set(a["cps"]) == {412, 656}
    trajectories = pd.read_parquet(
        tmp_path / "outlet_trajectories" / "all_scenarios.parquet"
    )
    for (_, _, _, _), group in trajectories.groupby(["pollutant", "oid", "x_axis", "y_axis"]):
        first = group[group["scenario"] == 1].sort_values("step").reset_index(drop=True)
        second = group[group["scenario"] == 2].sort_values("step").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            first.drop(columns="scenario"), second.drop(columns="scenario")
        )
