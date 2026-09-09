from __future__ import annotations

from pathlib import Path

from test_reproducibility_helpers import (
    assert_cross_output_consistency,
    assert_expected_scenarios_present,
    assert_outputs_equal,
    read_canonical_outputs,
    run_config_with_overrides,
)


def test_statistical_mode_repeat_run_reproducible_serial(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = repo_root / "examples" / "east_fork" / "inputs" / "default" / "east_fork.yaml"

    outputs_a = tmp_path / "serial_a"
    outputs_b = tmp_path / "serial_b"

    run_config_with_overrides(
        base_config_path=base_cfg,
        tmp_cfg_path=tmp_path / "serial_a.yaml",
        outputs_dir=outputs_a,
        n_jobs=1,
        n_scenarios=2,
        random_seed=202501,
        bmp_limit_n=5,
        verbose=False,
        monkeypatch=monkeypatch,
    )
    run_config_with_overrides(
        base_config_path=base_cfg,
        tmp_cfg_path=tmp_path / "serial_b.yaml",
        outputs_dir=outputs_b,
        n_jobs=1,
        n_scenarios=2,
        random_seed=202501,
        bmp_limit_n=5,
        verbose=False,
        monkeypatch=monkeypatch,
    )

    a = read_canonical_outputs(outputs_a, expect_load_parameters=False)
    b = read_canonical_outputs(outputs_b, expect_load_parameters=False)

    assert_expected_scenarios_present(a, n_scenarios=2)
    assert_expected_scenarios_present(b, n_scenarios=2)
    assert_cross_output_consistency(a, expect_load_parameters=False)
    assert_cross_output_consistency(b, expect_load_parameters=False)
    assert_outputs_equal(a, b)


def test_statistical_mode_serial_equals_parallel(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = repo_root / "examples" / "east_fork" / "inputs" / "default" / "east_fork.yaml"

    outputs_serial = tmp_path / "serial"
    outputs_parallel = tmp_path / "parallel"

    run_config_with_overrides(
        base_config_path=base_cfg,
        tmp_cfg_path=tmp_path / "serial.yaml",
        outputs_dir=outputs_serial,
        n_jobs=1,
        n_scenarios=2,
        random_seed=202502,
        bmp_limit_n=5,
        verbose=False,
        monkeypatch=monkeypatch,
    )
    run_config_with_overrides(
        base_config_path=base_cfg,
        tmp_cfg_path=tmp_path / "parallel.yaml",
        outputs_dir=outputs_parallel,
        n_jobs=2,
        n_scenarios=2,
        random_seed=202502,
        bmp_limit_n=5,
        verbose=False,
        monkeypatch=monkeypatch,
    )

    serial = read_canonical_outputs(outputs_serial, expect_load_parameters=False)
    parallel = read_canonical_outputs(outputs_parallel, expect_load_parameters=False)

    assert_expected_scenarios_present(serial, n_scenarios=2)
    assert_expected_scenarios_present(parallel, n_scenarios=2)
    assert_cross_output_consistency(serial, expect_load_parameters=False)
    assert_cross_output_consistency(parallel, expect_load_parameters=False)
    assert_outputs_equal(serial, parallel)


def test_plet_mode_repeat_run_reproducible_serial(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = repo_root / "examples" / "east_fork" / "inputs" / "plet" / "east_fork_plet.yaml"

    outputs_a = tmp_path / "plet_serial_a"
    outputs_b = tmp_path / "plet_serial_b"

    run_config_with_overrides(
        base_config_path=base_cfg,
        tmp_cfg_path=tmp_path / "plet_serial_a.yaml",
        outputs_dir=outputs_a,
        n_jobs=1,
        n_scenarios=2,
        random_seed=202503,
        bmp_limit_n=5,
        verbose=False,
        monkeypatch=monkeypatch,
    )
    run_config_with_overrides(
        base_config_path=base_cfg,
        tmp_cfg_path=tmp_path / "plet_serial_b.yaml",
        outputs_dir=outputs_b,
        n_jobs=1,
        n_scenarios=2,
        random_seed=202503,
        bmp_limit_n=5,
        verbose=False,
        monkeypatch=monkeypatch,
    )

    a = read_canonical_outputs(outputs_a, expect_load_parameters=True)
    b = read_canonical_outputs(outputs_b, expect_load_parameters=True)

    assert_expected_scenarios_present(a, n_scenarios=2)
    assert_expected_scenarios_present(b, n_scenarios=2)
    assert_cross_output_consistency(a, expect_load_parameters=True)
    assert_cross_output_consistency(b, expect_load_parameters=True)
    assert_outputs_equal(a, b)


def test_plet_mode_serial_equals_parallel(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = repo_root / "examples" / "east_fork" / "inputs" / "plet" / "east_fork_plet.yaml"

    outputs_serial = tmp_path / "plet_serial"
    outputs_parallel = tmp_path / "plet_parallel"

    run_config_with_overrides(
        base_config_path=base_cfg,
        tmp_cfg_path=tmp_path / "plet_serial.yaml",
        outputs_dir=outputs_serial,
        n_jobs=1,
        n_scenarios=2,
        random_seed=202504,
        bmp_limit_n=5,
        verbose=False,
        monkeypatch=monkeypatch,
    )
    run_config_with_overrides(
        base_config_path=base_cfg,
        tmp_cfg_path=tmp_path / "plet_parallel.yaml",
        outputs_dir=outputs_parallel,
        n_jobs=2,
        n_scenarios=2,
        random_seed=202504,
        bmp_limit_n=5,
        verbose=False,
        monkeypatch=monkeypatch,
    )

    serial = read_canonical_outputs(outputs_serial, expect_load_parameters=True)
    parallel = read_canonical_outputs(outputs_parallel, expect_load_parameters=True)

    assert_expected_scenarios_present(serial, n_scenarios=2)
    assert_expected_scenarios_present(parallel, n_scenarios=2)
    assert_cross_output_consistency(serial, expect_load_parameters=True)
    assert_cross_output_consistency(parallel, expect_load_parameters=True)
    assert_outputs_equal(serial, parallel)