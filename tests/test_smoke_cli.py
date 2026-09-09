from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import run_model


def test_example_config_runs_end_to_end(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_cfg = repo_root / "examples" / "east_fork" / "inputs" / "default" / "east_fork.yaml"
    cfg = yaml.safe_load(example_cfg.read_text(encoding="utf-8"))
    cfg["outputs"] = str(tmp_path / "outputs")
    cfg["n_scenarios"] = 1
    cfg["bmp_limit_n"] = 5
    cfg["parallel"] = {"n_jobs": 1}
    cfg["verbose"] = False

    smoke_cfg = tmp_path / "smoke.yaml"
    smoke_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["run_model.py", str(smoke_cfg)])

    run_model.main()
    outputs = tmp_path / "outputs"
    assert (outputs / "log.txt").exists()
    assert (outputs / "logs" / "s1.txt").exists()
    assert (outputs / "bmps" / "s1.parquet").exists()
    assert (outputs / "parcels" / "s1.parquet").exists()
    assert (outputs / "scenario_metrics" / "s1.parquet").exists()
    assert (outputs / "outlet_trajectories" / "all_scenarios.parquet").exists()


def test_plet_groundwater_is_rain_corrected_and_unchanged_by_bmps(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_cfg = repo_root / "examples" / "east_fork" / "inputs" / "plet" / "east_fork_plet.yaml"
    cfg = yaml.safe_load(example_cfg.read_text(encoding="utf-8"))

    cfg["outputs"] = str(tmp_path / "outputs")
    cfg["n_scenarios"] = 1
    cfg["bmp_limit_n"] = 5
    cfg["parallel"] = {"n_jobs": 1}
    cfg["verbose"] = True
    smoke_cfg = tmp_path / "plet_smoke.yaml"
    smoke_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["run_model.py", str(smoke_cfg)])

    run_model.main()
    outputs = tmp_path / "outputs"
    load_parameters = pd.read_parquet(outputs / "load_parameters" / "s1.parquet")
    parcels = pd.read_parquet(outputs / "parcels" / "s1.parquet")
    merged = load_parameters.merge(parcels, on=["scenario", "pid"], validate="one_to_one")

    # Cropland on HSG B resolves to PLET's 0.300 infiltration fraction.
    assert np.allclose(
        load_parameters["initial_annual_infiltration_in"],
        42.0 * 0.90 * 0.300,
    )
    assert np.allclose(load_parameters["initial_cn"], 78.0)
    assert set(load_parameters["initial_land_cover"]) == {"cropland"}
    assert set(load_parameters["initial_hsg"]) == {"B"}

    # In PLET/RUSLE mode, infiltration-derived nutrient load is the canonical
    # subsurface pathway. The example's subsurface BMP efficiencies are zero,
    # so that pathway must remain unchanged by BMP application.
    for pollutant in ("tn", "tp"):
        initial_subsurface_load_rate = load_parameters[
            f"initial_subsurface_{pollutant}_load_rate_kg_ha_yr"
        ]
        final_subsurface_load_rate = load_parameters[
            f"final_subsurface_{pollutant}_load_rate_kg_ha_yr"
        ]
        assert np.allclose(initial_subsurface_load_rate, final_subsurface_load_rate)

        initial_components = (
            merged[f"initial_surface_{pollutant}_load_rate_kg_ha_yr"]
            + merged[f"initial_subsurface_{pollutant}_load_rate_kg_ha_yr"]
        )
        final_components = (
            merged[f"final_surface_{pollutant}_load_rate_kg_ha_yr"]
            + merged[f"final_subsurface_{pollutant}_load_rate_kg_ha_yr"]
        )
        pollutant_upper = pollutant.upper()
        assert np.allclose(
            merged[f"baseline_load_rate_{pollutant_upper}_kg_ha_yr"],
            initial_components,
        )
        assert np.allclose(
            merged[f"final_load_rate_{pollutant_upper}_kg_ha_yr"],
            final_components,
        )
