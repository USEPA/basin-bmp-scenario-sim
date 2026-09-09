from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

import run_model


def run_config_with_overrides(
    *,
    base_config_path: Path,
    tmp_cfg_path: Path,
    outputs_dir: Path,
    n_jobs: int,
    n_scenarios: int,
    random_seed: int,
    bmp_limit_n: int = 5,
    verbose: bool = False,
    monkeypatch=None,
) -> Path:
    """
    Materialize a temporary config, run the CLI entry point, and return outputs_dir.
    """
    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    cfg["outputs"] = str(outputs_dir)
    cfg["n_scenarios"] = int(n_scenarios)
    cfg["bmp_limit_n"] = int(bmp_limit_n)
    cfg["random_seed"] = int(random_seed)
    cfg["parallel"] = {"n_jobs": int(n_jobs)}
    cfg["verbose"] = bool(verbose)

    tmp_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    if monkeypatch is not None:
        monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)
        monkeypatch.setattr("sys.argv", ["run_model.py", str(tmp_cfg_path)])

    run_model.main()
    return outputs_dir


def canonical_output_paths(outputs_dir: Path, *, expect_load_parameters: bool) -> Dict[str, Path]:
    paths = {
        "bmps_s1": outputs_dir / "bmps" / "s1.parquet",
        "parcels_s1": outputs_dir / "parcels" / "s1.parquet",
        "scenario_metrics_s1": outputs_dir / "scenario_metrics" / "s1.parquet",
        "outlet_trajectories_all": outputs_dir / "outlet_trajectories" / "all_scenarios.parquet",
    }
    if expect_load_parameters:
        paths["load_parameters_s1"] = outputs_dir / "load_parameters" / "s1.parquet"
    return paths


def assert_canonical_outputs_exist(outputs_dir: Path, *, expect_load_parameters: bool) -> None:
    for name, path in canonical_output_paths(
        outputs_dir, expect_load_parameters=expect_load_parameters
    ).items():
        assert path.exists(), f"Missing canonical output {name}: {path}"


def read_canonical_outputs(
    outputs_dir: Path, *, expect_load_parameters: bool
) -> Dict[str, pd.DataFrame]:
    assert_canonical_outputs_exist(outputs_dir, expect_load_parameters=expect_load_parameters)
    return {
        name: pd.read_parquet(path)
        for name, path in canonical_output_paths(
            outputs_dir, expect_load_parameters=expect_load_parameters
        ).items()
    }


def _stable_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(sorted(df.columns), axis=1)


def _stable_sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)

    preferred = ["scenario", "pid", "cps", "pollutant", "outlet", "x_axis", "y_axis", "x", "y"]
    present_preferred = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in present_preferred]
    sort_cols = present_preferred + remaining

    return df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype(str)

    out = _stable_sort_columns(out)
    out = _stable_sort_rows(out)
    return out


def assert_outputs_equal(
    outputs_a: Dict[str, pd.DataFrame],
    outputs_b: Dict[str, pd.DataFrame],
    *,
    atol: float = 1e-12,
    rtol: float = 0.0,
) -> None:
    assert set(outputs_a) == set(outputs_b), "Canonical output sets differ between runs"

    for name in sorted(outputs_a):
        left = normalize_df(outputs_a[name])
        right = normalize_df(outputs_b[name])
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_like=False,
            atol=atol,
            rtol=rtol,
        )


def _expected_scenarios_for_output(name: str, n_scenarios: int) -> set[int]:
    if name.endswith("_s1"):
        return {1}
    if name.endswith("_all"):
        return set(range(1, int(n_scenarios) + 1))
    return set(range(1, int(n_scenarios) + 1))


def assert_expected_scenarios_present(
    outputs: Dict[str, pd.DataFrame],
    *,
    n_scenarios: int,
) -> None:
    for name, df in outputs.items():
        if "scenario" not in df.columns:
            continue
        expected = _expected_scenarios_for_output(name, n_scenarios)
        got = set(pd.to_numeric(df["scenario"], errors="raise").astype(int).tolist())
        assert got == expected, f"{name} scenario ids differ: expected {expected}, got {got}"


def assert_cross_output_consistency(
    outputs: Dict[str, pd.DataFrame],
    *,
    expect_load_parameters: bool,
) -> None:
    metrics = outputs["scenario_metrics_s1"] if "scenario_metrics_s1" in outputs else None
    traj = outputs["outlet_trajectories_all"]

    assert "scenario" in traj.columns, "outlet_trajectories must include scenario"

    if metrics is not None and "scenario" in metrics.columns:
        metrics_scenarios = set(pd.to_numeric(metrics["scenario"], errors="raise").astype(int))
        traj_scenarios = set(pd.to_numeric(traj["scenario"], errors="raise").astype(int))
        assert metrics_scenarios.issubset(traj_scenarios), (
            "scenario_metrics scenario ids must be represented in outlet_trajectories"
        )

    parcels = outputs["parcels_s1"]
    assert "scenario" in parcels.columns, "parcels must include scenario"

    if expect_load_parameters:
        lp = outputs["load_parameters_s1"]
        assert "scenario" in lp.columns, "load_parameters must include scenario"
        assert "pid" in parcels.columns and "pid" in lp.columns, (
            "parcels and load_parameters must include pid for consistency checks"
        )

        parcels_keys = set(
            zip(
                pd.to_numeric(parcels["scenario"], errors="raise").astype(int),
                parcels["pid"].astype(str),
            )
        )
        lp_keys = set(
            zip(
                pd.to_numeric(lp["scenario"], errors="raise").astype(int),
                lp["pid"].astype(str),
            )
        )
        assert parcels_keys == lp_keys, "parcels and load_parameters keys differ"