from __future__ import annotations

import math

import pandas as pd
import pytest

from src.constants import (
    CFG_PARCEL_P,
    COL_OID,
    COL_OIDS,
    COL_PID,
    COL_PROBABILITY,
)
from src.input_config import _expand_pid_defaults, _load_parcel_selection


class RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.verbose_messages: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def verbose(self, message: str) -> None:
        self.verbose_messages.append(str(message))

    def info(self, message: str) -> None:
        pass


def _parcels(*pids: str) -> pd.DataFrame:
    return pd.DataFrame({COL_PID: list(pids)})


def test_parcel_p_wildcard_expands_and_exact_pid_overrides(tmp_path) -> None:
    path = tmp_path / "parcel_p.csv"
    pd.DataFrame(
        [
            {COL_PID: "*", COL_PROBABILITY: 1.0},
            {COL_PID: "B", COL_PROBABILITY: 2.0},
        ]
    ).to_csv(path, index=False)

    loaded = _load_parcel_selection(
        {CFG_PARCEL_P: str(path)},
        _parcels("A", "B", "C"),
        RecordingLogger(),
    )

    got = dict(zip(loaded[COL_PID].astype(str), loaded[COL_PROBABILITY].astype(float)))
    assert got.keys() == {"A", "B", "C"}
    assert math.isclose(got["A"], 0.25)
    assert math.isclose(got["B"], 0.50)
    assert math.isclose(got["C"], 0.25)
    assert math.isclose(sum(got.values()), 1.0)


def test_parcel_p_without_wildcard_still_allows_subset(tmp_path) -> None:
    path = tmp_path / "parcel_p.csv"
    pd.DataFrame(
        [
            {COL_PID: "A", COL_PROBABILITY: 1.0},
            {COL_PID: "C", COL_PROBABILITY: 3.0},
        ]
    ).to_csv(path, index=False)

    loaded = _load_parcel_selection(
        {CFG_PARCEL_P: str(path)},
        _parcels("A", "B", "C"),
        RecordingLogger(),
    )

    assert loaded[COL_PID].astype(str).tolist() == ["A", "C"]
    assert loaded[COL_PROBABILITY].astype(float).tolist() == pytest.approx([0.25, 0.75])


def test_parcel_out_wildcard_group_is_replaced_by_any_exact_pid_rows() -> None:
    df = pd.DataFrame(
        [
            {COL_PID: "*", COL_OIDS: "1"},
            {COL_PID: "*", COL_OIDS: "3"},
            {COL_PID: "B", COL_OIDS: "2"},
        ]
    )

    expanded = _expand_pid_defaults(
        df,
        ["A", "B", "C"],
        label="parcel_out",
        logger=RecordingLogger(),
        key_columns=None,
    )

    grouped = {
        pid: rows[COL_OIDS].astype(str).tolist()
        for pid, rows in expanded.groupby(COL_PID, sort=False)
    }
    assert grouped["A"] == ["1", "3"]
    assert grouped["B"] == ["2"]
    assert grouped["C"] == ["1", "3"]


def test_delivery_ratio_wildcards_are_overridden_per_pid_and_oid() -> None:
    cols = ["sdr_f_to_s", "sdr_s_to_o", "ndr_f_to_s", "ndr_s_to_o"]
    df = pd.DataFrame(
        [
            {COL_PID: "*", COL_OID: "1", **{c: 0.5 for c in cols}},
            {COL_PID: "*", COL_OID: "2", **{c: 0.6 for c in cols}},
            {COL_PID: "B", COL_OID: "1", **{c: 0.9 for c in cols}},
        ]
    )

    expanded = _expand_pid_defaults(
        df,
        ["A", "B"],
        label="delivery_ratios",
        logger=RecordingLogger(),
        key_columns=[COL_OID],
    )

    got = {
        (str(row[COL_PID]), str(row[COL_OID])): float(row["sdr_f_to_s"])
        for _, row in expanded.iterrows()
    }
    assert got == {
        ("A", "1"): 0.5,
        ("A", "2"): 0.6,
        ("B", "1"): 0.9,
        ("B", "2"): 0.6,
    }


def test_unknown_exact_pid_is_removed_and_logged() -> None:
    logger = RecordingLogger()
    df = pd.DataFrame(
        [
            {COL_PID: "*", COL_PROBABILITY: 1.0},
            {COL_PID: "MISSING", COL_PROBABILITY: 2.0},
        ]
    )

    expanded = _expand_pid_defaults(
        df,
        ["A", "B"],
        label="parcel_p",
        logger=logger,
        key_columns=[],
    )

    assert expanded[COL_PID].astype(str).tolist() == ["A", "B"]
    assert any("MISSING" in message for message in logger.warnings)


def test_multiple_parcel_p_wildcard_rows_are_rejected() -> None:
    df = pd.DataFrame(
        [
            {COL_PID: "*", COL_PROBABILITY: 1.0},
            {COL_PID: "*", COL_PROBABILITY: 2.0},
        ]
    )

    with pytest.raises(ValueError, match="at most one pid='\\*' default row"):
        _expand_pid_defaults(
            df,
            ["A", "B"],
            label="parcel_p",
            logger=RecordingLogger(),
            key_columns=[],
        )

from src.constants import COL_PID_UP
from src.input_config import _build_parcel_up_map


def test_parcel_up_blank_wildcard_defaults_to_no_upstream_with_exact_exceptions() -> None:
    df = pd.DataFrame(
        [
            {COL_PID: "*", COL_PID_UP: None},
            {COL_PID: "B", COL_PID_UP: "A"},
            {COL_PID: "C", COL_PID_UP: "A, B"},
        ]
    )

    got = _build_parcel_up_map(df, ["A", "B", "C", "D"])

    assert got == {
        "A": [],
        "B": ["A"],
        "C": ["A", "B"],
        "D": [],
    }


def test_parcel_up_blank_wildcard_alone_means_all_parcels_have_no_upstream() -> None:
    df = pd.DataFrame([{COL_PID: "*", COL_PID_UP: "   "}])

    assert _build_parcel_up_map(df, ["A", "B"]) == {"A": [], "B": []}


def test_parcel_up_wildcard_rejects_nonblank_upstream_value() -> None:
    df = pd.DataFrame([{COL_PID: "*", COL_PID_UP: "A"}])

    with pytest.raises(ValueError, match="wildcard may only declare the default of no upstream parcels"):
        _build_parcel_up_map(df, ["A", "B"])


def test_parcel_up_rejects_multiple_blank_wildcard_rows() -> None:
    df = pd.DataFrame(
        [
            {COL_PID: "*", COL_PID_UP: None},
            {COL_PID: "*", COL_PID_UP: ""},
        ]
    )

    with pytest.raises(ValueError, match="at most one pid='\\*' row"):
        _build_parcel_up_map(df, ["A", "B"])
