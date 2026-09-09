from __future__ import annotations

import pandas as pd
import pytest

from src.input_validation import (
    validate_bmp_selection_table,
    validate_distribution_catalog,
    validate_numeric_distribution_rows,
)


def test_validate_distribution_catalog_rejects_duplicate_ids() -> None:
    catalog = pd.DataFrame(
        [
            {"distribution_id": "rain", "value": 1.0},
            {"distribution_id": "rain", "value": 2.0},
        ]
    )

    with pytest.raises(ValueError, match="duplicate distribution_id"):
        validate_distribution_catalog(catalog)


def test_validate_distribution_catalog_accepts_unique_ids() -> None:
    catalog = pd.DataFrame(
        [
            {"distribution_id": "rain", "mean": 42.0, "sd": 3.0, "min": 30.0, "max": 55.0},
            {"distribution_id": "cn", "value": 78.0},
        ]
    )

    validate_distribution_catalog(catalog)


def test_validate_numeric_distribution_rows_accepts_blank_row() -> None:
    table = pd.DataFrame([{}])

    validate_numeric_distribution_rows(table, "x")


@pytest.mark.parametrize(
    ("table", "pattern"),
    [
        (pd.DataFrame([{"mean": 5.0, "sd": 1.0, "min": 6.0, "max": 10.0}]), "mean must lie between min and max"),
        (pd.DataFrame([{"min": 0.0, "p25": 4.0, "p50": 3.0, "max": 10.0}]), "distribution is not monotonic"),
    ],
)
def test_validate_numeric_distribution_rows_rejects_invalid_bound_relationships(table, pattern) -> None:
    with pytest.raises(ValueError, match=pattern):
        validate_numeric_distribution_rows(table, "x")


def test_validate_bmp_selection_table_rejects_non_integer_cps_values() -> None:
    df = pd.DataFrame(
        {
            "cps": [329.5, 412],
            "probability": [0.5, 0.5],
        }
    )

    with pytest.raises(ValueError, match="finite integers"):
        validate_bmp_selection_table(df, [329, 412])


def test_validate_bmp_selection_table_rejects_duplicate_cps_rows() -> None:
    df = pd.DataFrame(
        {
            "cps": [329, 329],
            "probability": [0.25, 0.75],
        }
    )

    with pytest.raises(ValueError, match="duplicate|duplicated|Duplicate"):
        validate_bmp_selection_table(df, [329])


def test_validate_bmp_selection_table_allows_extra_unknown_cps_rows() -> None:
    df = pd.DataFrame(
        {
            "cps": [329, 999],
            "probability": [0.5, 0.5],
        }
    )

    # Current validator requires configured CPS coverage but does not reject
    # extra CPS rows that are not in the configured set.
    validate_bmp_selection_table(df, [329])


def test_validate_bmp_selection_table_rejects_missing_configured_cps_rows() -> None:
    df = pd.DataFrame(
        {
            "cps": [329],
            "probability": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing probability rows"):
        validate_bmp_selection_table(df, [329, 412])


def test_validate_bmp_selection_table_rejects_negative_probabilities() -> None:
    df = pd.DataFrame(
        {
            "cps": [329, 412],
            "probability": [1.1, -0.1],
        }
    )

    with pytest.raises(ValueError, match="nonnegative"):
        validate_bmp_selection_table(df, [329, 412])


def test_validate_bmp_selection_table_rejects_zero_total_weight() -> None:
    df = pd.DataFrame(
        {
            "cps": [329, 412],
            "probability": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="sum to zero or negative"):
        validate_bmp_selection_table(df, [329, 412])


def test_validate_bmp_selection_table_accepts_valid_weights() -> None:
    df = pd.DataFrame(
        {
            "cps": [329, 412],
            "probability": [0.25, 0.75],
        }
    )

    validate_bmp_selection_table(df, [329, 412])