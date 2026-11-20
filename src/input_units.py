"""Unit validation and conversion for model numeric inputs.

The model uses canonical internal units. When a numeric input row supplies a
``unit`` or ``units`` value, this module validates that unit and converts all
statistics for the row to the canonical internal unit before sampling. Rows
that omit unit metadata retain the historical contract and are assumed to
already use canonical units.

Only unambiguous multiplicative conversions are supported. In particular,
RUSLE R and K factors are not independently converted between empirical unit
systems because those factors are coupled in the RUSLE equation.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .constants import (
    COST_AREA_SPEC,
    COST_LENGTH_SPEC,
    COST_PROJECT_SPEC,
    DIMENSIONLESS_SPEC,
    FRACTION_SPEC,
    PERCENT_SPEC,
    PRECIP_SPEC,
    EVENT_PRECIP_SPEC,
    DAYS_PER_YEAR_SPEC,
    CONCENTRATION_SPEC,
    LOAD_RATE_SPEC,
    MASS_RATE_SPEC,
    RUSLE_R_SPEC,
    RUSLE_K_SPEC,
    UnitSpec,
    _KIND_SPECS,
    _PARAMETER_KINDS,
    _unit_key,
)


def _nonblank(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def row_unit(row: Mapping[str, Any]) -> Optional[Any]:
    """Return a supplied ``units``/``unit`` value, if any."""
    for column in ("units", "unit"):
        if column in row and _nonblank(row.get(column)):
            return row.get(column)
    return None


def parameter_unit_kind(parameter: Any) -> Optional[str]:
    """Return the expected unit kind for a canonical model parameter."""
    name = str(parameter).strip().lower()
    if name.startswith("load_multiplier_"):
        return "dimensionless"
    return _PARAMETER_KINDS.get(name)


def _find_spec(unit: Any, specs: Sequence[UnitSpec]) -> Tuple[UnitSpec, float]:
    key = _unit_key(unit)
    matches = [(spec, spec.aliases[key]) for spec in specs if key in spec.aliases]
    if not matches:
        allowed = sorted({spec.canonical for spec in specs})
        raise ValueError(
            f"Unsupported or dimensionally incompatible unit {unit!r}; "
            f"expected a unit convertible to {allowed}"
        )
    first_spec, first_factor = matches[0]
    if any(abs(factor - first_factor) > 0.0 for _, factor in matches[1:]):
        raise ValueError(f"Ambiguous unit label {unit!r}; provide a more explicit unit")
    return first_spec, float(first_factor)


def infer_unit_kind(
    row: Mapping[str, Any], *, expected_kind: Optional[str] = None
) -> Optional[str]:
    """Infer an expected unit kind from model-row metadata when possible."""
    if expected_kind is not None:
        return expected_kind
    if "parameter" in row and _nonblank(row.get("parameter")):
        return parameter_unit_kind(row.get("parameter"))
    if "cps" in row and "pollutant" in row:
        return "fraction"
    if "cps" in row and row_unit(row) is not None:
        return "cost"
    if "pathway" in row and "pollutant" in row:
        return "load_rate"
    return None


def convert_statistics(
    stats: Mapping[str, Any],
    unit: Any,
    *,
    expected_kind: Optional[str] = None,
) -> Tuple[Dict[str, float], str]:
    """Convert numeric statistics from ``unit`` to a canonical model unit."""
    numeric = {str(key).strip().lower(): float(value) for key, value in stats.items()}
    if not _nonblank(unit):
        return numeric, ""
    if expected_kind is not None:
        if expected_kind not in _KIND_SPECS:
            raise ValueError(f"Unknown expected unit kind {expected_kind!r}")
        spec, factor = _find_spec(unit, _KIND_SPECS[expected_kind])
    else:
        candidates = []
        for kind, specs in _KIND_SPECS.items():
            try:
                spec, factor = _find_spec(unit, specs)
            except ValueError:
                continue
            candidates.append((kind, spec, factor))
        unique = {(spec.canonical, factor) for _, spec, factor in candidates}
        if len(unique) != 1:
            raise ValueError(
                f"Unit {unit!r} requires table or parameter context to determine "
                "the intended physical dimension"
            )
        _, spec, factor = candidates[0]
    return {key: float(value) * factor for key, value in numeric.items()}, spec.canonical


def convert_row_statistics(
    row: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    expected_kind: Optional[str] = None,
) -> Dict[str, float]:
    """Convert one input row's statistics to canonical internal units."""
    unit = row_unit(row)
    if unit is None:
        return {str(key).strip().lower(): float(value) for key, value in stats.items()}
    kind = infer_unit_kind(row, expected_kind=expected_kind)
    converted, _ = convert_statistics(stats, unit, expected_kind=kind)
    return converted


def convert_sampling_mapping(
    stats: Mapping[str, Any],
    *,
    expected_kind: Optional[str] = None,
) -> Dict[str, float]:
    """Extract and unit-normalize statistics from a runtime sampling mapping."""
    metadata = {"unit", "units", "notes", "sample_group"}
    numeric: Dict[str, float] = {}
    for key, value in stats.items():
        label = str(key).strip().lower()
        if label in metadata or pd.isna(value):
            continue
        if label in {
            "value", "mean", "average", "avg", "sd", "std",
            "min", "minimum", "max", "maximum",
        }:
            numeric[label] = float(value)
        elif label.startswith("p") and label[1:].isdigit():
            numeric[label] = float(value)
    unit = row_unit(stats)
    if unit is None:
        return numeric
    converted, _ = convert_statistics(numeric, unit, expected_kind=expected_kind)
    return converted


def canonical_cost_unit(unit: Any) -> Tuple[str, float]:
    """Return canonical BMP cost unit and conversion factor."""
    spec, factor = _find_spec(unit, _KIND_SPECS["cost"])
    return spec.canonical, factor


def convert_cost_value(value: Any, unit: Any) -> float:
    """Convert one BMP cost-rate value to its canonical rate unit."""
    _, factor = canonical_cost_unit(unit)
    return float(value) * factor


def unit_scale_signatures(unit: Any) -> set[Tuple[str, float]]:
    """Return canonical-unit/factor signatures for a recognized unit label."""
    key = _unit_key(unit)
    signatures: set[Tuple[str, float]] = set()
    seen: set[int] = set()
    for specs in _KIND_SPECS.values():
        for spec in specs:
            if id(spec) in seen:
                continue
            seen.add(id(spec))
            if key in spec.aliases:
                signatures.add((spec.canonical, float(spec.aliases[key])))
    return signatures


def unit_labels_same_scale(first: Any, second: Any) -> bool:
    """Return whether two labels interpret unchanged numeric values identically."""
    if _unit_key(first) == _unit_key(second):
        return True
    first_signatures = unit_scale_signatures(first)
    second_signatures = unit_scale_signatures(second)
    return bool(first_signatures) and first_signatures == second_signatures
