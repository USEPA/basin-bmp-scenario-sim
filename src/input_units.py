"""Unit validation and conversion for model numeric inputs.

The model uses canonical internal units.  When a numeric input row supplies a
``unit`` or ``units`` value, this module validates that unit and converts all
statistics for the row to the canonical internal unit before sampling.  Rows
that omit unit metadata retain the historical contract and are assumed to
already use canonical units.

Only unambiguous multiplicative conversions are supported.  In particular,
RUSLE R and K factors are not independently converted between empirical unit
systems because those factors are coupled in the RUSLE equation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class UnitSpec:
    """Describe one canonical unit and accepted conversion aliases.

    Parameters
    ----------
    canonical : str
        Canonical unit label used internally.
    aliases : Mapping[str, float]
        Normalized input-unit labels mapped to multiplicative conversion
        factors that yield the canonical unit.
    """

    canonical: str
    aliases: Mapping[str, float]


def _unit_key(value: Any) -> str:
    """Normalize a human-authored unit label for comparison."""
    text = str(value).strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = text.replace("²", "2").replace("^2", "2")
    text = text.replace("·", "*").replace("×", "*")
    text = text.replace("$", "usd")
    text = text.replace("_", " ")
    text = re.sub(r"\bper\b", "/", text)
    text = re.sub(r"\s+", "", text)
    return text


def _spec(canonical: str, aliases: Mapping[str, float]) -> UnitSpec:
    normalized = {_unit_key(canonical): 1.0}
    normalized.update({_unit_key(label): float(factor) for label, factor in aliases.items()})
    return UnitSpec(canonical=canonical, aliases=normalized)


FRACTION_SPEC = _spec(
    "fraction",
    {
        "ratio": 1.0,
        "decimal": 1.0,
        "dimensionless": 1.0,
        "unitless": 1.0,
        "1": 1.0,
        "percent": 0.01,
        "pct": 0.01,
        "%": 0.01,
    },
)
PERCENT_SPEC = _spec(
    "percent",
    {
        "pct": 1.0,
        "%": 1.0,
        "fraction": 100.0,
        "ratio": 100.0,
        "decimal": 100.0,
        "dimensionless": 100.0,
        "unitless": 100.0,
        "1": 100.0,
    },
)
DIMENSIONLESS_SPEC = _spec(
    "dimensionless",
    {
        "unitless": 1.0,
        "1": 1.0,
        "ratio": 1.0,
        "index": 1.0,
    },
)
PRECIP_SPEC = _spec(
    "in/year",
    {
        "in/yr": 1.0,
        "inch/year": 1.0,
        "inches/year": 1.0,
        "inch/yr": 1.0,
        "inches/yr": 1.0,
        "mm/year": 1.0 / 25.4,
        "mm/yr": 1.0 / 25.4,
        "cm/year": 1.0 / 2.54,
        "cm/yr": 1.0 / 2.54,
        "m/year": 100.0 / 2.54,
        "m/yr": 100.0 / 2.54,
    },
)
DAYS_PER_YEAR_SPEC = _spec(
    "days/year",
    {
        "day/year": 1.0,
        "days/yr": 1.0,
        "day/yr": 1.0,
        "d/year": 1.0,
        "d/yr": 1.0,
    },
)
CONCENTRATION_SPEC = _spec(
    "mg/L",
    {
        "mg/liter": 1.0,
        "mg/litre": 1.0,
        "ug/L": 1.0e-3,
        "ug/liter": 1.0e-3,
        "ug/litre": 1.0e-3,
        "g/L": 1.0e3,
        "g/liter": 1.0e3,
        "g/litre": 1.0e3,
        "mg/m3": 1.0e-3,
        "g/m3": 1.0,
        "kg/m3": 1.0e3,
    },
)
LOAD_RATE_SPEC = _spec(
    "kg/ha/yr",
    {
        "kg/ha/year": 1.0,
        "kg/hectare/yr": 1.0,
        "kg/hectare/year": 1.0,
        "kg/ac/yr": 1.0 / 0.40468564224,
        "kg/acre/yr": 1.0 / 0.40468564224,
        "lb/ha/yr": 0.45359237,
        "lb/hectare/yr": 0.45359237,
        "lb/ac/yr": 0.45359237 / 0.40468564224,
        "lb/acre/yr": 0.45359237 / 0.40468564224,
        "lb/ac/year": 0.45359237 / 0.40468564224,
        "lb/acre/year": 0.45359237 / 0.40468564224,
        "g/m2/yr": 10.0,
        "g/m2/year": 10.0,
        "mg/m2/yr": 0.01,
        "mg/m2/year": 0.01,
        "kg/km2/yr": 0.01,
        "kg/km2/year": 0.01,
    },
)
MASS_RATE_SPEC = _spec(
    "kg/yr",
    {
        "kg/year": 1.0,
        "g/yr": 1.0e-3,
        "g/year": 1.0e-3,
        "mg/yr": 1.0e-6,
        "mg/year": 1.0e-6,
        "lb/yr": 0.45359237,
        "lb/year": 0.45359237,
        "ton/yr": 907.18474,
        "ton/year": 907.18474,
        "tons/yr": 907.18474,
        "tons/year": 907.18474,
        "metric ton/yr": 1000.0,
        "metric ton/year": 1000.0,
        "tonne/yr": 1000.0,
        "tonne/year": 1000.0,
    },
)
COST_AREA_SPEC = _spec(
    "usd/ha",
    {
        "usd/hectare": 1.0,
        "usd/ac": 1.0 / 0.40468564224,
        "usd/acre": 1.0 / 0.40468564224,
        "usd/m2": 10000.0,
        "usd per unit area": 1.0,
    },
)
COST_LENGTH_SPEC = _spec(
    "usd/m",
    {
        "usd/meter": 1.0,
        "usd/metre": 1.0,
        "usd/ft": 1.0 / 0.3048,
        "usd/foot": 1.0 / 0.3048,
        "usd/km": 0.001,
        "usd per unit length": 1.0,
    },
)
COST_PROJECT_SPEC = _spec(
    "usd/project",
    {
        "usd/bmp": 1.0,
        "usd/practice": 1.0,
        "usd/each": 1.0,
        "usd/unit": 1.0,
    },
)
# RUSLE R and K are empirical factors whose numerical units are coupled.  The
# conversion layer intentionally accepts only the canonical formulation rather
# than pretending that either factor can be converted independently.
RUSLE_R_SPEC = _spec(
    "rusle-r-us-customary",
    {
        "us customary rusle r": 1.0,
        "us-customary-rusle-r": 1.0,
        "plet-r": 1.0,
        # Current PLET/RUSLE example inputs use ``index`` for the R factor.
        # It is accepted as the model's established customary R convention;
        # no cross-system numerical conversion is attempted.
        "index": 1.0,
    },
)
RUSLE_K_SPEC = _spec(
    "rusle-k-us-customary",
    {
        "us customary rusle k": 1.0,
        "us-customary-rusle-k": 1.0,
        "plet-k": 1.0,
        # Established PLET customary K-factor label used by the example data.
        "ton acre hour/(acre foot ton inch)": 1.0,
    },
)

_KIND_SPECS: Dict[str, Tuple[UnitSpec, ...]] = {
    "fraction": (FRACTION_SPEC,),
    "percent": (PERCENT_SPEC,),
    "dimensionless": (DIMENSIONLESS_SPEC,),
    "precip": (PRECIP_SPEC,),
    "days_per_year": (DAYS_PER_YEAR_SPEC,),
    "concentration": (CONCENTRATION_SPEC,),
    "load_rate": (LOAD_RATE_SPEC,),
    "mass_rate": (MASS_RATE_SPEC,),
    "cost": (COST_AREA_SPEC, COST_LENGTH_SPEC, COST_PROJECT_SPEC),
    "rusle_r": (RUSLE_R_SPEC,),
    "rusle_k": (RUSLE_K_SPEC,),
}

_PARAMETER_KINDS: Dict[str, str] = {
    "annual_precip_in": "precip",
    "rain_days": "days_per_year",
    "rain_correction_fraction": "fraction",
    "runoff_day_fraction": "fraction",
    "cn": "dimensionless",
    "ia_ratio": "fraction",
    "infiltration_fraction": "fraction",
    "runoff_multiplier": "dimensionless",
    "groundwater_multiplier": "dimensionless",
    "r": "rusle_r",
    "k": "rusle_k",
    "ls": "dimensionless",
    "c": "dimensionless",
    "p": "dimensionless",
    "sdr": "fraction",
    "sediment_multiplier": "dimensionless",
    "sediment_delivery_multiplier": "dimensionless",
    "sediment_n_pct": "percent",
    "sediment_p_pct": "percent",
    "enrichment_ratio": "dimensionless",
    "fraction_subsurface_shallow": "fraction",
}


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
    # Overlapping aliases are safe only when they imply the same conversion.
    first_spec, first_factor = matches[0]
    if any(abs(factor - first_factor) > 0.0 for _, factor in matches[1:]):
        raise ValueError(f"Ambiguous unit label {unit!r}; provide a more explicit unit")
    return first_spec, float(first_factor)


def infer_unit_kind(row: Mapping[str, Any], *, expected_kind: Optional[str] = None) -> Optional[str]:
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
        # With no table/parameter context, recognize only a unique unit family.
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
    metadata = {"unit", "units", "notes", "distribution_id", "sample_group"}
    numeric: Dict[str, float] = {}
    for key, value in stats.items():
        label = str(key).strip().lower()
        if label in metadata or pd.isna(value):
            continue
        if label in {"value", "mean", "average", "avg", "sd", "std", "min", "minimum", "max", "maximum"}:
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
