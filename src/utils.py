"""
Utility helpers (stateless).

Contains small utilities used across the codebase:
- Case-insensitive config lookup
- DataFrame column normalization
- Pollutant label normalization to canonical names
- Percentile key parsing
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def ci_get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return a value from a mapping using a case-insensitive key lookup.

    Parameters
    ----------
    d : Mapping[str, Any]
        Source dictionary-like object.
    key : str
        Key to look up, case-insensitively.
    default : Any, optional
        Default value when key is not found.

    Returns
    -------
    Any
        Value if found; otherwise default.
    """
    key_l = str(key).lower()
    for k, v in d.items():
        if str(k).lower() == key_l:
            return v
    return default


def normalize_columns(df: Any) -> Any:
    """Normalize DataFrame column labels to lowercase strings (in place)."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def normalize_pollutant_label(label: str) -> str:
    """Normalize a pollutant label to a canonical form.

    Parameters
    ----------
    label : str
        Arbitrary label (e.g., 'tp', 'TP', 'total phosphorus').

    Returns
    -------
    str
        Canonical label recognized by the model.

    Raises
    ------
    ValueError
        If the label cannot be mapped to a known canonical name.

    Notes
    -----
    The mapping is defined by constants.POLLUTANT_ALIAS_MAP.
    """
    from .constants import POLLUTANT_ALIAS_MAP

    canonical = POLLUTANT_ALIAS_MAP.get(str(label).strip().lower())
    if canonical is None:
        raise ValueError(f"Unknown pollutant label: {label}")
    return canonical


def parse_percent_keys(cols: Iterable[Any]) -> Dict[int, Any]:
    """Parse percentile-style column labels (e.g., 'p5', 'p50', 'p95').

    Parameters
    ----------
    cols : Iterable[Any]
        Column labels to inspect.

    Returns
    -------
    Dict[int, Any]
        Mapping from percentile integer (5, 50, 95, 100, ...) to the original label.
    """
    import re

    percents: Dict[int, Any] = {}
    for c in cols:
        c_l = str(c).lower().strip()
        m = re.fullmatch(r"p(\d{1,2}|100)", c_l)
        if m:
            p = int(m.group(1))
            percents[p] = c
    return percents