import re
from typing import Any, Dict, Iterable, Mapping, Optional

def ci_get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Case-insensitive dictionary lookup for config keys."""
    key_l = key.lower()
    # Iterate through the mapping and compare keys in lowercase.
    # This allows configuration dictionaries to be used with mixed case keys.
    for k, v in d.items():
        if str(k).lower() == key_l:
            return v
    return default

def normalize_columns(df: Any) -> Any:
    """Normalize DataFrame column labels to lowercase strings."""
    # Convert column labels to clean, predictable lowercase strings so
    # downstream joins and lookups are not affected by casing or whitespace.
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def normalize_pollutant_label(label: str) -> str:
    """Normalize pollutant labels to a canonical form used by the model."""
    from .constants import POLLUTANT_ALIAS_MAP

    canonical = POLLUTANT_ALIAS_MAP.get(str(label).strip().lower())
    if canonical is None:
        raise ValueError(f"Unknown pollutant label: {label}")
    return canonical


def parse_percent_keys(cols: Iterable[Any]) -> Dict[int, Any]:
    """Return a mapping from percentile column names to integer keys."""
    # Return a mapping for 'p5', 'p10', etc.
    # Percentile keys are normalized to integer values for easier sorting
    # and lookup by percentile rank.
    percents: Dict[int, Any] = {}
    for c in cols:
        c_l = str(c).lower().strip()
        m = re.fullmatch(r"p(\d{1,2}|100)", c_l)
        if m:
            p = int(m.group(1))
            percents[p] = c
    return percents
