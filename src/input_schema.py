"""Declarative schema for consolidated model input GeoPackages.

The registry in this module is the single source of truth for user-facing
``input_*`` tables. Its keys describe parcel inputs; HUC tables substitute
``huc_id`` for ``pid`` and domain tables omit the spatial key. Hydrology lookups
retain their classification keys and may optionally include a parcel ``pid``.
Where a variable supports a package-wide default, ``pid IS NULL`` represents
that default and exact integer ``pid`` rows override it. Loaders may still perform
variable-specific semantic validation, but table discovery, key columns,
required modes, and basic input kinds are defined here so the contract is not
spread across several modules.
"""

from __future__ import annotations

from .constants import (
    InputVariableSpec,
    INPUT_SCHEMA_TABLE,
    INPUT_SCHEMA_VERSION,
    INPUT_VARIABLE_SPECS,
    KNOWN_INPUT_TABLES,
)


def required_input_tables(mode: str) -> frozenset[str]:
    """Return tables that must exist across the selected input scopes for a mode."""
    normalized = str(mode).strip().lower()
    return frozenset(
        spec.table
        for spec in INPUT_VARIABLE_SPECS.values()
        if normalized in spec.required_modes
    )
