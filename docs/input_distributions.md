# Standardized numeric inputs and distributions

[← Back to main README](../readme.md)

## Purpose

Numeric model inputs use one common **inline** row-level convention. A row contains either a fixed value or all statistics needed to define its distribution directly. There is no separate distribution catalog and no named distribution reference.

The convention is used by statistical parcel load rates, PLET/RUSLE numeric parameters, curve number and infiltration fraction, pollutant concentrations, BMP efficiencies, BMP costs, and the [AFO manure excretion reference CSV](afo_mass_based.md#starting-animal-excretion-distributions).

## Canonical distribution columns

Use these columns where applicable:

    value, mean, sd, min, p05, p50, p95, max

Other percentile levels such as `p10`, `p25`, `p75`, and `p90` are also supported. Metadata columns such as `units`, `unit`, `notes`, `citations`, `cost_component`, and `sample_group`, plus identifiers such as `pid`, `pollutant`, `parameter`, `pathway`, or `cps`, depend on the input table.

Recognized aliases remain accepted: `average`/`avg` → `mean`, `std` → `sd`, `minimum` → `min`, `maximum` → `max`, `p0` → `min`, and `p100` → `max`.

## Accepted numeric forms

Each numeric row must use exactly one coherent specification.

| Form | Required statistics | Sampling behavior |
|---|---|---|
| Fixed | `value` | Always returns the supplied value |
| Normal | `mean`, `sd` | Normal sampling, truncated when semantic or physical bounds apply |
| Bounded normal | `mean`, `sd`, `min`, `max` | Truncated normal within the stated bounds |
| Legacy bounded normal | `min`, `mean`, `max` | Truncated normal with inferred `sd = (max - min) / 4` |
| Uniform | `min`, `max` | Uniform between the endpoints |
| Percentile distribution | `min`, one or more `pXX`, `max` | Piecewise-linear inverse-CDF sampling |

Examples:

    Fixed:                 value=8.5
    Normal:                mean=8.5, sd=1.2
    Bounded normal:        mean=8.5, sd=1.2, min=5, max=12
    Legacy bounded normal: mean=8.5, min=5, max=12
    Uniform:               min=5, max=12
    Percentile:            min=5, p10=6, p50=8.5, p90=11, max=12

The loader rejects ambiguous rows: do not mix `value` with distribution statistics; provide both `min` and `max`; do not combine percentile statistics with `mean` or `sd`; and give percentile distributions both endpoints. It also checks finite values, nonnegative `sd`, ordered bounds, and monotonic percentiles.

## Parcel defaults and overrides

For parcel-keyed input tables that allow defaults, `pid IS NULL` defines the default row. An integer `pid` row overrides that default for the matching parcel. Both the default and the override may be fixed values or direct inline distributions.

For example, this defines a watershed-wide precipitation distribution and a different distribution for parcel 127:

    pid    mean    sd    min    max    units
    NULL   42      3     34     50     in/year
    127    46      2     40     52     in/year

Without an explicit `sample_group`, the default distribution is sampled independently for each parcel. Use `sample_group` only when multiple rows are intentionally meant to share the same sampled value within a scenario.

## HUC and domain inheritance

The same distribution columns apply in HUC and domain `input_*` tables. Resolution is **parcel-specific row > assigned-HUC row > domain row > NULL-pid parcel default**, for each variable and natural key. The selected row supplies the entire distribution; statistics are never combined across scopes. Omit a row to fall back rather than supplying a row with no value/statistics.

An inherited HUC or domain row automatically receives a stable sample group. All parcels inheriting it share its draw in that scenario, but draws are resampled in each scenario. Different source rows remain independent unless an explicit `sample_group` requests sharing for the same variable. This applies to both statistical load rates and PLET/RUSLE inputs, including hydrology lookups. Classification and selection/delivery-weight tables remain fixed-value-only. Serial and parallel runs use the same scenario-level grouping.
