# Standardized numeric inputs and distributions

[← Back to main README](../readme.md)

## Purpose

Numeric model inputs use one common row-level convention. The convention is used by:

- statistical-mode parcel pollutant load-rate inputs
- PLET numeric parameters
- the PLET land-cover/HSG Curve Number and infiltration table
- RUSLE parameters
- runoff concentrations
- groundwater concentrations
- BMP efficiencies
- BMP costs

The goal is to make a fixed value and an uncertain input interchangeable without changing the structure of the model, while keeping large parcel datasets manageable.

## Canonical distribution columns

New input files should use these names where applicable:

    value, distribution_id, mean, sd, min, p05, p50, p95, max

Other percentile levels such as `p10`, `p25`, `p75`, and `p90` are also supported. Metadata columns such as `units`, `unit`, `notes`, and identifiers such as `pid`, `pollutant`, `parameter`, `pathway`, or `cps` depend on the input table.

Recognized legacy aliases remain accepted:

- `average` or `avg` → `mean`
- `std` → `sd`
- `minimum` → `min`
- `maximum` → `max`
- `p0` → `min`
- `p100` → `max`

Use the canonical names for new files.

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
| Reusable named distribution | `distribution_id` | Uses a definition from `input_distributions.csv` |

Examples:

    Fixed:                 value=8.5
    Normal:                mean=8.5, sd=1.2
    Bounded normal:        mean=8.5, sd=1.2, min=5, max=12
    Legacy bounded normal: mean=8.5, min=5, max=12
    Uniform:               min=5, max=12
    Percentile:            min=5, p10=6, p50=8.5, p90=11, max=12

### Invalid or ambiguous combinations

The loader rejects ambiguous rows. In particular:

- do not mix `value` with distribution statistics
- do not combine `distribution_id` with inline distribution statistics
- do not provide only one of `min` or `max`
- do not combine percentile statistics with `mean` or `sd`
- percentile distributions require both `min` and `max` endpoints

The loader also checks finite numeric values, nonnegative `sd`, ordered bounds, and monotonic percentile values.

## Reusable distribution catalog

A configuration may define an optional catalog:

    input_distributions: ./path/to/input_distributions.csv

The catalog contains one row per `distribution_id` and uses the same distribution columns:

    distribution_id,value,mean,sd,min,p05,p50,p95,max,units,notes
    annual_precip_default,,42,3,34,,,,50,in/year,Example bounded normal
    runoff_tn_default,3,,,,,,,,mg/L,Fixed value
    cover_crop_surface_tn,,0.35,,0.15,,,,0.55,fraction,Legacy min/mean/max form

A use-site row can then be short:

    pid,parameter,distribution_id,units
    *,annual_precip_in,annual_precip_default,in/year

`distribution_id` reuses the **distribution definition**. It does **not** mean that different parcels receive the same sampled number.

The catalog is most useful when many rows share the same uncertainty assumption. If each parcel genuinely has a unique distribution, put that distribution directly on the parcel row rather than creating thousands of one-use catalog IDs.

## Shared draws versus reused definitions

Assigning the same text label to different variables does not create a multivariate or correlated distribution between those variables.

A wildcard row or reused `distribution_id` also does not create a shared draw by itself. Where an input type supports `sample_group`, use that explicitly when multiple rows are intended to share the same sampled value within a scenario.