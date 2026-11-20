# Input and output reference

[← Back to main README](../readme.md)

## Physical input files

A typical run uses:

```text
inputs/
  config/model.yaml
  domain/domain.gpkg
  hucs/hucs.gpkg                # optional, selected by huc_parameters
  parcels/parcels.gpkg
  outlets/outlets.gpkg
  bmps/bmp_efficiency.csv
  bmps/bmp_cost.csv              # optional
```

Model variables may be stored at parcel, HUC, or domain scale. GeoPackage files may be renamed or moved by updating their YAML paths; the directory name does not control behavior. Internal table names are the model schema.

## `parcels.gpkg`

### Structural tables

`parcels` is the required spatial layer. It contains only parcel identity and geometry:

- `pid` — unique integer parcel identifier and the GeoPackage feature/primary key (`INTEGER PRIMARY KEY`). There is no separate `fid` parcel identifier.
- geometry — polygon/multipolygon geometry.

All parcel references in other GeoPackage tables (`pid`, `pid_down`) are integers. Non-spatial attribute tables use a technical integer `id` primary key where QGIS needs a stable editable row identifier; the model ignores that `id`.

Parcel area and perimeter are derived from geometry after clipping to the domain.

**`parcel_down` is required**, even when it contains zero rows. Each row
contains one source parcel PID and its **immediate downstream** parcel PID:

```text
pid | pid_down
1   | 3
2   | 3
3   | 4
```

Every `pid` appears **at most once**; a parcel with no downstream neighbor,
including a watershed outlet, has **no row**. Multiple sources may flow to one
recipient. Every referenced PID must exist in the parcel geometry layer and
self-links or directed cycles are invalid. The source-to-downstream graph is
reversed **in memory** to find the complete upstream catchment of a wetland.
The model cannot read the old `parcel_up` table and provides no runtime
conversion or fallback. Convert old datasets using the explicit migration tool
`python utils/migrate_parcel_down.py --help`, or regenerate links with Whitebox.

The representative link uses the largest share of runoff originating in the
source parcel, even when less than half of that parcel drains to the recipient.
Exactly tied shares use the lower numerical downstream PID. The simplification
routes the **entire modeled parcel** through the chosen link. It does not split
parcel pollutant mass among downstream branches. Some parcels will have no
selected connection because water leaves the parcel study area or a link would
form a cycle. Those cases are logged and do not prohibit automatic update.

`parcel_outlets` is required and contains one parcel-to-outlet relationship per row:

```text
pid | oid
1   | O1
1   | O2
```

For an immediate connection `pid = p` to `pid_down = q`, the
`parcel_outlets` assignments must be consistent with downstream routing:
**every modeled outlet associated with `q` must also be associated with `p`**.
Thus `outlets(q) ⊆ outlets(p)`. This allows nested monitoring outlets (an
upstream parcel may have additional gauges) and separate watershed outlets,
but rejects a connection between incompatible outlet-membership sets.
Parcels without a selected downstream neighbor are not constrained by this
rule. These relationships are validated after parcel-domain clipping, before
Monte Carlo simulation; the error reports the offending PIDs and outlet IDs.
When building `parcel_outlets`, include *all* modeled downstream monitoring
outlets that receive each parcel's load, not merely its nearest gauge.

The structural tables do not contain modeled input variables.

A parcel GeoPackage may also contain the metadata table:

```text
model_input_schema
schema_version
3
```

Schema version `3` identifies the integer-PID/NULL-default one-variable-per-table layout with inline-only distributions (no named distribution IDs). Unknown
`input_*` tables are rejected so misspelled variable names cannot be silently
ignored.

### Editing in QGIS

Schema-v3 GeoPackages are designed to be editable directly in QGIS. The spatial `parcels` layer uses `pid` itself as its integer feature/primary key. Attribute-only tables have a separate technical `id INTEGER PRIMARY KEY` so QGIS can identify and edit individual rows; this `id` is not a parcel identifier and is ignored by the model.

For a default row, leave `pid` empty/NULL in QGIS. Do not enter `*`. For delivery-ratio defaults, leave both `pid` and `oid` NULL. Parcel-specific rows must use an existing integer `pid`. Because relationship and input tables reference `parcels.pid`, changing an existing parcel PID should be treated as a schema/data migration rather than an ordinary attribute edit.

## One variable per `input_*` table

Every parcel-side user input variable has its own GeoPackage attribute table. Numeric tables use the same standardized value/distribution columns:

```text
value, mean, sd, min,
p05, p10, p25, p50, p75, p90, p95, max,
sample_group, units, notes
```


Most parcel-specific variable tables are keyed by integer `pid`. Where a table supports a package-wide default, leave `pid` as SQL `NULL`; an exact integer `pid` row overrides that default. The legacy string `"*"` is invalid in PID fields.

### Parcel selection

`input_selection_weight`

```text
pid | value | units | notes
```

Weights are deterministic, finite, and nonnegative. If the table is absent, parcel selection is uniform. A row with `pid IS NULL` may supply a default.

### HUC and domain parameter GeoPackages

Both load-generation modes support the following hierarchy, independently for each variable and natural key:

**parcel-specific row > assigned-HUC row > domain row > NULL-pid parcel default**

Select one HUC package with `huc_parameters: ../hucs/hucs.gpkg`. The spatial layer is named `hucs`:

```text
huc_id   | huc_level | geometry
05090202 | 8         | Polygon/MultiPolygon
```

`huc_id` is text to preserve leading zeroes. Supported lengths are 2, 4, 6, 8, 10, 12, 14, and 16 digits. `huc_level` is optional; if present it must match the code length. A package must use one level, unique IDs, valid non-overlapping polygons, and a declared CRS. The loader rejects mixed/nested levels rather than choosing a precedence between them.

The `domain:` GeoPackage retains its domain boundary and may also contain `input_*` attribute tables. It represents **one run-wide domain**, not multiple independently assigned domains. Prefer the spatial layer name `domain`; its features may describe a multipart boundary.

Reuse the same variable table names and value/distribution columns at every scope. Replace the parcel key with `huc_id` at HUC scope; omit it at domain scope:

| Variable tables | Parcel keys | HUC keys | Domain keys |
|---|---|---|---|
| PLET/RUSLE parameter tables, selection weight | `pid` | `huc_id` | none |
| `input_pollutant_load_rate` | `pid`, `pollutant`, optional `pathway` | `huc_id`, `pollutant`, optional `pathway` | `pollutant`, optional `pathway` |
| Surface/subsurface concentrations | `pid`, `pollutant` | `huc_id`, `pollutant` | `pollutant` |
| Curve number/infiltration fraction | `land_cover`, `hsg`, optional `pid` | `huc_id`, `land_cover`, `hsg` | `land_cover`, `hsg` |
| Delivery ratios | `pid`, `oid` | `huc_id`, `oid` | `oid` |

For HUC/domain RUSLE factors and concentrations, an optional `land_cover` key selects rows matching each parcel's effective classification. HUC/domain delivery rows with NULL `oid` apply to the assigned parcels' connected outlets. Parcel delivery defaults still require both `pid` and `oid` to be NULL. Lookup rows without a parcel `pid` act as parcel defaults below HUC and domain lookup rows; an optional exact `pid` supplies a local override.

Examples:

```text
# hucs.gpkg:input_pollutant_load_rate
huc_id   | pollutant | mean | sd  | units
05090202 | TN        | 8.0  | 1.5 | kg/ha/year

# domain.gpkg:input_pollutant_load_rate
pollutant | value | units
TN        | 6.0   | kg/ha/year

# domain.gpkg:input_rain_days
value | units
100   | days/year
```

Inherited distributions share one draw per source row per scenario, with separate groups for different variables/natural keys. An explicit `sample_group` can request broader sharing for the same variable. Parcel-default distributions remain independent unless explicitly grouped. Classifications and selection/delivery weights remain fixed-only.

Missing rows fall through to the next scope; malformed rows fail even if overridden. Entire rows override, so fixed values and distribution statistics are never blended. Required coverage is validated after inheritance; a required table need not physically exist in the parcel package if HUC/domain inputs supply it. Unknown `input_*` tables are rejected. Do not mix aggregate and pathway-specific statistical load-rate rows across scopes.

### Parcel-to-HUC assignment

The optional `parcel_huc` table in the parcel GeoPackage stores:

```text
pid | huc_id   | area_fraction
1   | 05090202 | 0.95
```

If supplied, it must assign every modeled parcel exactly once and reference known parcel/HUC IDs. If absent, the loader computes assignments from the modeled (domain-clipped) parcels without changing the input files. The greatest positive area overlap wins; equal overlaps are broken by ascending HUC ID. Values are not blended across boundaries. A parcel intersecting no HUC is an error.

`area_fraction` is optional for stored assignments, must be in [0, 1], and records the fraction of parcel area in its assigned HUC. Values below 0.75 are logged to `log.txt` without a console warning. Regenerate stored assignments after changing parcel or HUC geometry.

To persist assignments using local HUC polygons at any supported level:

```text
python utils/create_parcel_huc.py parcels.gpkg hucs.gpkg
```

The utility supports `--huc-layer`, `--huc-field`, and `--parcel-layer`; it updates only the registered `parcel_huc` table. It uses the source parcel geometry, whereas automatic runtime assignment uses the clipped modeled geometry.

### Existing HUC12 packages and migration

Existing spatial `huc12` layers, `huc12` input keys, and `parcel_huc12` relationships remain readable. The `plet_forcing` YAML key remains a deprecated alias; do not supply it together with `huc_parameters`. Prefer `parcel_huc` if both relationship tables exist. East Fork inputs now live in `inputs/hucs`; their legacy internal table names and data are unchanged.

The existing HUC12 download/workbook utilities remain HUC12-specific. See [PLET/RUSLE preparation](plet_rusle_mode.md#build-or-refresh-the-huc12-forcing-input) for their options. Use the generic local assignment utility for other levels. Do not run the HUC12-specific refresh utilities against generic HUC packages.

### PLET parcel variables

Each of these tables is keyed by `pid`:

- `input_avg_rain_in` (PLET `AVG_RAIN`, inches/event; optional direct event-rainfall forcing)
- `input_annual_precip_in` (legacy/derived forcing)
- `input_rain_days`
- `input_rain_correction_fraction` (required only when `input_avg_rain_in` is not supplied)
- `input_runoff_day_fraction`
- `input_land_cover`
- `input_hsg`
- `input_ia_ratio`

`input_land_cover` and `input_hsg` are categorical and therefore require fixed `value` entries; distributions are not allowed.

PLET precipitation forcing may be supplied in either of two forms:

1. **Direct PLET form:** `input_avg_rain_in` + `input_rain_days` + `input_runoff_day_fraction`. `avg_rain_in` is the PLET Input Data Server `AVG_RAIN` value and is used directly as event rainfall `P`.
2. **Derived form:** `input_annual_precip_in` + `input_rain_correction_fraction` + `input_rain_days` + `input_runoff_day_fraction`. The model derives `P = annual_precip_in * rain_correction_fraction / (rain_days * runoff_day_fraction)`.

If `input_avg_rain_in` is present for a parcel, it is authoritative. Annual runoff is still multiplied by the corrected number of runoff-producing rain days, `rain_days * runoff_day_fraction`. For subsurface infiltration, corrected annual precipitation is reconstructed as `avg_rain_in * rain_days * runoff_day_fraction`, which is algebraically equivalent to `annual_precip_in * rain_correction_fraction` when `AVG_RAIN` comes from PLET Equation 3.


### Curve number and infiltration fraction

PLET hydrology assumptions are user inputs stored in the same parcel GeoPackage:

- `input_curve_number`
- `input_infiltration_fraction`

These two tables are keyed by:

```text
land_cover | hsg
```

and then use the standard numeric/distribution columns. Every land-cover × HSG pair actually used by modeled parcels must have both a curve-number definition and an infiltration-fraction definition. Reusable tables may contain additional valid pairings. Either variable may be fixed or stochastic.

There is no separate `hydrology_lookup` file or source-code fallback.

### RUSLE and sediment variables

Each table is keyed by `pid`:

- `input_rusle_r`
- `input_rusle_k`
- `input_rusle_ls`
- `input_rusle_c`
- `input_rusle_p`
- `input_sediment_delivery_ratio`
- `input_sediment_n_pct`
- `input_sediment_p_pct`
- `input_enrichment_ratio`

The loader assembles these tables into the existing runtime RUSLE parameter representation before scenario execution.

### Statistical load rate

`input_pollutant_load_rate` is required in statistical mode.

Keys:

- `pid`
- `pollutant`
- optional `pathway`

The table uses the common numeric/distribution schema. `pid IS NULL` defaults and parcel-specific integer-PID overrides are supported.

### PLET pollutant concentrations

PLET/RUSLE uses two separate input variables:

- `input_surface_concentration`
- `input_subsurface_concentration`

Both are keyed by:

```text
pid | pollutant
```

and use the common numeric/distribution schema. The pathway is implied by the table name rather than stored in a `pathway` column.

### Delivery-ratio variables

The parcel-to-outlet relationship remains in `parcel_outlets`. Each optional delivery variable is stored separately:

- `input_sdr_f_to_s`
- `input_sdr_s_to_o`
- `input_ndr_f_to_s`
- `input_ndr_s_to_o`

Keys:

```text
pid | oid
```

These are deterministic fractions in `[0, 1]`. An exact parcel/outlet row overrides an optional global row where both `pid` and `oid` are SQL `NULL`. Partial defaults (only one key NULL) are invalid. If a table is absent, the neutral value `1.0` is used.

## `outlets.gpkg`

Configured with:

```yaml
outlets: ../outlets/outlets.gpkg
```

`outlets` is the required spatial layer with unique `oid` and point geometry.

`outlet_stats` is an optional attribute table keyed by `oid × pollutant` with optional nonnegative `target` and `mean` columns.

## Other CSV inputs


`bmp_efficiency` defines BMP efficiency values/distributions by CPS, pollutant, and pathway where applicable.

`bmp_cost` is optional and defines BMP cost inputs. Each row represents one
additive cost component for a CPS code. Multiple rows may therefore share the
same `cps`; their independently sampled/scaled component costs are summed for
each individual BMP placement. Use `cost_component` to give each component a
unique descriptive name. Supported units include `USD/ha`, `USD/m`, and
`USD/project`, so components with different scaling can be combined for one
BMP. Legacy files without `cost_component` remain valid and are interpreted as
a single component named `total`.

The optional `citations` and `notes` columns are metadata only and do not
affect sampling. For example:

```csv
cps,cps_name,cost_component,unit,value,mean,sd,min,p05,p50,p95,max,citations,notes
340,Cover Crop,seed,USD/ha,,,,40,45,55,70,80,https://example.org/source,Example only
340,Cover Crop,establishment,USD/ha,,,,20,30,50,80,100,https://example.org/source,Example only
```

## Canonical outputs

Below the configured `outputs` directory, the model writes:

- `bmps/s{scenario}.parquet`
- `parcels/s{scenario}.parquet`
- `load_parameters/s{scenario}.parquet` for PLET/RUSLE diagnostics
- `scenario_metrics/s{scenario}.parquet`
- `outlet_trajectories/all_scenarios.parquet`
- logs and summary plots

### Distinction between BMP removal and outlet trajectories

`bmps/s{scenario}.parquet` includes the signed, pre-delivery
`removed_mass_{pollutant}_kg` for each placement, summing actual changes in
load from **all** treated parcels. The annual unit is kg per one-year model
timestep, not kg/ha/yr; the pre-/post- parcel load-rate columns in
`parcels/s{scenario}.parquet` use kg/ha/yr.

In `outlet_trajectories/all_scenarios.parquet`, `total` means the **cumulative
signed change in delivered outlet mass** (kg per annual timestep), not the
cumulative unadjusted BMP mass removed. Each affected parcel's change is
multiplied by its own two-factor TSS or nutrient delivery coefficients, and is
credited only to its mapped outlets. For `target` and `mean` trajectory types,
the cumulative delivered reduction is divided by the corresponding configured
positive outlet reference load and multiplied by 100 (%). A negative
trajectory value represents an increase in delivered loading.

The accounting is a parcel-level delivery convention and does not explicitly
transport loads into and back out of a wetland. It is common to statistical
and PLET/RUSLE load generation and handles serial BMP stacking.
