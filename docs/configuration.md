# Configuration reference

[← Back to main README](../readme.md)

## Configuration structure

A YAML configuration points to a small set of physical input packages and selects one of two load-generation modes:

- `statistical` (default)
- `plet_rusle`

Parcel-related inputs are consolidated in one GeoPackage and outlet-related inputs in a second GeoPackage. Within `parcels.gpkg`, every modeled user input variable has its own `input_*` table.

## Common configuration

```yaml
domain: ../domain/domain.gpkg
parcels: ../parcels/parcels.gpkg
outlets: ../outlets/outlets.gpkg

pollutants: [TN, TP, TSS]
cps: [340, 329, 590, 412, 656]

bmp_efficiency: ../bmps/bmp_efficiency.csv
bmp_cost: ../bmps/bmp_cost.csv                         # optional

outputs: ../../outputs/run_1
random_seed: 42
n_scenarios: 100
bmp_limit_n: 50
deterministic_parcel_order: false
deterministic_bmp_type_order: false
deterministic_bmp_design_parameters: false
```

Set `deterministic_parcel_order: true` to place BMPs on eligible parcels in
ascending parcel-ID order, cycling back to the first eligible parcel when
additional BMPs are placed. When false (the default), parcel selection remains
probability-weighted and random.

Set `deterministic_bmp_type_order: true` to select BMP types in ascending
CPS-code order, cycling through the available types. Types already applied to
the selected parcel are skipped, and selection resumes at the next type in
the cycle.

Set `deterministic_bmp_design_parameters: true` to use fixed BMP geometry
instead of the default sampled geometry: wetland area is 20% of the selected
parcel's area with a 4:1 catchment-to-wetland ratio (the remaining 80% of that
parcel), and grassed-waterway length is 25% of parcel perimeter with 25% of
parcel area treated. This option does not make other model inputs or BMP
selection deterministic.

## Path resolution

Relative filesystem paths are resolved relative to the YAML file, not the process working directory. The GeoPackage files themselves can therefore be renamed or moved by changing only the corresponding YAML path.

Internal GeoPackage table names are schema names. Structural tables are `parcels`, `parcel_up`, and `parcel_outlets`; variable tables use the `input_*` prefix.

## Statistical mode

```yaml
load_generation:
  mode: statistical
```

`input_pollutant_load_rate` must be supplied by the parcel, HUC, or domain GeoPackage, with effective coverage for every modeled parcel and pollutant.

## PLET/RUSLE mode

```yaml
load_generation:
  mode: plet_rusle
```

PLET/RUSLE inputs, including curve number and infiltration fraction, use dedicated `input_*` tables. They can be supplied at parcel, HUC, or domain scale.

## HUC and domain parameter inheritance

In **either** load mode, select one optional HUC parameter GeoPackage:

```yaml
huc_parameters: ../hucs/huc12s_plet_deterministic.gpkg
```

The directory is not scanned: the YAML path selects one package at one HUC level. Canonical packages use a `hucs` polygon layer and text `huc_id`; existing `huc12` layers and keys are also supported. The old `plet_forcing` key is accepted with a deprecation warning; supplying both keys is an error.

The existing `domain:` GeoPackage may contain the same `input_*` tables, with the spatial key omitted. Its boundary (which may be multipart) represents one run-wide domain; no additional YAML setting or per-parcel domain mapping is needed.

Precedence is **parcel-specific row > assigned-HUC row > domain row > NULL-pid parcel default**, resolved separately for each variable and its natural keys (for example pollutant and pathway). An override replaces the entire fixed value/distribution, not individual statistic cells. Missing rows fall through; malformed rows fail validation even if overridden. Empty HUC tables, or HUCs without parameter rows, may fall back to domain inputs or parcel defaults. Required coverage is checked after inheritance.

Use a `parcel_huc` relationship table in the parcel package, or let the loader calculate greatest-area-overlap assignments in memory. Every modeled parcel must have one assigned HUC when a HUC package is selected. See the [input reference](input_output_reference.md#huc-and-domain-parameter-geopackages) for schemas, boundary handling, and the assignment utility.

## Standard numeric input schema

Numeric variable tables share the same uncertainty columns:

```text
value, mean, sd, min,
p05, p10, p25, p50, p75, p90, p95, max,
sample_group, units, notes
```


## Optional mass-based AFO inputs

```yaml
animal_operations: ../afo/afo_mass_example_generated.gpkg
manure_loading:
  method: mass_based
  excretion_factors: ../afo/my_excretion_factors.csv # optional
  runoff_adjusted_export: false # default (annual mass * export fraction)
  # When true, also specify reference_runoff_mm: <positive annual mm>
```

Omit `excretion_factors` to use the bundled
`src/data/afo/manure_excretion_reference.csv`. A custom CSV replaces this
reference and its path is relative to the YAML file. Both use long-form
`species, parameter, value/mean/sd/...` rows for TN and TP production
rates in kg per 1,000 kg live animal mass per day, plus a required
`representative_mass_kg` row per species. Mass uses the PDF's fixed value by
default and may instead use a distribution. Standard species names are
Dairy, Beef, Veal, Swine, Sheep, Goat, Horse, Layer, Broiler, Turkey, and Duck.
Wet-manure production rates are not inputs; only nutrient masses are calculated.
Supply `tn_surface_export_fraction` and `tp_surface_export_fraction` (0–1,
fixed or sampled) per AFO in the GeoPackage `afo_manure_management` table.
If `runoff_adjusted_export: true`, both fractions are multiplied by annual
surface runoff divided by `reference_runoff_mm`, with the result capped at 1.
The default requires no receiving-field runoff data for manure export.
APLE `season`, `solids_pct`, `wep_fraction`, `f_n`, `alpha_n` and
`incorporated_fraction` are not supported in the mass-based method.
The option applies only to `mass_based` loading;
animal-row rate overrides retain precedence. See
[mass-based AFO loading](afo_mass_based.md) for all required GeoPackage
tables, distribution semantics, and D384.1 source assumptions.

## Parallel configuration

```yaml
parallel:
  n_jobs: -1
  max_nbytes: "1M"
  temp_folder: "/tmp/bmp-loky"
```

All input paths are resolved and all input tables are loaded before parallel scenario workers are launched.

The AFO examples use mode-specific GeoPackages: legacy
`afo_example.gpkg`, PLET mass-based `afo_mass_example_generated.gpkg`, and
statistical mass-based `afo_mass_statistical_example.gpkg`. See the
[GeoPackage reference](afo_example_geopackage_reference.md) for table/column inventories.
