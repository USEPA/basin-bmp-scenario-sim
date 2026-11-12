# Audit of the example AFO GeoPackages

The East Fork example configurations use **three separate, deliberately minimal
AFO GeoPackages**. They exercise software behavior; their AFO sites,
animal counts, nutrient coefficients, and hydrology are synthetic and **not**
a surveyed watershed inventory or calibrated nutrient export factors.

| Example configuration | GeoPackage | Necessary AFO tables |
|---|---|---|
| `east_fork_afo_demo.yaml` (PLET-style concentration increment) | `afo_example.gpkg` | `afo_operations`, `afo_animals`, `afo_manure_fields`, `afo_manure_concentration_increments` |
| `east_fork_afo_mass_demo.yaml` (PLET, mass-based, Option A) | `afo_mass_example_generated.gpkg` | `afo_operations`, `afo_animals`, `afo_manure_fields`, `afo_manure_management` |
| `east_fork_afo_mass_statistical_demo.yaml` (statistical, mass-based, Option A) | `afo_mass_statistical_example.gpkg` | `afo_operations`, `afo_animals`, `afo_manure_fields`, `afo_manure_management`, `afo_field_eligibility`, `afo_hydrology` |

## Common input columns

- `afo_operations`: `afo_id`, `pid`, `feedlot_area_ha`, `paved_pct`,
  `allocation`, `n_fields_min`, `n_fields_p50`, `n_fields_max`. The last three
  columns define the field-count distribution for the sampled operation; they
  are necessarily null for the explicitly assigned operation. `selection` is
  omitted because all examples use the loader's `nearest` default.
- `afo_animals`: `afo_id`, `species`, `count`, `factor_tn`, `factor_tp`.
  The TN/TP feedlot animal-equivalence factors are required to retain direct
  feedlot TN and TP loading. `factor_bod` is omitted because BOD is **not** an
  example pollutant, and TSS has no such required equivalent-animal factor.
- `afo_manure_fields`: `afo_id`, `pid`. Only explicit receiving fields are
  listed; sampled operations use modeled eligible land instead.

## Mode-dependent input columns

- PLET-style `afo_operations.manure_months` is retained **only** in
  `afo_example.gpkg`, because the PLET method actively
  uses it. The PLET-style `afo_manure_concentration_increments` has
  `land_cover`, `pollutant`, `value` and covers cropland/pastureland for TN/TP.
- The mass-based packages **omit** `manure_months` and the entire PLET-style
  concentration-increment table. Their `afo_manure_management` has `afo_id`,
  `tn_surface_export_fraction`, `tp_surface_export_fraction`,
  `collection_fraction`, `export_fraction`, `n_retention_fraction`, and
  `p_retention_fraction`. All four handling fractions actively affect the
  applied nutrient budget and are intentionally retained.
- **Statistical only**: `afo_field_eligibility` has `pid`, `eligible`; all four
  synthetic parcels are eligible. `afo_hydrology` has `pid`,
  `annual_precip_mm`, `feedlot_runoff_mm`: a single NULL-pid domain-default
  row covers the feedlot hosts. The Option A manure-field export does not
  require `annual_runoff_mm` or HUC12 IDs; neither column is included. Direct
  feedlot loading **does** require annual precipitation and feedlot runoff.

All packages omit the old unrelated `example_marker` layer. Instead,
`afo_operations` is an actual spatial POINT feature layer in EPSG:4326. Its
geometry is the centroid of the host parcel polygon joined on `pid` in the
example's configured parcels GeoPackage. Those points are synthetic mapping
aids, **not surveyed AFO locations**. The remaining AFO tables are nonspatial
attribute layers; every input layer has a `notes` column describing its own
example rows and providing units/citations where possible. There are no
stand-alone `input_column_guide` or `input_record_notes` tables. See
[afo_example_metadata.md](afo_example_metadata.md).

## Opt-in runoff adjustment

The PLET mass-based example can activate Option B by setting
`manure_loading.runoff_adjusted_export: true` and a positive
`reference_runoff_mm` in YAML; field runoff is calculated from PLET inputs.

The statistical example intentionally ships **without** `annual_runoff_mm`.
To enable Option B, add valid annual receiving-field runoff depths (or
supported distributions) to its `afo_hydrology` table and set the same two
YAML options. Otherwise the loader will correctly reject the incomplete
Option B inputs. Do **not** introduce runoff dependency into Option A.

## Regeneration and validation

From the repository root:

```bash
python scripts/create_afo_example_packages.py
pytest -q tests/test_afo_example_metadata.py tests/test_afo_example_gpkgs.py tests/test_afo.py tests/test_afo_mass.py
```

The script uses Python's standard library and Shapely to compute geometric
centroids from the host parcels. It regenerates the notes described in
`afo_example_metadata.md`, normalizes the committed PLET-style `afo_example.gpkg`,
and rebuilds the two mass-based files. It registers `afo_operations` as a
GeoPackage feature layer, other AFO tables as attributes, and checks SQLite
integrity.
The dedicated test checks exact mode-specific input tables and columns,
GeoPackage registration, retained sample values, and successful loading using
the appropriate AFO method. It also verifies that the statistical default
requires no field runoff observations while leaving Option B opt-in.
