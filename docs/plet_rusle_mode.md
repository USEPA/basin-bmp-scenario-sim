# PLET/RUSLE load-generation mode

[← Back to main README](../readme.md)

`plet_rusle` mode derives annual surface and subsurface loads from user-supplied parcel parameters, hydrology assumptions, pollutant concentrations, and optional RUSLE sediment parameters.

## Configuration

```yaml
domain: ../domain/domain.gpkg
parcels: ../parcels/parcels.gpkg
# Optional HUC-scale parameters:
huc_parameters: ../hucs/plet_inputs.gpkg
outlets: ../outlets/outlets.gpkg

pollutants: [TN, TP, TSS]
cps: [340, 329, 590, 412, 656]

bmp_efficiency: ../bmps/bmp_efficiency.csv

load_generation:
  mode: plet_rusle
```

PLET/RUSLE inputs and overrides live in dedicated `input_*` tables in the parcel, HUC, or domain GeoPackage, including curve number and infiltration fraction. There is no separate hydrology-lookup path. See the [general scope schemas](input_output_reference.md#huc-and-domain-parameter-geopackages) for variable-level HUC inputs and domain defaults. The following section describes the legacy HUC12 layout retained by the PLET workbook/download utilities.

## HUC12 climate forcing

Existing HUC12 packages may use a non-spatial `parcel_huc12` relationship table in `parcels.gpkg`:

```text
pid | huc12 | area_fraction
```

`huc12` is stored as 12-character text. The preparation utility writes `area_fraction`, the fraction of parcel area inside its assigned (dominant) HUC12; when present, the loader checks that it lies between 0 and 1 and records values below 0.75 in the model's `log.txt` without printing the warning to the console. Every modeled parcel must have exactly one assignment. Forcing uses the dominant HUC12 only; it does not blend values across HUC12 boundaries.

The forcing GeoPackage separates spatial structure from model variables. The spatial `huc12` layer contains only WBD geometry/metadata:

```text
huc12 | name | states | areaacres | areasqkm | geometry
```

HUC12-scale PLET variables are stored in dedicated attribute tables in the same GeoPackage:

- `input_avg_rain_in`
- `input_annual_precip_in`
- `input_rain_days`
- `input_rain_correction_fraction`
- `input_runoff_day_fraction`
- `input_hsg` (fixed `A`, `B`, `C`, or `D`; optional alternative to parcel HSG)
- `input_rusle_r`, `input_rusle_k`, `input_rusle_ls`, `input_rusle_c`, `input_rusle_p` (optional, keyed by `huc12 × land_cover`)
- `input_sediment_n_pct`, `input_sediment_p_pct` (optional HUC12 sediment nutrient percentages)
- `input_surface_concentration`, `input_subsurface_concentration` (optional, keyed by `huc12 × land_cover × pollutant`)

Each table is keyed by `huc12` and uses the same numeric uncertainty schema as parcel inputs:

```text
huc12 | value | mean | sd | min | p05 | p10 | p25 | p50 | p75 | p90 | p95 | max | sample_group | units | notes
```

A numeric row may therefore be a fixed value or a distribution; `input_hsg` must use a fixed classification. By default, a distribution is sampled **once per HUC12 per scenario** and that draw is shared by all parcels assigned to the HUC12 (and for land-cover-indexed inputs, by parcels with that land cover). An explicit `sample_group` can intentionally share a draw across multiple rows for the same parameter.

Individual HUC12 variables, or all rows for a HUC, may be omitted when parcel-specific, domain, or `pid=NULL` parcel defaults supply them. Required land-cover, HSG, hydrology, climate, and load coverage is checked after inheritance; those tables no longer have to reside in the parcel package.

For HUC12-supplied variables, precedence is:

```text
parcel-specific input_* override > HUC12 input_* row > domain input_* row > pid=NULL parcel default
```

For land-cover-indexed HUC12 RUSLE and concentrations, only rows matching the parcel's effective PLET `land_cover` apply. Concentration rows also match pollutant and surface/subsurface pathway. Thus parcel tables remain useful for defaults and local overrides without duplicating HUC12 values across every parcel. If `runoff_day_fraction` is absent from the HUC12 tables, it must resolve from parcel or domain inputs because PLET annualizes event runoff using corrected rain days.

### Build or refresh the HUC12 forcing input

Both PLET workbook utilities explicitly use pandas' `calamine` Excel reader
(`pandas>=2.2` and `python-calamine` in the repository dependencies). This avoids
the native openpyxl/lxml XML parsing path, which can cause Windows access
violations when used alongside the GDAL-based geospatial libraries. No
`OPENPYXL_LXML` environment setting is required. Update an existing environment
with `python -m pip install -r requirements.txt` before using these utilities.
Excel fixtures in the tests use the pure-Python `xlsxwriter` writer rather than
loading openpyxl/lxml. An existing openpyxl installation need not be removed.

`utils/download_wbd_huc12.py` reads the parcel extent, downloads candidate 12-digit hydrologic-unit polygons from the USGS WBD service, filters them using a parcel-overlap threshold, writes/refreshes the spatial `huc12` layer, preserves the HUC12 `input_*` tables, and populates `parcels.gpkg:parcel_huc12` using greatest parcel-area overlap.

For the East Fork example:

```bash
python utils/download_wbd_huc12.py \
  examples/east_fork/inputs/parcels/parcels.gpkg \
  examples/east_fork/inputs/hucs/huc12s_plet_deterministic.gpkg \
  --initial-plet-export examples/east_fork/inputs/misc/EastFork_plet_inputs.xlsx
```

`--initial-plet-export` reads the `1. Watershed Land Use` sheet and can initialize `AVG_RAIN`, `RAIN_DAYS`, and `ANNUAL_RAINFALL` as fixed rows in `input_avg_rain_in`, `input_rain_days`, and `input_annual_precip_in`. It cannot be used again once any HUC12 input table contains values; edit those tables directly in the GeoPackage. Re-running the utility refreshes WBD geometry/metadata and preserves input rows for HUC12s still retained by the refreshed layer; rows for HUC12s no longer retained are dropped, so back up edited inputs before refreshing. The default `--area-fraction-threshold 0.75` keeps a HUC12 only if at least one parcel has 75% or more of its area inside that HUC12. A refresh fails if the retained HUC12s do not cover every parcel; lower the threshold or inspect the geometries. The utility reports dominant assignments below the chosen threshold, while the model loader warns below its fixed 0.75 QA threshold.

`utils/create_parcel_huc12.py` remains available for legacy HUC12 assignments. Use `utils/create_parcel_huc.py` for generic local HUC polygons at other levels; if no mapping is stored, the model computes assignments in memory.

### Generate both PLET model inputs from a workbook

To create **new** GeoPackages without modifying the source parcel GeoPackage:

```bash
python utils/create_plet_model_inputs.py \
  examples/east_fork/inputs/parcels/parcels.gpkg \
  parcels \
  examples/east_fork/inputs/misc/EastFork_plet_inputs.xlsx
```

This creates `parcels_plet.gpkg` (a copy preserving existing tables and parcel overrides) and `parcels_plet_huc12.gpkg` alongside the source. It refuses to overwrite either output. Set `parcels:` to the first output and `huc_parameters:` to the second in the model configuration. The parcel layer name is the second positional argument. If that layer is not named `parcels`, the utility uses GDAL (`ogrinfo`) to rename **only the output copy's** spatial layer to the model's canonical `parcels`, preserving feature IDs, GeoPackage metadata, spatial indexes and relationships. The selected layer must use schema-v3 `pid INTEGER PRIMARY KEY`; a different existing `parcels` table is a hard error. Add `--sssurgo_hsg_true` to download SSURGO polygons and assign each parcel its dominant HSG; the HUC12 package then retains an empty registered `input_hsg` table and does not import workbook SHG. Otherwise the workbook's `SHG` column provides HUC12 HSG and the copied parcel's old `input_hsg` table is removed. Existing `input_curve_number` and `input_infiltration_fraction` rows take precedence over workbook values for the same `land_cover × hsg`, including existing uncertainty distributions.

The utility downloads USGS WBD HUC12 polygons and queries the public [annual NLCD ImageServer](https://di-nlcd.img.arcgis.com/arcgis/rest/services/USA_NLCD_Annual_LandCover/ImageServer) for its latest available `Year` (2024 at the time of writing), exporting that year's 30 m land-cover raster. It does **not** silently fall back to 2021. The source URL and year are logged and stored in each classification's notes. It classifies parcels by actual area intersected by each raster cell, not by pixel-center counts: parcels smaller than one NLCD cell are supported. No valid intersected raster area is an error; partial nodata coverage is warned and percentages use valid covered area. The CLI writes an audit log next to the input, `<parcel-stem>_plet.log`; `--log-path PATH` selects another location, and `--verbose` includes each parcel's area-weighted NLCD code/category percentages, intersected-pixel count, valid-coverage percentage, dominant code and resulting PLET class. Summary class percentages and the rainfall audit are always logged. NLCD developed, forest, grass/pasture and cultivated-crop classes map to PLET `urban`, `forest`, `pastureland`, and `cropland`. **By user-specified assumptions, dominant NLCD barren class 31 maps to PLET `pastureland` and woody wetlands class 90 maps to PLET `forest`; neither is an NLCD equivalence.** The CLI warns on the console and in the audit log with the affected parcel counts and example IDs, and marks each affected `input_land_cover.notes` row `ASSUMPTION` for review. Water, other wetlands, shrub and other unmapped dominant classes still cause an explicit error rather than an invented PLET classification. Review classifications before modeling. Exported watershed land-use acreage is *not* used to classify parcels.

The workbook supplies HUC climate, HSG, land-cover-specific RUSLE values, TN/TP surface concentration (with explicit zero subsurface TN/TP), and parcel `land_cover × hsg` curve-number and infiltration lookups. Sheet `4. Nutrient and E.Coli Content` supplies `SOIL_N_CONC` and `SOIL_P_CONC` for HUC12 `input_sediment_n_pct` and `input_sediment_p_pct`, respectively. Values such as `0.08` and `0.0308` are stored **as percent** (not multiplied by 100); the model divides them by 100 when computing sediment-bound nutrient mass. The other workbook sheets for animals, BOD, E. coli, and BMPs are outside this TN/TP/TSS input utility's scope. **TSS needs review:** the workbook has neither TSS concentration nor urban RUSLE factors. The utility writes a *zero placeholder* surface TSS concentration for every retained HUC12 and each of the five PLET land covers, marks those rows `PLACEHOLDER`, and warns on the console and in the audit log. Replace these values or supply parcel-specific TSS concentration overrides before relying on TSS results. Where complete RUSLE inputs resolve (including existing parcel defaults), modeled TSS comes from RUSLE rather than this concentration; urban RUSLE values are **not** synthesized from the workbook. The existing East Fork `parcels_plet_deterministic.gpkg` has default RUSLE factors, nonzero user-defined curve numbers, and TN/TP surface/subsurface concentrations, so these particular missing workbook fields need not block that configuration; this does not validate its scientific TSS assumptions or guarantee the entire model run.

Existing unrelated parcel inputs and overrides are preserved. Zero-valued PLET `User Defined` curve numbers are undefined and omitted with a warning: provide positive numbers before using that class. The workbook has no raw `Rcor`/`RDcor` columns. The utility searches 0.001-spaced `Rcor` values from 0.840 to 0.900 and `RDcor` values from 0.350 to 0.500 using `RDcor / Rcor = ANNUAL_RAINFALL / (AVG_RAIN * RAIN_DAYS)` and requires a near-exact match. Where multiple grid solutions exist, it weights up to four geographically nearest *uniquely reconstructed* HUC12 polygon centroids by inverse distance and selects the matching pair nearest those neighbors' factor estimate. East Fork's ambiguous `050902021005` and `050902021101` consequently select `.870/.435` and `.874/.437`, respectively. These are reconstructed estimates, **not independently verified factors**; neither existing HUC-package factor values nor its notes are used as source inputs. Logs audit exported versus predicted rain, relative error and ambiguity status, and all reconstructed HUC rows are marked in table notes.

## Parcel PLET tables

Each variable gets its own table keyed by `pid`:

- `input_avg_rain_in` (PLET `AVG_RAIN`, inches/event; optional direct event-rainfall forcing)
- `input_annual_precip_in` (legacy/derived forcing)
- `input_rain_days`
- `input_rain_correction_fraction` (required only when `input_avg_rain_in` is not supplied)
- `input_runoff_day_fraction`
- `input_land_cover`
- `input_hsg`
- `input_ia_ratio`

Numeric tables use the standard fixed-value/distribution schema. `input_land_cover` and `input_hsg` require fixed categorical values. Where defaults are supported, `pid IS NULL` supplies the default and an exact integer `pid` row overrides it.

PLET precipitation forcing may be supplied in either of two forms:

1. **Direct PLET form:** `input_avg_rain_in` + `input_rain_days` + `input_runoff_day_fraction`. `avg_rain_in` is the PLET Input Data Server `AVG_RAIN` value and is used directly as event rainfall `P`.
2. **Derived form:** `input_annual_precip_in` + `input_rain_correction_fraction` + `input_rain_days` + `input_runoff_day_fraction`. The model derives `P = annual_precip_in * rain_correction_fraction / (rain_days * runoff_day_fraction)`.

If `input_avg_rain_in` is present for a parcel, it is authoritative. Annual runoff is still multiplied by the corrected number of runoff-producing rain days, `rain_days * runoff_day_fraction`. For subsurface infiltration, corrected annual precipitation is reconstructed as `avg_rain_in * rain_days * runoff_day_fraction`, which is algebraically equivalent to `annual_precip_in * rain_correction_fraction` when `AVG_RAIN` comes from PLET Equation 3.


## User-controlled hydrology

Curve number and infiltration fraction are also separate user-input tables in any supported scope:

```text
input_curve_number
input_infiltration_fraction
```

Both are keyed by `land_cover × hsg` and use the same numeric/distribution schema as other numeric inputs. Every supported pair must be covered by both tables.

These are the values actually used by PLET. They may be fixed or distributions. There is no source-code fallback and no external `hydrology_lookup` configuration key.

## RUSLE/sediment tables

- `input_rusle_r`
- `input_rusle_k`
- `input_rusle_ls`
- `input_rusle_c`
- `input_rusle_p`
- `input_sediment_delivery_ratio`
- `input_sediment_n_pct`
- `input_sediment_p_pct`
- `input_enrichment_ratio`

The input layer assembles these dedicated tables into the existing runtime RUSLE parameter table, so the simulation equations and parallel scenario execution remain unchanged.

## Pollutant concentrations

Surface and subsurface concentrations are distinct variables:

```text
input_surface_concentration
input_subsurface_concentration
```

Each table is keyed by `pid × pollutant` and uses the common numeric/distribution schema. A row with `pid IS NULL` supplies the pollutant default; an exact integer `pid` overrides it.

Surface concentrations are required for TN and TP and for TSS when complete RUSLE inputs are unavailable. Subsurface concentrations are required for modeled non-TSS pollutants.

## BMP pathways

PLET/RUSLE production pathways are fixed to `surface` and `subsurface`. Surface BMP efficiencies are required. Missing correctly labeled subsurface efficiencies are completed with zero and logged.

## Optional mass-based manure loading

An optional APLE-manure-P/APLE-inspired-N model accepts animal inventories,
allocates manure TN and TP by field, and converts predicted surface mass
exports into annual runoff concentration increments. It uses existing
PLET surface hydrology and leaves groundwater concentrations unchanged.
See [Mass-based AFO loading](afo_mass_based.md).
