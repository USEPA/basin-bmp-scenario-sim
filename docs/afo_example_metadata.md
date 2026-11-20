# AFO example GeoPackages: spatial points and in-row notes

The three East Fork AFO GeoPackages contain **synthetic examples**, not a
survey of livestock operations or calibrated nutrient coefficients. The
`afo_operations` points are visual examples anchored to parcels, **not actual
facility coordinates**. A method citation explains how a model input is used;
it does **not** establish that the example number is realistic.

| Example package | Application | AFO point layer | Other input tables |
|---|---|---|---|
| `afo_example.gpkg` | Legacy PLET concentration increment | `afo_operations` | `afo_animals`, `afo_manure_fields`, `afo_manure_concentration_increments` |
| `afo_mass_example_generated.gpkg` | Mass-based PLET, Option A | `afo_operations` | `afo_animals`, `afo_manure_fields`, `afo_manure_management` |
| `afo_mass_statistical_example.gpkg` | Mass-based statistical, Option A | `afo_operations` | `afo_animals`, `afo_manure_fields`, `afo_manure_management`, `afo_field_eligibility`, `afo_hydrology` |

## Spatial example geometry

`afo_operations` is a **GeoPackage POINT feature layer in EPSG:4326**. Every
operation has a point at the *geometric centroid* of its host `pid` polygon in
the corresponding configured parcels GeoPackage:

- PLET and legacy examples: `parcels_ex2dot1_plet_based_load_gen_deterministic.gpkg`, spatial layer `parcels`.
- Statistical example: `parcels_ex1dot1_lit_based_load_gen_deterministic.gpkg`, spatial layer `parcels`.

For example, `AFO_DEMO_01` is placed at the centroid of **pid 1**, and
`AFO_DEMO_02` at **pid 3**. The geographic point is **only for locating the
synthetic demonstration visually in QGIS**; the model identifies operations
and computes feedlot and manure loading using `afo_operations.pid`, not the
point coordinates. Editing a `pid` alone will not move the point: update
the geometry to keep the two consistent. Geometric
centroids can lie outside irregular polygons; that is consistent with the
requested centroid rule and is not an error in facility siting.

## Explanatory notes in the corresponding tables

Every `afo_*` input table contains an editable **`notes`** column. Its notes
interpret the values *in that exact row*, give applicable units or percentage
conventions, and include methodological citations. For instance:

- `afo_operations.notes`: host `pid`, feedlot area in **ha**, paved share as
  **percent**, field-count distributions in **number of fields**, allocation
  method, and whether legacy `manure_months` applies. It also explains the
  illustrative host-parcel-centroid point.
- `afo_animals.notes`: animal counts in **head** and TN/TP feedlot equivalence
  factors as **dimensionless ratios** (not manure excretion factors).
- `afo_manure_fields.notes`: interpretation of the receiving-parcel `pid`.
- `afo_manure_concentration_increments.notes`: additive TN/TP concentration
  increments in **mg/L** (legacy only).
- `afo_manure_management.notes`: annual manure nutrient surface export,
  collection, offsite export and handling retention as **fractions in [0, 1]**
  (mass-based examples only). `0.05` means 5% of the specified mass, not 5 kg.
- `afo_field_eligibility.notes`: whether a parcel is eligible for sampled
  receiving-field allocation (statistical example only).
- `afo_hydrology.notes`: **mm/year** for precipitation and direct-feedlot
  runoff, with `pid=NULL` identifying the domain-wide default (statistical
  example only).

There are **no `input_column_guide`, `input_record_notes`, or other custom
explanation tables**. Normal GeoPackage `gpkg_*` infrastructure tables remain
because they are required by the file format. Explanatory columns are ignored
by `src/afo.py`, including when loading the legacy concentration-increment
method, and do not change simulations.

The methods documented in notes are primarily the
[EPA PLET User Guide (2022)](https://www.epa.gov/system/files/documents/2022-04/user-guide-final-04-18-22_508.pdf),
[`animal_feeding_operations.md`](animal_feeding_operations.md), and
[`afo_mass_based.md`](afo_mass_based.md). None supports the *specific synthetic
example values*. Use empirical citations and calibrated inputs before treating
any input as a prediction for a real watershed.

## Inspect in QGIS or SQLite

In QGIS, open `afo_operations` as a point layer; inspect its attribute table
and `notes` column. Open other `afo_*` tables and read their `notes` in the
same table as the values they explain.

Example SQLite queries:

```sql
SELECT afo_id, pid, feedlot_area_ha, paved_pct, notes
FROM afo_operations;

SELECT afo_id, tn_surface_export_fraction, tp_surface_export_fraction, notes
FROM afo_manure_management;
```

## Maintaining the example packages

If a GeoPackage is changed in QGIS or SQLite, update the notes in the same
rows, verify its registration and integrity, and check that host-point
coordinates correspond to the current host `pid` centroids. For the mass-based
examples, run the associated model configuration and inspect the allocation,
operation budget, and source-component outputs. The [example GeoPackage
reference](afo_example_geopackage_reference.md) describes the expected tables
and mode-specific inputs.
