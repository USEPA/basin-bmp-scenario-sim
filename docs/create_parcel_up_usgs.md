# Generate the required `parcel_down` table using WhiteboxTools and USGS 3DEP

`utils/create_parcel_down.py` computes a single *representative downstream*
parcel per source parcel, using a local DEM or downloading a USGS 3DEP DEM.
It preserves the existing local DEM, USGS tile caching and DEM selection paths.
Its GeoPackage output uses **`parcel_down(pid,pid_down)`**, not the former
`parcel_up(pid,pid_up)` schema. The model requires `parcel_down` and offers
**no legacy fallback**.

## Commands

```sh
python utils/create_parcel_down.py parcels.gpkg local_dem.tif
python utils/create_parcel_down.py parcels.gpkg --list-usgs-resolutions
python utils/create_parcel_down.py parcels.gpkg --usgs-resolution 30m
python utils/create_parcel_down.py parcels.gpkg local_dem.tif --no-gpkg-update --out parcel_down.csv
```

When no DEM is supplied, the script queries official USGS 3DEP index and
TNMAccess resources, presents resolutions, downloads tiles, projects and
mosaics the DEM, verifies raster validity and caches available metadata.

## Representative, acyclic downstream graph

WhiteboxTools conditions the DEM (`BreachDepressionsLeastCost` by default) and
computes D8 cell pointers. Each rasterized parcel cell is traced to the first
*different* parcel encountered downstream, or to the edge of the modeled
parcel dataset. Count only cells **originating** in each source parcel, not
upstream throughflow. The largest share wins, even if its share is less than
50%; exact ties use the **lower numerical PID**. If the largest destination is
outside the parcel dataset, the source becomes an outlet and gets no row.
Connections that would introduce parcel-level cycles are rejected; the
next-highest supported connection is considered. If no noncyclic connection
is available, the parcel has no downstream row. This whole-parcel approximation
is explicitly **not** fractional downstream load routing.

The selected relationship is written as `pid` (origin) and `pid_down`
(immediate recipient); the model reverses the table in memory for upstream
wetland traversal. No transitive upstream ancestry is stored in the GPKG.
The `parcel_down` table has an integer primary key `id`, and `pid` is
`UNIQUE`, so a source cannot have two downstream recipients. Valid downstream
outlets have **no row**. The table is rewritten transactionally while
preserving all other input tables. A successful rewrite also removes the
obsolete `parcel_up` table and its GeoPackage registration.

**Acyclicity is mandatory. Completeness is not.** The script writes `parcel_down`
automatically even when some parcels are unrasterized, outlet-like or
cycle-unresolved. It records all candidate drainage shares in the diagnostics
CSV (`<parcels_stem>_parcel_down_candidates.csv`). You do not need a separate
acceptance flag.

## Geometry and DEM safety checks

Before running WhiteboxTools, the utility rejects significant overlapping
parcel polygons (default threshold: `--max-overlap-m2 0.01`) and verifies
full parcel geometry coverage by a valid-data DEM. A local DEM that covers
only a corner of a parcel is invalid. Full coverage is different from cell
resolution: a fully covered parcel may still be too small to rasterize at
30 m, and this is documented as unresolved in the candidate diagnostics.

`--layer`, `--id-field`, `--dem-buffer-m`, `--max-dem-cells`,
`--usgs-cache-dir`, and `--refresh-usgs-availability` remain available.
A fully checked source does not guarantee the representative whole-parcel
connection is the only real hydrologic route.

## Provenance

A JSON manifest (`<parcels_stem>_parcel_down_manifest.json` or
`--manifest-out`) records sorted parcel geometry fingerprint, DEM and D8
raster hashes, Whitebox settings, source/DEM metadata, selected graph count,
connection hash, and unresolved routing metrics. This allows a user to
recognize drainage outputs generated from an outdated parcel geometry layer.

## Strict migration of older GeoPackages

To update existing GeoPackages, do **not** just rename `parcel_up`: its
original orientation is inverted, and older example graphs may have cycles.
The one-time migration tool accepts a prevalidated graph as CSV:

```sh
python utils/migrate_parcel_down.py parcels.gpkg --reference-csv parcel_down.csv --dry-run
python utils/migrate_parcel_down.py parcels.gpkg --reference-csv parcel_down.csv
```

For the East Fork reference inputs, use the previously audited Whitebox
connectivity CSV bundled in `utils/data`:

```sh
python utils/migrate_parcel_down.py --east-fork --dry-run
python utils/migrate_parcel_down.py --east-fork
```

The migration creates `.before_parcel_down.bak` backups, replaces the
`parcel_down` table with a normalized acyclic graph, and deletes the obsolete
`parcel_up` table. It does **not** alter parcel geometries or scenario-specific
input variables. Separate geometry deduplication is a distinct data-cleaning
step and is not performed automatically by this migration.

**Local execution note:** WhiteboxTools requires the `whitebox` Python package
(`python -m pip install whitebox`) or `--whitebox-exe PATH`. Validate a local
DEM run with your installed Whitebox executable before using a new graph for
scientific production.
