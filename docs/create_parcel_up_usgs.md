# USGS 3DEP DEM discovery for `create_parcel_up.py`

`utils/create_parcel_up.py` now **creates or replaces the nonspatial `parcel_up`
table inside the input parcel GeoPackage**. It now uses **WhiteboxTools**
for depression conditioning and D8 flow pointers, then estimates one dominant
representative drainage destination per parcel from the fraction of DEM cells
originating inside that parcel. One `pid,pid_up` edge per selected connection
is written, not the full upstream catchment or comma-separated ID lists. This
is an explicit single-outlet simplification: real parcels can have more than
one downstream drainage pathway. The model computes
the full contributing network by traversing these immediate edges. The local
DEM workflow remains supported. When the second positional argument is omitted, it
queries official USGS services, lists resolution availability for the **entire**
parcel GeoPackage footprint (plus a configurable buffer), lets you choose a
resolution, downloads the required products, mosaics/reprojects them to a metric
GeoTIFF, and uses that raster for D8 flow routing.

## Interactive workflow

```bash
python utils/create_parcel_up.py path/to/parcels.gpkg
```

For quicker discovery, the script checks up to **four resolutions in parallel**,
uses short bounded availability-request timeouts (12 seconds per attempt, up
to two attempts), and saves successful results. With `--usgs-resolution 10m`,
it only queries that requested resolution and marks other resolutions `NOT
CHECKED`; use `--list-usgs-resolutions` for the complete seven-resolution list.
The speedup affects *availability discovery*, not DEM downloading, raster
mosaicking or the WhiteboxTools hydrologic analysis.

The script first queries the USGS 3DEP elevation coverage index for full,
partial, and missing coverage. If a particular index layer returns a service
error (including HTTP 500), it independently checks the USGS TNMAccess
downloadable-product catalog. Such catalog results are marked `CATALOG FULL*`,
`CATALOG PARTIAL*`, or `NO DOWNLOADABLE PRODUCTS*`; they are **provisional**
because rectangular product footprints do not prove that every raster cell
contains elevation values. `UNKNOWN` means neither service could verify that
resolution, **not** that elevation data is unavailable. It shows availability for 1 m standard, 1 m seamless S1M,
~3 m (1/9 arc-second), Alaska 5 m, ~10 m (1/3 arc-second), ~30 m
(1 arc-second), and Alaska ~60 m (2 arc-second), as appropriate for the area.
The script will only offer FULL-coverage resolutions, then display the projected
raster dimensions and a product download estimate. It asks you to confirm before
starting downloads. `s1m` identifies the newer seamless 1-meter product.

```bash
# Query availability only; do not download or modify any DEM or parcel files.
python utils/create_parcel_up.py path/to/parcels.gpkg --list-usgs-resolutions

# Specify a resolution to check *only that resolution* before download.
# This avoids querying all seven resolutions and is fastest on a first run.
python utils/create_parcel_up.py path/to/parcels.gpkg --usgs-resolution 10m

# Force a fresh availability check rather than using previously cached answers.
python utils/create_parcel_up.py path/to/parcels.gpkg --refresh-usgs-availability

# Also export a normalized CSV of the exact edges written to parcel_up.
python utils/create_parcel_up.py path/to/parcels.gpkg --out parcel_up.csv \
    --usgs-cache-dir path/to/dem_cache --dem-buffer-m 200

# Unchanged local-DEM path: performs no USGS requests.
python utils/create_parcel_up.py path/to/parcels.gpkg path/to/dem.tif
```

The download directory defaults to `path/to/parcels_usgs_dem/`. Cached source
products remain for reuse and the projected output is named
`usgs_3dep_<resolution>_study_dem.tif`. A matching
`usgs_3dep_<resolution>_sources.json` manifest records the dataset, tile URLs,
dates, analysis CRS and processing buffer. A small
`usgs_availability_cache.json` file also records successful resolution-coverage
queries for the buffered study area. It is reused on subsequent runs, avoiding
repeated slow USGS calls. Index-verified results expire after **24 hours**;
provisional TNMAccess catalog results after **3 hours**. Unknown results are
never cached; changes to the parcel footprint or buffer invalidate the cache.
`--refresh-usgs-availability` bypasses it. In `--list-usgs-resolutions` mode,
only this small metadata cache is written, **not** a DEM or product download.
The GeoPackage is only read, not modified, **in this listing-only mode**.
A full run, whether using a local or downloaded DEM, updates the `parcel_up`
attribute table in the input GeoPackage unless `--no-gpkg-update` is set.

GeoPackages with multiple layers automatically use the `parcels` geometry
layer when present, rather than relying on the file's default layer. Use
`--layer NAME` to choose another geometry layer. If `pid` is stored as the
GeoPackage feature-ID primary key (rather than a normal attribute), the
script recovers its **actual** feature-ID values and uses them in the output.
It does not replace them with zero-based row numbers.

## Coverage and resource safeguards

- Availability is preferentially assessed against **USGS index polygons**, not
  an arbitrary USGS elevation value at the watershed's center. If the index
  service fails, the script falls back to TNMAccess **downloadable-product tile
  bounding boxes** for the failing resolution rather than aborting discovery.
  The buffered union of all parcel geometries must have essentially complete
  verified or provisionally cataloged coverage before a resolution may be selected.
  Catalog coverage is explicitly labeled provisional until raster inspection.
- Product selection queries the official TNMAccess API for the user-selected
  resolution and finds a collection of tile bounding boxes covering the full
  study area. Downloads may be GeoTIFF, IMG, or ZIP containing raster data.
- After mosaic/reprojection, it checks the DEM raster's real pixel coverage
  and raises an error if any buffered study area cells are NoData. An index
  polygon or product footprint alone does **not** prove raster coverage.
- The generated raster uses a locally estimated UTM CRS and nominally
  1/3/5/10/30/60-meter cells, as appropriate for the chosen product. This is a
  **reprojected analysis grid**, not a claim that the raw 3DEP DEM is natively
  projected in meters or has exact metric cell spacing.
- It adds a 100 m buffer by default to reduce DEM edge artifacts. Adjust
  `--dem-buffer-m` as needed; this is only a processing margin, and does not
  delineate catchments outside the provided parcel network.
- High-resolution DEMs can be extremely large. By default the script refuses
  requested rasters over 20 million cells **before downloading**. The limit can
  be overridden with `--max-dem-cells N` if enough memory and compute time are
  available. The existing Python D8 routine processes every raster cell and
  is not optimized for very large 1-meter watersheds.
- Temporary USGS HTTP errors (429, 500, 502, 503, 504) receive bounded retries.
  If both USGS services fail for a resolution, the script prints `UNKNOWN` and
  skips that choice; it does **not** equate HTTP 500 with missing coverage.
  Incomplete product coverage, invalid archives, and incomplete raster coverage
  still cause errors instead of silently falling back to a lower-resolution DEM.
  A user-supplied local DEM never contacts either USGS service.

All online requests use public endpoints and require internet access. For
long-term reproducibility, archive the downloaded projected DEM and note the
selected resolution and product filenames. The script does not use elevation
web-services when a local raster is supplied.

**Official sources:** [3DEP coverage index](https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/),
[S1M coverage index](https://index.nationalmap.gov/arcgis/rest/services/USGS_Seamless1m_Index/MapServer),
[TNMAccess API](https://tnmaccess.nationalmap.gov/api/v1/docs),
[3DEP products](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services).

## Normalized `parcel_up` table — important output change

## WhiteboxTools drainage analysis and parcel selection

The old Python steepest-descent routine used an uphill fallback in pits and
flats, which can create cycles. **That algorithm has been replaced** by the
WhiteboxTools `BreachDepressionsLeastCost` (default, with filling of remaining
sinks) or optional `FillDepressions`, followed by `D8Pointer` in Whitebox's
**native non-ESRI pointer encoding**. The script validates the raster for
cell-level cycles before interpreting parcel drainage. WhiteboxTools is a new
optional runtime dependency of this utility:

```bash
# From the environment where you run create_parcel_up.py:
python -m pip install whitebox
# Or: python -m pip install '.[hydrology]' from the project root
```

Alternatively, provide `--whitebox-exe "C:\path\to\whitebox_tools.exe"`
if you already have the WhiteboxTools executable installed. You do **not**
need GRASS GIS or `richdem`. The previous `--use-filled-dem` flag remains as a
deprecated alias for WhiteboxTools' fill-conditioning mode.

The default conditioning uses least-cost depression breaching with up to 100
cells of search distance and fills remaining depressions to prevent unresolved
interior sinks. Adjust with `--whitebox-breach-dist N` or explicitly choose
`--whitebox-conditioning fill`. These settings change routing and should be
reported in scientific analyses. Intermediate files are retained in a
`<DEM_stem>_whitebox_flow` directory beside the DEM (or in
`--whitebox-output-dir DIR`), allowing the conditioned DEM and D8 pointers to
be examined in QGIS. They are recomputed for each full run rather than
silently reusing possibly stale output.

**Drainage evidence:** Each raster cell within a parcel is followed through the
Whitebox single-flow D8 network until it first reaches a *different* parcel or
exits the modeled parcel domain. Flow through unlabeled raster gaps is traced,
and reentering the same parcel does not count as an exit. Each cell contributes
exactly one vote based on its own parcel of origin; upstream flow that only
passes through a parcel does **not** inflate its drainage fraction.

For every source parcel, the script measures the fraction of its own rasterized
area reaching each candidate downstream parcel, or leaving the parcel dataset.
It retains a uniquely dominant connection only when the leading destination
has a fraction of at least **0.50** (configurable by
`--min-dominant-share FRACTION`) and no tie for first place. If most cells
leave the modeled parcel domain, that parcel has **no** `parcel_up` outgoing
edge (it is an outlet). Parcels with insufficient dominant evidence, ties, or
no raster cells at the DEM resolution remain unresolved, **not arbitrarily
assigned** a downstream parcel.

Candidate edges are processed in descending order of their parcel-originating
area share, with deterministic numeric PID tie breaks. If accepting an edge
would create a parcel-level cycle, it is excluded, even though the raster flow
may be acyclic. This ensures the `parcel_up` table is a directed acyclic graph
with at most one downstream parcel per source. Multiple parcels may still
contribute to the same downstream parcel. Cycle rejection is recorded, not
hidden. Because the model's `parcel_up` schema cannot represent distributed
outlets and split-catchment fractions, such cases are an approximation.

**Diagnostics:** By default the script writes
`<parcels_stem>_parcel_up_candidates.csv` next to the input parcel file.
It retains *all* candidate destinations and their parcel-local pixel counts,
contributing areas in m², drainage shares, selection/exclusion status and
reason. Output statuses include `selected`, `rejected_cycle`,
`rejected_secondary`, `outside_parcel_domain`, `unresolved_ambiguous`, and
`unresolved_not_rasterized`. Use `--diagnostics-out FILE` to change its path.
Inspect this file, especially cycle rejections and unresolved parcels, before
using `parcel_up` for wetland BMP analysis. `--abort-on-ambiguity` optionally
fails *before* modifying the GeoPackage when any parcel's dominant outlet is
ambiguous or its geometry is too small for the DEM raster; the CSV is still
written for investigation. Without this option unresolved parcels remain
unconnected and a warning is logged. The 30-meter 3DEP DEM may be too coarse
for small parcels or divergent drainage paths. Higher-resolution data or
manual checking against known streams can be needed.

For example, parcel 1 draining into 2, then 2 into 3, writes only:

```text
pid  pid_up
2    1
3    2
```

It does **not** write `(3,1)` because that is an indirect upstream ancestor.
A downstream parcel can have multiple immediate upstream neighbors. The
script still validates and writes the normalized graph transactionally.
WhiteboxTools does not remove uncertainty from DEM quality, missing flow
structures or rasterized parcel boundaries. Its outputs require review.

Official WhiteboxTools documentation:
[BreachDepressionsLeastCost](https://jblindsay.github.io/wbt_book/available_tools/hydrological_analysis.html),
[D8Pointer](https://github.com/jblindsay/whitebox-tools/blob/master/whitebox-tools-app/src/tools/hydro_analysis/d8_pointer.rs),
[Python frontend](https://whitebox.readthedocs.io/en/latest/readme.html).

For a model-compatible `.gpkg`, the script updates `parcel_up` **by default**:

```text
id   INTEGER PRIMARY KEY AUTOINCREMENT
pid  INTEGER NOT NULL
pid_up  INTEGER NOT NULL
UNIQUE(pid, pid_up)
```

On the model's standard `parcels(pid INTEGER PRIMARY KEY)` GeoPackage it also
creates the appropriate foreign keys to `parcels(pid)`. The resulting table
is registered as a nonspatial `attributes` layer in `gpkg_contents`. Other
GeoPackage tables and the parcel geometries remain unchanged. If the input
GeoPackage stores `pid` as a nonunique attribute rather than a declared unique
key, the table is still generated without SQL foreign keys; model-level parcel
ID and cycle validation still applies.

**Replacing `parcel_up` is intentional.** A SQLite transaction ensures the
previous table and contents remain intact if an update fails. Once a valid
replacement commits, the old connectivity table is not retained; back up
important original connectivity data before running the command.

Add `--out path/to/parcel_up.csv` to export a normalized CSV, **one pair per
row**, in addition to writing the GeoPackage table. Rows with no upstream
connections are omitted rather than represented as blank `pid_up` values.
Use `--no-gpkg-update --out path/to/parcel_up.csv` to generate only the CSV
and leave the GeoPackage unchanged. Inputs other than GeoPackage require
`--out` for CSV-only output; the model expects a `parcels` geometry layer with
integer `pid` keys for direct GeoPackage updates.

```bash
# Local DEM, write parcel_up to original geopackage using WhiteboxTools:
python utils/create_parcel_up.py path/to/parcels.gpkg path/to/dem.tif \
    --diagnostics-out parcel_connection_candidates.csv

# Download 3DEP and write parcel_up to original geopackage:
python utils/create_parcel_up.py path/to/parcels.gpkg --usgs-resolution 30m

# CSV-only diagnostic run (no GeoPackage changes):
python utils/create_parcel_up.py path/to/parcels.gpkg path/to/dem.tif \
    --no-gpkg-update --out direct_connections.csv
```

### Verify uncertain drainage before updating the GeoPackage

```bash
# Produce a CSV plus candidate evidence without changing the input GeoPackage:
python utils/create_parcel_up.py path/to/parcels.gpkg path/to/dem.tif \
    --no-gpkg-update --out parcel_up.csv --diagnostics-out candidate_edges.csv

# Strict option: refuse an update when a parcel cannot be assigned a unique,
# sufficiently dominant outlet from the raster evidence.
python utils/create_parcel_up.py path/to/parcels.gpkg path/to/dem.tif \
    --abort-on-ambiguity
```

WhiteboxTools routing requires a projected DEM; nominal raster resolution and
pixel size should be appropriate for the parcel size. The script reports
raster geometry or NoData mismatches instead of inferring flow through them.
