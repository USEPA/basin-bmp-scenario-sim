# Mass-based AFO loading: annual surface nutrient export fractions

This extension allows the **same** annual animal inventory, manure nutrient production,
spatial allocation, source-specific BMP reductions, and output ledger to operate in
`plet_rusle` **and** `statistical` load-generation modes. All quantities are one-year
quantities. This method is a **source addition** to a baseline assumed to exclude
manure and feedlot loading represented by the module. Do not use without checking
for baseline double counting. The pre-existing `concentration_increment` mode
remains available for PLET/RUSLE, with no changes to its equations or input tables.

```yaml
animal_operations: ../afo/afo_mass_example_generated.gpkg
manure_loading:
  method: mass_based
  surface_pathway: surface # must be one of the statistical-mode pathways
  runoff_adjusted_export: false # default (Option A)
  # Set to true (Option B) and specify reference_runoff_mm (positive, mm/year).
  # reference_runoff_mm: 200.0 # illustrative value ONLY
  # excretion_factors: ../afo/my_excretion_factors.csv # optional replacement CSV
feedlot_bmp_cps: [313] # optional: selected BMP codes exclusively treating feedlot source
```

When `manure_loading` is omitted, the existing PLET concentration-increment mode
is still the default. `mass_based` requires model pollutants `TN` and `TP`.
All AFO IDs and PIDs must refer to the modeled parcel universe. Baseline loading
without manure and direct feedlot sources is a **required assumption** for both modes.
An annual statistical load based on previously manured fields is NOT automatically
made manure-free and adding manure loads to it can double count.

## Example GeoPackage metadata and coefficient provenance

The example GeoPackages store value explanations, units, methodological
citations, and the **synthetic** provenance of each example directly in a
`notes` column in the corresponding `afo_*` input table. `afo_operations` is
also a POINT layer whose locations are the centroids of host parcel polygons
matched by `pid`; these are illustrative, not observed facility locations.
See [afo_example_metadata.md](afo_example_metadata.md). **Do not treat the
example export fractions or manure handling factors as sourced, calibrated,
or recommended values.** Methods cited in the notes describe model behavior,
not the chosen example numbers.

## Equations and units

The following are calculated once per Monte Carlo scenario. Animal counts are
realized once and reused for nutrient production and direct feedlot density.

For an operation with animal classes `s`:

```
excreted_N_kg = sum(count_s * days_s * representative_mass_kg_s / 1000 * tn_kg_1000kg_day_s)
excreted_P_kg = sum(count_s * days_s * representative_mass_kg_s / 1000 * tp_kg_1000kg_day_s)
field_N_kg = excreted_N_kg * collection_fraction * (1-export_fraction) * n_retention_fraction
field_P_kg = excreted_P_kg * collection_fraction * (1-export_fraction) * p_retention_fraction
```

Here, **mass-based** refers to N and P mass, not total wet-manure mass.
Wet manure production is not sampled, calculated, allocated, or reported.
Representative animal mass remains necessary to scale the reference nutrient
rates. The method does not simulate manure solids or APLE phosphorus availability.

Field nutrient masses are allocated proportionally to eligible
hectares across an operation's selected receiving parcels, unless all explicit
receivers specify positive `weight` values (then use normalized weights).
**Allocated N and P are conserved across receiving parcels**, with multiple
operations contributing additive nutrient masses to overlapping recipients.
This is **allocation conservation only**: the separately estimated direct
feedlot runoff is **not deducted** from manure available for field allocation,
and the model does not enforce a combined feedlot-plus-field annual nutrient
mass balance. Users must assess whether treating these two sources as separate
increments is defensible for their inventory and management assumptions.
No crop demand/cap or complete manure lifecycle is modeled. Nitrogen handling
losses enter before field export fractions and must not be duplicated.

### Optional coupling of feedlot BMPs to field manure availability

A feedlot BMP may increase the fraction of manure collected for subsequent
field application (for example, collection/storage of previously uncollected
material). The optional **nonspatial** `afo_feedlot_manure_effects` table in the
AFO GeoPackage describes this effect **per operation and feedlot-only CPS**.
The table is optional and omitted by the shipped synthetic examples. Include
one row for each desired `(afo_id, cps)` pair:

| Column | Meaning |
|---|---|
| `afo_id` | Existing operation ID in `afo_operations` |
| `cps` | Integer CPS code also listed in YAML `feedlot_bmp_cps` |
| `additional_collection_fraction` | Fraction (0–1) of *remaining uncollected* nutrient mass collected when the BMP is installed; may use supported `additional_collection_fraction_*` distribution columns instead |
| `notes` | Optional justification and site-specific provenance (ignored by the loader) |

For example, a GeoPackage edited in SQLite could add the table with:

```sql
CREATE TABLE afo_feedlot_manure_effects (
    afo_id TEXT NOT NULL,
    cps INTEGER NOT NULL,
    additional_collection_fraction REAL NOT NULL,
    notes TEXT
);
INSERT INTO afo_feedlot_manure_effects
VALUES ('AFO_DEMO_01', 313, 0.25,
        'Illustrative recovery assumption; replace with documented site data');
```

This would recover 25% of that operation's previously uncollected manure
when CPS 313 is actually selected on its host parcel, provided that CPS 313
is in `feedlot_bmp_cps`. **The 0.25 value is illustrative, not an EPA/NRCS
efficiency or recommended value.** The model cannot derive this
fraction from the CPS code or pollutant-removal efficiency: it must be supplied
and justified independently. With baseline collection fraction `c` and BMP
recovery fraction `r`, the new fraction is

```text
c_after = c_before + (1 - c_before) * r
field_available_N_after = excreted_N * c_after * (1-export_fraction) * n_retention_fraction
field_available_P_after = excreted_P * c_after * (1-export_fraction) * p_retention_fraction
```

For multiple selected BMPs, this equation is applied sequentially to the
remaining uncollected fraction, never exceeding 1. BMP failure scales `r`
with the same failure-reduction factor as its runoff efficiencies. TN/TP
availability is reallocated over the **same already selected field parcels and
weights**, and the existing Option A/B field export equations are reapplied.
An earlier field BMP's reduction continues to apply to any newly collected
manure on that receiving parcel. A later field BMP treats the expanded field
source. Net BMP loads and outlet reductions include induced field export from
the **receiving parcels**, which can partially offset a feedlot reduction.

This is an *explicit collection-management scenario*, not proof that avoided
feedlot runoff automatically becomes land-applied manure. Feedlot runoff and
manure production remain independently estimated: **the optional coupling
does not create a complete operation-wide N/P mass balance**. Other BMPs and
all BMPs without a matching operation/CPS row continue to reduce their assigned
load source without changing collection. The older concentration-increment
method does not accept this table. The spatial receiving-parcel assignments
are fixed within a realization; only annual nutrient masses change.

### Field surface loading: Option A (default)

For each receiving parcel `i`, let `M_N,i` and `M_P,i` be its allocated annual
manure TN/TP mass (kg/year) **after** the collection/export/retention factors.
A management row supplies two **annual** surface export fractions `f_N` and `f_P`
from 0 through 1 (fixed values or distributions):

```text
TN_surface_kg = M_N,i * f_N
TP_surface_kg = M_P,i * f_P
```

`manure_loading.runoff_adjusted_export` defaults to **false**, so field
surface runoff, precipitation, month count, season, incorporation, manure
solids and APLE equations do not affect the increment. These empirical
fractions are not supplied/calibrated by the model; users must select and
document plausible values. They are not the old `f_n` runoff-availability
coefficient and should **not** be inferred by mechanically converting APLE
inputs. For example, values `f_N=0.05` and `f_P=0.02` would be **hypothetical
illustrations only**, not recommended defaults. Each fraction is drawn **once
per operation per scenario** and then applied to all its receivers. The model
never increases subsurface manure loads.

Because this mode does not depend on runoff, it can produce a positive
surface load when field runoff is zero. This is an acknowledged screening-
level inconsistency (and may be inappropriate for a runoff-only physical
interpretation), not an implicit zero-runoff cutoff. In PLET mode the
reported equivalent concentration is updated only if annual runoff is
positive; it cannot represent a positive load at zero water volume.

### Option B (explicit opt-in): annual runoff adjustment

```yaml
manure_loading:
  method: mass_based
  runoff_adjusted_export: true
  reference_runoff_mm: 200.0  # user-provided reference, illustrative only
```

For annual parcel surface runoff depth `Q_i` (mm/year) and a strictly positive
reference depth `Q_ref` (mm/year):

```text
f_N_effective = min(1, f_N * Q_i/Q_ref)
f_P_effective = min(1, f_P * Q_i/Q_ref)
TN_surface_kg = M_N,i * f_N_effective
TP_surface_kg = M_P,i * f_P_effective
```

`f_N` and `f_P` are the export fractions **at the reference runoff**; the
ratio is dimensionless. For positive rainfall and real physical runoff this
method is a runoff-responsive approximation, not a mechanistic fate model or
validated universal relationship. At `Q_i=0` it returns zero manure-field
surface export, and each export is capped at the applied nutrient mass.
PLET's modeled annual surface runoff is used in PLET mode; in statistical
mode `afo_hydrology.annual_runoff_mm` (possibly sampled) is used. The
reference depth must be chosen and justified for the region, not inferred
from the annual precipitation ratio or hard-coded. If adjustment is off,
providing `reference_runoff_mm` is an error rather than a silently ignored
setting. The option is **not** available to the legacy
`concentration_increment` method.

### Scope and accounting

The field exports are *manure-attributable annual surface TN and TP increments*
added to the configured surface pathway, **not dissolved P alone** and not
total field export. Other background loads must exclude the sources already
represented here, or manure could be counted twice. No prior-year manure,
legacy soil pool, mineralization, crop uptake or manure-driven groundwater
increment is modeled. The annual export fractions encode these omitted
processes only as an empirical same-year approximation.

Nutrient masses are divided by **total parcel area** to obtain added
kg/ha/year, preserving mass when summed over the field. In PLET, if modeled
runoff is positive, the equivalent annual concentration addition is
`100 * (kg/ha/year) / runoff_mm` mg/L. In statistical mode, the incremental
kg/ha/year surface rate is added directly; there is no concentration state.
Manure allocations remain mass conserving, and both methods keep exported
TN/TP at or below applied TN/TP.

### Direct feedlot runoff

Uses the existing PLET pollutant-specific feedlot animal-equivalence and
manure-pack concentration method. Feedlot wetland/land-cover background loads
are displaced over each feedlot footprint to avoid double counting. In PLET,
feedlot runoff is calculated using the documented 91–94 feedlot CN and the
existing PLET event climate. In **statistical** mode, annual feedlot runoff is **always derived from**
`afo_hydrology.annual_precip_mm` with the same PLET curve-number formula and
feedlot imperviousness CN table (91, 92, 93, or 94). It uses PLET's zero
initial abstraction, 120 rainy days/year by default (optional `rain_days`
fixed or distribution override in `afo_hydrology`), a rainfall correction
fraction of 0.90, and a runoff-day fraction of 0.50. These are transparent
**screening-level assumptions**, *not* universal spatial PLET defaults.
PLET ordinarily derives rain-day and correction values from local precipitation
data. Replace the assumed `rain_days` with local values where available and
interpret runoff sensitivity accordingly. `feedlot_runoff_mm` is no longer
accepted as an input. Derived annual runoff is necessarily no greater than
annual precipitation with the specified PLET factors. Direct feedlot surface loads and
manure-applied-field loads remain separate BMP components.

**Sediment/TSS limitation (applies to both AFO methods):** The feedlot
occupies land that previously generated parcel background load, so the model
reduces *all* background pollutants/pathways by the occupied area fraction.
The replacement direct-feedlot routine currently estimates TN/TP (and BOD
where enabled), **not TSS**. Consequently, the feedlot footprint contributes
no direct sediment source, and baseline TSS can appear to decrease when AFOs
are added. Do not treat the resulting TSS load as a complete feedlot sediment
estimate or interpret that decrease as a real environmental benefit. A
mechanistic feedlot sediment export module is outside the scope of this PR.
 The animal-excretion
factors and feedlot animal-equivalence factors are **not interchangeable**.
**`animal_days` scales annual excreted nutrient mass but does not scale direct
feedlot loading**: the latter uses the realized `count`, feedlot area, and
annual runoff. A seasonal or short-duration animal inventory therefore needs
special care when interpreting year-round direct feedlot loading. **This is an
explicit PLET-style annual occupancy assumption, not a physical prediction of
seasonal loading.** `count` represents the inventory driving feedlot manure-pack
concentrations throughout the modeled precipitation year, while `animal_days`
controls only annual excretion and field-applied manure mass. For example, an
inventory present 30 versus 365 days yields the same direct feedlot loading
when count, area, and rainfall are unchanged, but different manure production.
Do **not** infer that the model accounts for seasonal correlations between
occupancy and rainfall. A mechanistic occupancy/rainfall treatment is deferred.

## GeoPackage tables

Existing tables `afo_operations`, `afo_animals`, and `afo_manure_fields` are
retained. The old `afo_manure_concentration_increments` table is **not used**
when `method: mass_based`. New/extended fields:

| Table | Columns | Meaning |
|---|---|---|
| `afo_animals` | `species`, `count` (or `count_value`, `count_mean`, etc.), `animal_days` (default 365) | One animal inventory row per class and operation |
| `afo_animals` | `representative_mass_kg` | Optional override of required reference mass; fixed or inline distribution, strictly positive kg per animal |
| `afo_animals` | `tn_kg_1000kg_day`, `tp_kg_1000kg_day` | Optional overrides of nutrient excretion rates, kg N or P per 1,000 kg live animal mass per day |
| `afo_manure_management` | `afo_id`, `tn_surface_export_fraction`, `tp_surface_export_fraction` | **Required** per operation; each 0–1, either a scalar or a `_value`/`_mean`/`_sd`/percentile distribution |
| `afo_manure_management` | `collection_fraction`, `export_fraction`, `n_retention_fraction`, `p_retention_fraction` | Optional: defaults 1, 0, 1, 1 respectively |
| `afo_operations` | `manure_months` | Ignored for `mass_based`; retained only for legacy concentration-increment runs |
| `afo_manure_fields` | `weight` | Optional positive allocation weight: specify for **all** receivers of an operation or none |
| `afo_field_eligibility` | `pid`, `eligible` (0/1) | **Statistical only**, receiver eligibility for a parcel; no PLET land-cover input required |
| `afo_hydrology` | `pid`, optional `huc12`, required host `annual_precip_mm`, optional `rain_days`, `annual_runoff_mm` | **Statistical only**: feedlot runoff is computed from precipitation using PLET CN and documented climate assumptions; receiving fields need `annual_runoff_mm` only when Option B is enabled |

Explicit `afo_manure_fields.weight` values must be supplied for **every**
receiver of an AFO or for **none**; partial weighting is rejected at input
loading (before Monte Carlo execution). In statistical mode, sampled
`afo_operations.n_fields` values and supplied distribution bounds are checked
against the number of eligible parcels with **positive area remaining after
all feedlot footprints are subtracted**. Unbounded count distributions are
truncated to that maximum during sampling. In PLET/RUSLE mode eligibility
can vary by realization, so exceeding its realized field count remains a
runtime error. Optional statistical `rain_days` distributions are physically
truncated to the inclusive range 1–366 days/year even without explicit
`_min`/`_max` statistics.

When in statistical mode, `afo_hydrology` rows may use an exact integer
`pid`, a `huc12` (with NULL pid), or NULL `pid`/`huc12` as the domain default;
precedence is **parcel > HUC12 > domain**. HUC12 associations come from the
canonical `parcels.gpkg/parcel_huc12` mapping if present; otherwise use parcel
or domain rows. At least one eligible agricultural parcel and valid hydrology
for receivers and operation hosts are required. Host `annual_precip_mm`
accepts the usual fixed/distribution statistics; the optional `rain_days`
parameter can be fixed or distributed in the range 1–366. The obsolete
`feedlot_runoff_mm` input and all its distribution columns are rejected,
including when null. Host feedlot runoff is derived internally and satisfies
0 ≤ Q ≤ annual precipitation; user-supplied `annual_runoff_mm` remains a
separate, optional field-only input for Option B, not a feedlot runoff override.

### Feedlot-specific BMP costing (all AFO methods)

A feedlot-only CPS is eligible on an AFO host parcel, but the full parcel is
**not** automatically its treated area for costing. The model computes BMP
pollutant treatment exactly as before and separately computes its cost using:

- **USD/ha components**: the sum of each hosted AFO's `feedlot_area_ha` by
  default. If a practice covers a different documented extent, override the
  corresponding operation/CPS row with `cost_area_ha`.
- **USD/m components**: an explicitly supplied `cost_length_m` for **every**
  operation hosted on a parcel that could receive this CPS. Length cannot be
  inferred from feedlot area. If a feedlot CPS has a USD/m cost component but
  an eligible operation is missing its length, input loading fails *before*
  scenarios start.
- **USD/project components**: one project charge per parcel/CPS placement,
  regardless of the number of AFO operations sharing that parcel.
- **Multiple cost components**: each component is sampled independently per
  BMP placement, then added. The same cost basis is used when calculating
  representative costs for optional inverse-cost BMP selection.

You may add the optional nonspatial table `afo_feedlot_bmp_cost_dimensions`
inside the configured AFO GeoPackage (works in both mass-based and legacy
concentration-increment AFO modes). It contains one row per `(afo_id, cps)`:

| Column | Description |
|---|---|
| `afo_id` | An existing operation ID |
| `cps` | A CPS code listed in `feedlot_bmp_cps` |
| `cost_area_ha` | Optional positive area to use **instead of** that operation's feedlot footprint for USD/ha costs |
| `cost_length_m` | Optional positive physical length required for a USD/m cost component |
| `notes` | Optional documentation, ignored by the loader |

At least one dimension must be provided in each row. Omitting the entire table
means footprint-based area costs and ordinary per-project charges still work,
but a feedlot-specific USD/m component will require the table. Units are
**hectares** and **meters**, respectively. Do not use a whole-parcel area for a
feedlot facility unless it is actually the treated area. Multiple AFOs on a
single host have their areas and lengths summed for one placement, and one
USD/project charge per placement. No inferred length or fabricated cost rate is
introduced by this change.

Sample optional table creation:

```sql
CREATE TABLE afo_feedlot_bmp_cost_dimensions (
  afo_id TEXT NOT NULL,
  cps INTEGER NOT NULL,
  cost_area_ha REAL,
  cost_length_m REAL,
  notes TEXT
);
INSERT INTO afo_feedlot_bmp_cost_dimensions
VALUES ('AFO_DEMO_01', 313, 0.35, 80.0,
        'Illustrative facility dimensions; replace with documented estimates');
```

### Unified input validation

All numeric model input tables share the same fixed-versus-distribution
validation rules. In wide AFO tables, for example, `count` cannot appear
alongside populated `count_*` statistics, nor may `collection_fraction`
coexist with `collection_fraction_mean` or `collection_fraction_value`.
The same rule applies to animal days, excretion overrides, manure export and
retention fractions, feedlot recovery inputs, statistical hydrology and sampled
field counts. Negative animal counts, incomplete/invalid distributions, and
animal-day means/endpoints outside **0–366 days** are rejected at input load.
Normal distributions with valid means are truncated to that interval on
sampling. An omitted `animal_days` defaults to 365 in mass-based mode.
Validation errors identify the relevant table, operation and parameter.

Fields representing uncertain scalar values may use existing inline
`*_value`, `*_mean`, `*_sd`, `*_min`, `*_max`, `*_pXX` statistics in place of
fixed values. Distributions are drawn independently within the existing
scenario RNG unless the input table/source provides a deliberate grouping.

### Starting animal excretion distributions

The default [manure excretion CSV](../src/data/afo/manure_excretion_reference.csv)
uses the same [inline distribution schema](input_distributions.md) as other
numeric model inputs: **one row per species and parameter**, not separate
parameter-specific mean/SD columns. Its required identifiers are `species`
and `parameter`; parameters are `representative_mass_kg`,
`tn_kg_1000kg_day`, and `tp_kg_1000kg_day`.
Supply all three rows for every species included in the CSV. Representative
mass is a required numerical input, not optional source metadata.

```csv
species,parameter,value,mean,sd,units
Beef,representative_mass_kg,360,,,kg
Beef,tn_kg_1000kg_day,,0.34,0.073,kg/1000kg/day
Beef,tp_kg_1000kg_day,,0.092,0.027,kg/1000kg/day
```

Rows may instead use `value`, bounded distributions, or percentile statistics
under the common schema. Do not mix `value` with distribution statistics.
Rates are nonnegative. Their canonical unit `kg/1000kg/day` means **kg per
1,000 kg live animal mass per day**, as in the PDF; `g/1000kg/day` is also
supported. Representative mass uses `kg` (also accepting `g` or `lb`) and
must be strictly positive. Omitted units imply the parameter's canonical
unit. Unit conversions apply equally to all supplied statistics.
`source` is provenance metadata, not a runtime multiplier.

The default CSV is included with the model and located relative to the code,
not the working directory. `manure_loading.excretion_factors` optionally
selects a replacement CSV, relative to the YAML file. A custom file replaces
the default reference entirely; missing files or invalid rows fail input
validation rather than falling back silently. The selected CSV is read and
validated once before simulation workers start. It is not a GeoPackage table
and is not used by `concentration_increment` mode.

#### D384.1 source and conversion assumptions

Means and SDs come from **ASAE D384.1 FEB03, Table 1, pp. 683-684** in
[ASAE_D384.1.pdf](../src/data/afo/ASAE_D384.1.pdf), not the separate D384.2
document. The source reports total Kjeldahl N (TKN) and elemental P excreted
in fresh whole manure (feces and urine combined) per 1,000 kg live animal mass
per day. The CSV retains these native nutrient mean/SD values. The simulation draws the rates
and multiplies them by the realized representative mass divided by 1,000.
The standard animal names and default fixed masses from the table footnote are:

| Species | Representative mass (kg) |
|---|---:|
| Dairy | 640 |
| Beef | 360 |
| Veal | 91 |
| Swine | 61 |
| Sheep | 27 |
| Goat | 64 |
| Horse | 450 |
| Layer | 1.8 |
| Broiler | 0.9 |
| Turkey | 6.8 |
| Duck | 1.4 |

Names are case-insensitive in both the CSV and `afo_animals`. Legacy names
`beef_steer`/`beef_finishing`, `dairy_cow`/`dairy_lactating`,
`swine_grow_finish`, `poultry_broiler`, and `sheep_feeder` remain accepted as
aliases, but new inputs should use the standard names above. Unknown names
fail validation. These are broad categories, not production-stage-specific
measurements. The rates are **as excreted**, not post-storage, and are not calibrated
to East Fork livestock; site-specific data should replace them when available.

**Nitrogen assumption:** nitrate and nitrite constitute an inconsequential
proportion of TN in fresh manure, so the PDF's **TKN is treated as representative
of TN**. Do not add ammonia N separately; it is already included in TKN.
Phosphorus is already elemental P, so no P2O5 conversion is applied. Urine
and feces are not simulated separately. N and P masses are calculated
from their own excretion rates, not from manure mass times concentrations.

#### Sampling and overrides

Each realization draws TN and TP rates independently using the existing
normal sampler **truncated at zero**. The supplied mean and SD describe the
underlying normal; truncation can shift the realized moments. The reported SDs
represent source-data variability, not standard errors of the mean: they are
not divided by the square root of animal count.

Mass rows use fixed PDF values by default. To vary mass, replace `value` with
`mean` and `sd` (or another complete supported distribution). The positive
mass draw is made once and reused for both nutrient production calculations.
For example, `Beef,representative_mass_kg,,360,30,kg` samples mass rather
than fixing it. Mass is independent of the native rate draws; because the
same mass scales N and P, stochastic mass introduces shared
variation in their resulting totals. This is not a manure-concentration
calculation. No mass is inferred from animal count.

Reference draws are shared within an operation/species combination for that
realization, but independent across operations and species. An optional
`sample_group` explicitly shares a reference draw for the same parameter
across grouped rows/operations. Draws are resampled each realization using the
scenario RNG, reproducibly in serial and parallel runs. No correlation among native rate draws is inferred from the source.

Animal-row fixed values or complete `parameter_*` distributions override the
selected reference **for that parameter**, without merging statistics.
For example, `tn_kg_1000kg_day=0.34` fixes the native N rate while P remains
stochastic; `tn_kg_1000kg_day_mean` requires a matching `tn_kg_1000kg_day_sd` unless
another complete supported distribution is supplied. Animal-row overrides
are sampled independently per inventory row. Mass can likewise be overridden
using `representative_mass_kg` or `representative_mass_kg_mean`/`_sd`.
Each operation's generated masses are calculated
once and then allocated to its fields, not redrawn per receiving field.

To make reference rates deterministic, use `value` rows or `mean`/`sd`
rows with `sd=0`.

## Outputs and validation

`afo_manure_assignments/sN.parquet` contains each realized AFO-to-field
allocation (before BMPs). `afo_mass_ledger/sN.parquet` adds annual generated
N/P, applied N/P, exported surface TN/TP (`tn_exported_kg`, `tp_exported_kg`),
sampled and effective export fractions, optional field runoff and reference depth,
and hectare-based application rates.
`afo_operation_budgets/sN.parquet` contains one row per AFO with generated,
**baseline** available, allocated and surface-exported N/P masses, plus automatic
allocation-conservation diagnostics. It also reports `final_tn/tp_available_kg`,
`final_tn/tp_applied_kg`, and `final_tn/tp_exported_kg` after the BMP sequence.
The final exported values reflect feedlot recovery and agricultural BMPs.
If a matching recovery effect is applied, `afo_manure_bmp_effects/sN.parquet`
records the operation, CPS, effective recovery fraction, before/after collection,
additional nutrient availability and induced field surface TN/TP exports.
The normal `bmps` output includes `induced_manure_tn/tp_kg_yr`,
`direct_bmp_removed_{pollutant}_kg_yr`, and the existing net
`removed_mass_{pollutant}_kg` (direct mass removed minus induced manure export).
For AFO BMPs that increase manure-field loading, `baseline_mass_{pollutant}_kg`
now includes the current pre-BMP loading of the host **and every receiving parcel
with nonzero induced export**, counted only once; `mass_accounting_pids` lists
this accounting boundary. Thus `overall_reduction_fraction_{pollutant}` uses a
spatially consistent net-removal denominator instead of host-only loading.
`treated_baseline_mass_{pollutant}_kg` still measures the directly treated source,
so `realized_efficiency_{pollutant}` may be negative or exceed familiar practice
ranges when it incorporates off-site induced manure; consult direct and induced
masses when interpreting feedlot BMP performance. These are placement-level
mass ratios, not deduplicated watershed-wide percentages; overlapping BMP
placements can contribute the same parcel baseline to multiple records.
**Operation-wide masses in the per-field ledger repeat on every receiver row;
do not sum them across receivers.**
`afo_load_components/sN.parquet` records before/after loads separated by
feedlot/manure-field source and transport pathway. Receivers are sampled once
per realization and unchanged along that realization's BMP trajectory.

Input validation rejects unknown parcel IDs, impossible footprints, missing
management records, invalid excretion factors, missing required host/field
hydrology, invalid runoff-adjustment configuration and invalid surface pathway
names. Runtime validation checks sampled export fraction bounds, impossible
receiving fields and mass constraints. Other loads, sediment and groundwater use the original
PLET or statistical methodology; no manure-related subsurface loading is added.

## Migration from the earlier APLE-based mass method

This is a **breaking change** for `method: mass_based` (the legacy
`concentration_increment` mode is unchanged):

1. Replace the `afo_manure_management` inputs `f_n`, `alpha_n`,
   `wep_fraction`, `solids_pct`, `season`, `incorporated_fraction` with
   `tn_surface_export_fraction` and `tp_surface_export_fraction`. Old
   APLE parameters are **rejected if populated**; there is no defensible
   numeric one-to-one conversion. Select/justify new annual export fractions.
2. Keep upstream `collection_fraction`, `export_fraction`,
   `n_retention_fraction`, `p_retention_fraction` and other AFO tables as before.
3. `afo_operations.manure_months` may remain for compatibility with mixed
   inputs but has no effect and is omitted from **mass-based** assignment
   and ledger outputs. It remains required in legacy concentration mode.
4. The former ledger/budget field `tp_dissolved_exported_kg` is now
   `tp_exported_kg` to reflect estimated **total** surface P, not a
   modeled dissolved-P subfraction. Update downstream analysis accordingly.
5. To retain a runoff-responsive *empirical* loss calculation, explicitly
   opt into Option B and provide a positive `reference_runoff_mm`.

The example GeoPackage shipped with this patch is updated to the new schema;
its 5% TN and 2% TP fractions are illustrative, **not** scientifically
calibrated defaults.

## Executable examples

From the repository root:

```bash
python run_model.py examples/east_fork/inputs/config/east_fork_afo_mass_demo.yaml
```

The supplied PLET-specific `afo_mass_example_generated.gpkg` contains
only four AFO input tables: operations, animals, explicit receiving fields,
and `afo_manure_management`. It deliberately omits the legacy manure-concentration
table and the statistical-only field-eligibility/hydrology tables. The
example livestock locations and TN/TP export fractions are **illustrative
only**. Do not interpret the outputs as measurements of manure in East Fork.

The [statistical-mode example](../examples/east_fork/inputs/config/east_fork_afo_mass_statistical_demo.yaml)
uses the synthetic statistical parcel package and a **separate**
`afo_mass_statistical_example.gpkg`. This adds only `afo_field_eligibility`
and `afo_hydrology` to the four mass-based tables. Its single domain-default
hydrology row includes annual precipitation and feedlot runoff for direct
feedlot loading. It intentionally omits `annual_runoff_mm`, because the
default Option A does not use runoff to calculate manure-field export. To
use Option B with this statistical example, supply receiving-field annual
runoff inputs and configure the positive reference runoff depth.

Run the default statistical (Option A) example with:

```bash
python run_model.py examples/east_fork/inputs/config/east_fork_afo_mass_statistical_demo.yaml
```

It explicitly sets `manure_loading.surface_pathway: surface` and partitions
aggregate background loads into `surface: 0.30` and `subsurface: 0.70`,
matching the two pathways in the AFO demo BMP efficiency CSV. These **30%/70%
pathway shares are demonstration assumptions, not an empirically established
partition for East Fork**. The fractions apply to background loads only; AFO
nutrient and direct-feedlot loads are added to the selected surface pathway.
The example assumes that background rates exclude these additional sources;
that exclusion is not independently verified by the demonstration. PLET
climate/HUC parameters are not required for this statistical example.

## Example input package reference

The three example AFO GeoPackages are intentionally distinct. In each,
`afo_operations` is a **POINT feature layer** at the geometric centroid of its
host parcel (using `pid` and the relevant example parcels GeoPackage). The
model still uses the **PID**, not this illustrative geometry, for calculations.
All other AFO inputs remain attribute tables, each with its own `notes` column.
No stand-alone documentation tables or dummy geometry layer are included.
AFO example-specific table/column inventories and validation are documented
in [the example GeoPackage reference](afo_example_geopackage_reference.md).

When changing the example GeoPackages, keep their row-level notes and host-point
geometry consistent with the input values and host parcel IDs. Inspect the
updated package and run its associated example configuration to check the
resulting manure allocations, operation budgets, and source-component loads.
See the [example GeoPackage reference](afo_example_geopackage_reference.md)
for the mode-specific table inventory and validation checks.

Changing the statistical example to Option B requires new receiving-field
runoff inputs; the committed GeoPackage is the default *Option A* example.
If you edit values manually, keep row-specific notes and illustrative POINT
geometries in sync with the updated fields and host parcels.
