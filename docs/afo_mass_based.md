# Optional mass-based AFO loading: APLE manure P + runoff-responsive N

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
feedlot_bmp_cps: [313] # optional: selected BMP codes exclusively treating feedlot source
```

When `manure_loading` is omitted, the existing PLET concentration-increment mode
is still the default. `mass_based` requires model pollutants `TN` and `TP`.
All AFO IDs and PIDs must refer to the modeled parcel universe. Baseline loading
without manure and direct feedlot sources is a **required assumption** for both modes.
An annual statistical load based on previously manured fields is NOT automatically
made manure-free and adding manure loads to it can double count.

## Equations and units

The following are calculated once per Monte Carlo scenario. Animal counts are
realized once and reused for manure production and direct feedlot density.

For an operation with animal classes `s`:

```
wet_manure_kg = sum(count_s * days_s * manure_kg_head_day_s)
excreted_N_kg = sum(count_s * days_s * tn_kg_head_day_s)
excreted_P_kg = sum(count_s * days_s * tp_kg_head_day_s)
field_N_kg = excreted_N_kg * collection_fraction * (1-export_fraction) * n_retention_fraction
field_P_kg = excreted_P_kg * collection_fraction * (1-export_fraction) * p_retention_fraction
field_manure_kg = wet_manure_kg * collection_fraction * (1-export_fraction)
```

Field nutrient and wet manure masses are allocated proportionally to eligible
hectares across an operation's selected receiving parcels, unless all explicit
receivers specify positive `weight` values (then use normalized weights).
**Allocated N and P are conserved across receiving parcels**, with multiple
operations contributing additive nutrient masses to overlapping recipients.
No crop demand/cap or complete manure lifecycle is modeled. Nitrogen handling
losses enter before the runoff availability factor and must not be duplicated.

### Annual P export: APLE 2.5 manure-derived dissolved-P component

For parcel `i`, applied P mass `M_P` in kg, surface runoff `Q` and annual total
precipitation `P` in mm, use the actual annual runoff ratio `r=Q/P` (0 to 1).

```
surface_P = M_P * (0.4 if solids_pct < 15 else 1.0) * (1 - incorporated_fraction)
WEP_available = surface_P * (wep_fraction + (1-wep_fraction) * mineralization_season)
WEP_available *= 0.75 if season == 'fall' else 1
WEP_available *= [2.2*(250*0.5)/(250*0.5+300.1)] if solids_pct < 15 else 1
P_runoff_dissolved_kg = WEP_available * r**1.225
```

Annual non-WEP mineralization coefficients from USDA APLE 2.5 are winter=0.20,
spring=0.15, summer=0.10, fall=0.05. For fall, only 75% of runoff-available
WEP is counted in the current year; the other 25% is **not** carried into
future years in this version. The 60% immediate infiltration of liquid manure
P and 50% surface coverage come from APLE. The incorporation term is an
**effective user-specified fraction** removed from the soil surface: APLE's
full injection-rate/depth-dependent module is **not** replicated. This is the
APLE *manure-derived dissolved P submodel*, **not** complete APLE TP,
soil-P pools, sediment-P, crop uptake, or multi-year behavior.

Reference: USDA ARS, *Annual Phosphorus Loss Estimator, Theoretical
Documentation*, ver. 2.5 (2017), equations 1, 3 and 6, pp. 3–4:
https://www.ars.usda.gov/ARSUserFiles/50901500/SOFTWARE/APLE/APLETHEORETICALDOCUMENTATION25Rev.PDF

### Annual N export: APLE-inspired, not an APLE TN formulation

```
N_runoff_kg = applied_N_kg * f_n * r**alpha_n
```

`alpha_n` defaults to 1.0 (parsimonious modeling assumption); `f_n` is
**required** and must be user specified or sampled between 0 and 1. It is a
runoff-transfer/availability coefficient *not* a calibrated literature
constant. `f_n` may NOT simply be set to an observed runoff-loss fraction:
actual exported fraction is `f_n * (Q/P)**alpha_n`. The MANAGE v6 analysis
was insufficient to independently identify both parameters from manure-only
incremental TN loads. Subsurface manure TN/TP increases remain outside scope.

The computed manure-derived `TN` and dissolved `TP` export masses are
converted to parcel-total kg/ha/year by dividing by parcel area, and added to
the **surface** pathway. For PLET output, the additional equivalent annual
concentration is `100 * (kg/ha/yr) / (runoff_mm)` mg/L, when runoff > 0;
there is **no extra months/12 factor**. Zero runoff produces zero export.
The modeled manure-derived load never exceeds applied nutrient mass.

### Direct feedlot runoff

Uses the existing PLET pollutant-specific feedlot animal-equivalence and
manure-pack concentration method. Feedlot wetland/land-cover background loads
are displaced over each feedlot footprint to avoid double counting. In PLET,
feedlot runoff is calculated using the documented 91–94 feedlot CN and the
existing PLET event climate. In **statistical** mode, it uses a separate,
user-supplied `feedlot_runoff_mm` (annual); this is a runoff-depth substitution,
not an independent event-CN simulation. Direct feedlot surface loads and
manure-applied-field loads remain separate BMP components. The animal-excretion
factors and feedlot animal-equivalence factors are **not interchangeable**.

## GeoPackage tables

Existing tables `afo_operations`, `afo_animals`, and `afo_manure_fields` are
retained. The old `afo_manure_concentration_increments` table is **not used**
when `method: mass_based`. New/extended fields:

| Table | Columns | Meaning |
|---|---|---|
| `afo_animals` | `species`, `count` (or `count_value`, `count_mean`, etc.), `animal_days` (default 365) | One animal inventory row per class and operation |
| `afo_animals` | `manure_kg_head_day`, `tn_kg_head_day`, `tp_kg_head_day` | Optional overrides of bundled species factors, as-excreted kg per animal-day |
| `afo_manure_management` | `afo_id`, `season`, `solids_pct`, `wep_fraction`, `f_n` | **Required** per operation; season winter/spring/summer/fall |
| `afo_manure_management` | `collection_fraction`, `export_fraction`, `n_retention_fraction`, `p_retention_fraction`, `incorporated_fraction`, `alpha_n` | Optional: defaults 1, 0, 1, 1, 0, 1, respectively |
| `afo_manure_fields` | `weight` | Optional positive allocation weight: specify for **all** receivers of an operation or none |
| `afo_field_eligibility` | `pid`, `eligible` (0/1) | **Statistical only**, receiver eligibility for a parcel; no PLET land-cover input required |
| `afo_hydrology` | `pid`, optional `huc12`, `annual_precip_mm`, `annual_runoff_mm`, `feedlot_runoff_mm` | **Statistical only**: annual total precipitation, field surface runoff, and operation feedlot surface runoff |

When in statistical mode, `afo_hydrology` rows may use an exact integer
`pid`, a `huc12` (with NULL pid), or NULL `pid`/`huc12` as the domain default;
precedence is **parcel > HUC12 > domain**. HUC12 associations come from the
canonical `parcels.gpkg/parcel_huc12` mapping if present; otherwise use parcel
or domain rows. At least one eligible agricultural parcel and valid hydrology
for receivers and operation hosts are required. `feedlot_runoff_mm` must be
provided for each operation host. User-supplied Q/P must satisfy 0 ≤ Q ≤ P.

Fields representing uncertain scalar values may use existing inline
`*_value`, `*_mean`, `*_sd`, `*_min`, `*_max`, `*_pXX` statistics in place of
fixed values. Distributions are drawn independently within the existing
scenario RNG unless the input table/source provides a deliberate grouping.

### Starting animal excretion factors

`src/data/manure_excretion_reference.csv` provides documented, editable
starting values for five animal classes: `beef_finishing`, `dairy_lactating`,
`swine_grow_finish`, `poultry_broiler`, `sheep_feeder`. Aliases `beef_steer`,
`dairy_cow`, `swine`, and `broiler` are accepted. Reference values were
converted from ASABE D384.2 excretion tables quoted by Michigan State
University Extension (livestock animal manure kg/head/day, N kg/head/day,
P2O5 to **elemental P** using 0.4364). Default values are *as excreted*,
not post-storage and not calibrated to East Fork livestock. Override per
animal inventory row as needed; unfamiliar species must supply all three
explicit rates. Data source:
https://www.canr.msu.edu/resources/manure_management_for_fairs_and_exhibitions

## Outputs and validation

`afo_manure_assignments/sN.parquet` contains each realized AFO-to-field
allocation (before BMPs). `afo_mass_ledger/sN.parquet` adds annual generated
manure/N/P, applied manure/N/P, exported surface N/dissolved P, field runoff,
precipitation, effective coefficients and hectare-based application rates.
`afo_operation_budgets/sN.parquet` contains one row per AFO with generated,
available, allocated and surface-exported N/P masses, plus automatic
allocation-conservation diagnostics. **Operation-wide masses in the per-field
ledger repeat on every receiver row; do not sum them across receivers.**
`afo_load_components/sN.parquet` records before/after loads separated by
feedlot/manure-field source and transport pathway. Receivers are sampled once
per realization and unchanged along that realization's BMP trajectory.

Input validation rejects unknown parcel IDs, impossible footprints, missing
management records, invalid excretion factors, missing hydrology and invalid
surface pathway names. Runtime validation checks the sampled runoff/precip
ratio, nutrient availability bounds, impossible receiving fields, and mass
export constraints. Other loads, sediment and groundwater use the original
PLET or statistical methodology; no manure-related subsurface loading is added.

## Executable example

From the repository root:

```bash
python scripts/create_afo_mass_example.py
python run_model.py examples/east_fork/inputs/config/east_fork_afo_mass_demo.yaml
```

The script creates `afo_mass_example_generated.gpkg` by copying the existing
AFO demo and adding attribute tables. The example livestock locations, manure
factors, runoff, WEP and N fractions are **illustrative only**. Do not
interpret its outputs as measurements of manure in the East Fork watershed.
A statistical-mode example can reuse the generated AFO GeoPackage with any
validated `statistical` baseline config that includes TN and TP, provided the
configured `surface_pathway` matches its statistical pathway names and the
background rates exclude the separately modeled manure/feedlot loads.
