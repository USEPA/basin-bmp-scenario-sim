# Animal feeding operations (AFOs) and parcel-specific manure exposure

## Scope and model interpretation

The optional `animal_operations` GeoPackage activates **PLET-style** direct
feedlot runoff and explicit/probabilistic field manure exposure in
`load_generation.mode: plet_rusle`. The model does **not** simulate a manure
nutrient mass balance, tonnage moved, fertilizer displacement, application-rate
response, manure decomposition, or increased groundwater concentrations.

The inputs are **additional to** the base PLET/RUSLE parcel/HUC12 inputs. The
original `input_surface_concentration` is interpreted as the *non-manured*
(background) concentration (mg/L) for a receiving parcel; a new AFO table
provides a **nonnegative additional concentration** associated with manure
application (mg/L), rather than an independently sampled absolute manured
concentration. The field-level concentration for fully eligible agricultural
acreage is:

`C_effective = C_nonmanured + (months/12) * delta_C_manure`.

For host parcels partly occupied by a feedlot, the model applies the increment
only to the eligible agricultural area. The effective whole-parcel concentration
reported in outputs is:

`C_parcel = C_nonmanured + (months/12) * delta_C_manure * (eligible_area_ha / parcel_area_ha)`.

`delta_C_manure` is sampled independently of `C_nonmanured`, using a
nonnegative distribution or fixed value. It may legitimately be numerically
smaller than the background concentration because it is an **increment**,
not an absolute manured concentration. A value of zero gives no manure-related
increase. Both background and increment are in **mg/L**. The fraction
`months/12` is the same annual exposure approximation used in the preceding
implementation; it assumes a proportional share of annual runoff occurs during
manure-influenced conditions, rather than explicitly simulating application or
storm dates.

The change is applied only to eligible agricultural area (cropland/pastureland)
and only to **surface** runoff. The model uses the parcel's existing PLET
annual CN runoff depth, pollutant concentration and load multiplier to obtain
kg/ha/year, then adds feedlot-specific runoff. Existing subsurface
concentrations are not increased by manure application.

An AFO occupying a footprint within a host parcel replaces the ordinary PLET
background surface **and subsurface** load over that footprint. Direct feedlot
runoff TN/TP is calculated for the footprint from PLET's CN, pollutant-specific
animal-equivalent density and manure-pack equations. Feedlot TSS is *not*
calculated by those equations; no TSS source is invented. This uses the same
representative-event precipitation inputs as the host parcel. Feedlot CN values
are 91/92/93/94 for paved percentages 0–<25/25–<50/50–<75/75–100.

For each pollutant TN and TP in the current model (the PLET formula also supports BOD,
but this repository currently configures TN, TP, and TSS only), feedlot-equivalent animal units
are `sum(count * factor_{pollutant})`. In PLET, equivalent animal density is
animal units per **feedlot acre**; manure-pack fraction is
`min(equivalent_units / acres, 100) / 100`. At 100% manure-pack coverage, PLET
uses concentrations TN=1500, TP=300, BOD=2000 mg/L. Annual feedlot mass is

`annual_runoff_depth_in * 254000 L/(ha in) * feedlot_area_ha * C_mg_L / 1e6`.

`factor_tn`, `factor_tp` (and, for future BOD support, `factor_bod`) are **pollutant-specific PLET animal
nutrient equivalence factors**, **not** animal live-weight animal units or
manure excretion factors. Supply these from the PLET reference table (or an
explicit, documented local assumption). The example uses an illustrative
slaughter-steer equivalence of 1 for each nutrient; it is not an animal
inventory for East Fork.

## Configuration

```yaml
load_generation:
  mode: plet_rusle
animal_operations: ../afo/afo_example.gpkg
cps: [313, 590]
feedlot_bmp_cps: [313]
```

Relative paths resolve from the YAML file's parent directory. Omitting
`animal_operations` disables the feature and leaves preexisting model
calculations unchanged. `feedlot_bmp_cps` identifies CPS codes that **only**
treat direct feedlot source. All other selected CPS codes act on the regular
agricultural/background load (including the manure concentration adjustment)
and not the direct feedlot source. Provide the relevant pollutant/pathway
removal efficiencies in the usual BMP efficiency CSV. Only place feedlot-only
CPS on operation host parcels. It is the user's responsibility to select
appropriate feedlot efficiencies and CPS definitions; the model does not
infer an animal-manure management pathway from the CPS code alone.

Wetland/buffer CPS 412, 656, 657 cannot be flagged as feedlot-only. Other
legacy BMP designs and selection probabilities remain in force; all BMPs are
scored with the existing full-parcel-cost assumptions, including a feedlot-only
BMP. BMP failure and signed efficiency conventions are unchanged.

## Required GeoPackage attribute tables

The AFO file is a standalone GeoPackage with four relational **attribute**
tables. QGIS can edit them; their PIDs reference the canonical
`parcels.gpkg/parcels.pid` after domain clipping. IDs must be integers.

**`afo_operations`** (one row per AFO):

| Column | Meaning |
|---|---|
| `afo_id` | Unique nonblank text identifier |
| `pid` | Host parcel ID |
| `feedlot_area_ha` | Strictly positive contributing feedlot hectares, <= host area |
| `paved_pct` | 0–100 impervious percentage used to select feedlot CN |
| `allocation` | `explicit` or `sampled` |
| `selection` | For `sampled`, `nearest` or `random` |
| `manure_months` | Number of application months, 0–12 |
| `n_fields_value` | Fixed number of receiving fields for `sampled`; includes eligible host |
| `n_fields_mean`, `n_fields_sd`, `n_fields_min`, `n_fields_max`, `n_fields_p50`, etc. | Alternative inline field-count distribution (same statistics as existing input tables) |

Sampled field count is rounded to the nearest integer (`floor(x + .5)`), and
must lie within `[1, number_of_eligible_parcels]`. Counts exceeding the parcel
universe raise an error instead of silently being clipped. Multiple AFOs may
share a host parcel, but their combined feedlot area cannot exceed it.

**`afo_animals`** (one or more rows per AFO):

| Column | Meaning |
|---|---|
| `afo_id` | Operation reference |
| `species` | Descriptive species or animal class |
| `count` | Nonnegative animal count |
| `factor_tn`, `factor_tp`, `factor_bod` | PLET's pollutant-specific animal equivalence factors; only factors for selected supported pollutants are required |

**`afo_manure_fields`** (only for explicit AFOs):

| Column | Meaning |
|---|---|
| `afo_id` | Operation reference |
| `pid` | User-selected agricultural receiving parcel ID |

One row per AFO/receiving-parcel relation, no duplicate pairs. The host may
be included if it has eligible cropland/pastureland **outside** the feedlot
footprint, but is not automatically required for explicit allocation.

**`afo_manure_concentration_increments`** (one row per land cover and pollutant):

| Column | Meaning |
|---|---|
| `land_cover` | `cropland` or `pastureland` |
| `pollutant` | `TN`, `TP`, optionally `BOD` |
| `value` or `mean`/`sd`/`min`/`max`/percentiles | **Additional** manure-associated runoff concentration, **mg/L**, not an absolute manured concentration |

Provide values for every land-cover/pollutant combination that might receive
manure. Every fixed value and distribution statistic must be finite and
nonnegative, and the sampled increment is constrained to be nonnegative by the
existing distribution sampler. No constraint links the sampled increment to
the sampled background concentration; they are independent and additive.
A value of 0 mg/L is valid and represents no manure-associated increase.
The model no longer reads the former `afo_manured_concentrations` table:
**rename the table and replace its absolute values with increments**, not with
the old absolute concentrations. This is intentionally a breaking input change.
Sources/defaults should be documented separately; example values are NOT
vetted/calibrated literature inputs. The formulation retains PLET's water
volume × concentration load calculation but changes how the manure-associated
concentration is specified.

## Assignment semantics

* **Explicit**: every `afo_manure_fields` relation is used in every
  realization. These eliminate uncertainty in both source and receiving IDs.
* **Sampled, nearest**: use host parcel first *when eligible*, then select
  remaining parcels in ascending geometry-to-geometry distance (ties broken by
  numeric PID). Distance is boundary-to-boundary in the model's projected CRS.
* **Sampled, random**: use eligible host first (if present), then select
  remaining recipients without replacement, uniformly over all eligible
  cropland/pastureland parcels in the domain, independent of HUC12 boundaries.
* Eligibility: whole agricultural parcel area minus any AFO footprint on the
  same parcel. Mixed land-use areas inside a polygon are not explicitly
  modeled; an input parcel's PLET land-cover class determines eligibility.
* Multiple operations applying to the **same parcel** do not add repeated
  manure concentration increments. For a simple PLET-style approximation,
  the largest `manure_months` is used once per receiving parcel. This is a
  modeling assumption, not a nutrient mass balance.
* Assignment is sampled **once per realization** before BMP application. It
  is unchanged throughout that realization's BMP trajectory. BMPs never
  automatically change allocation.
* Selecting more recipient fields can increase total manure-related
  pollutant loads, because the original PLET concentration model **does not
  dilute a conserved manure nutrient mass over receiving acres**. This is
  intentional and should be reported as an uncertainty in application extent,
  rather than as a simulation of redistribution of a fixed tonnage.

## Outputs and validation

Alongside the existing parcel, BMP, scenario and load-parameter Parquet files,
AFO runs write:

`afo_manure_assignments/s1.parquet`, `s2.parquet`, ...

`afo_load_components/s1.parquet`, `s2.parquet`, ...

The source component table reports baseline and final direct feedlot and
manure-associated **surface/subsurface** kg/year by parcel and pollutant;
manure-associated subsurface contributions remain zero in this version.

Each record includes `scenario`, `afo_id`, `host_pid`, `receiving_pid`,
`manure_months`, `eligible_area_ha`, and `allocation`. Paired baseline/final
loads appear in existing parcel and load-parameter outputs. Surface
`initial_concentration` includes the nonnegative manure increment; `initial_subsurface`
loads do not include any manure-specific addition. Source-specific feedlot
loads are kept separately during BMP calculations to prevent a feedlot-only
BMP from treating neighboring fields and vice versa.

Input failures are explicit for unknown PIDs, nonagricultural receiving
parcels, duplicate relations, exceeding feedlot footprints, missing animal
factors, inappropriate concentration coverage, and impossible field counts.

## Repeatable demonstration

`examples/east_fork/inputs/config/east_fork_afo_demo.yaml` references the
illustrative AFO GeoPackage and a small example efficiency table. Run:

```bash
python run_model.py examples/east_fork/inputs/config/east_fork_afo_demo.yaml
```

The demonstration assigns AFOs to example East Fork parcel IDs *solely to
exercise the software*. It is **not** evidence those locations contain feedlots
or receive manure, and its manure concentration increments have not been
calibrated or validated. Replace them with vetted project data before interpreting results.

## Source-method references

- EPA. *Pollutant Load Estimation Tool User's Guide*, v2.0 (2024), Appendix C,
  cropland/pasture and feedlot runoff sections:
  https://www.epa.gov/system/files/documents/2024-10/user-guide-oct-24-update_final_508-full.pdf
- EPA. *BMP Efficiency References for PLET* (2023), Table 3:
  https://ordspub.epa.gov/ords/grts/r/grtsadm/files/static/v287/BMP%20Efficiency%20References%20Doc%20082023.pdf
