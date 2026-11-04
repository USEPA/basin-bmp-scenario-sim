# Statistical load-generation mode

[← Back to main README](../readme.md)

Statistical mode uses parcel load-rate distributions directly rather than deriving loads from PLET/RUSLE hydrology.

## Configuration

```yaml
domain: ../domain/domain.gpkg
parcels: ../parcels/parcels.gpkg
outlets: ../outlets/outlets.gpkg

pollutants: [TN, TP, TSS]
cps: [340, 329, 590]

bmp_efficiency: ../bmps/bmp_efficiency.csv

load_generation:
  mode: statistical
```

## `input_pollutant_load_rate`

The statistical load-rate variable is stored in `input_pollutant_load_rate` in the parcel, HUC, or domain GeoPackage. Parcel-specific values override assigned-HUC values, then domain values, then NULL-pid parcel defaults.

To select HUC inputs, add `huc_parameters: ../hucs/hucs_statistical.gpkg` to the configuration. A HUC table replaces `pid` with text `huc_id`; a domain table omits `pid`. Both retain `pollutant`, optional `pathway`, and the same value/distribution columns. For example, a domain-wide fixed rate is:

```text
pollutant | value | units
TN        | 8.0   | kg/ha/year
```

Required rates may come entirely from the domain, so the parcel load-rate table is optional when inherited inputs provide complete coverage. Do not mix aggregate and pathway-specific rates across scopes in one run. Inherited distributions share a draw across the parcels inheriting each source row, separately by pollutant/pathway and scenario. Parcel-default distributions remain independent unless `sample_group` is supplied. See the [scope schemas](input_output_reference.md#huc-and-domain-parameter-geopackages).

The natural key is parcel × pollutant, optionally extended by pathway.

Example aggregate rows:

```text
pid | pollutant | value / distribution statistics | units
NULL| TN        | yield_tn        | kg/ha/year
NULL| TP        | yield_tp        | kg/ha/year
```

Example pathway-aware rows:

```text
pid | pollutant | pathway            | mean | sd  | units
NULL| TN        | surface            | 8.0  | 1.5 | kg/ha/year
NULL| TN        | shallow subsurface | 2.0  | 0.5 | kg/ha/year
```

`pid IS NULL` provides defaults; exact integer parcel rows override matching default rows. The legacy string `"*"` is not a valid PID.

## Aggregate inputs and pathway fractions

If load rates are aggregate but BMP efficiencies distinguish multiple pathways, define the split in YAML:

```yaml
pollutant_load_rate_pathway_fractions:
  surface: 0.70
  shallow subsurface: 0.20
  tile: 0.10
```

Fractions must match active efficiency pathways and sum to `1.0`.

## Parcel selection

Optional parcel-selection weights are stored in `input_selection_weight`, not on the spatial `parcels` layer. The table is keyed by `pid` and uses deterministic `value` entries. If the table is absent, parcel selection is uniform.

## Optional AFO manure mass and feedlot runoff

`load_generation.mode: statistical` supports the **mass-based** AFO module
with independent animal/manure production, parcel allocation, annual runoff/
precipitation and separate direct-feedlot surface loading. An explicit
statistical `surface_pathway` and a baseline excluding the modeled manure/
feedlot loads are required. See [Mass-based AFO loading](afo_mass_based.md)
for inputs, hierarchy and equations. The legacy `concentration_increment`
AFO method is PLET-only.
