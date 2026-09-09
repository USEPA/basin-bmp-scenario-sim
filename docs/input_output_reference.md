# Input and output reference

[← Back to main README](../readme.md)

## Common numeric value and distribution schema

Numeric model inputs use a common row-level schema. Depending on the table, identifier columns come first, followed by any of these distribution columns:

    value, distribution_id, mean, sd, min, p05, p50, p95, max

Percentile columns may use other `pXX` levels. Accepted forms and validation rules are documented in [Standardized numeric inputs and distributions](input_distributions.md).

The optional top-level `input_distributions` CSV stores reusable named distributions. A use-site row can reference one with `distribution_id` instead of repeating its statistics.

## Common input files

### `domain`

Watershed or domain geometry used to define the modeled spatial extent.

### `parcels`

Parcel or field polygons. Parcel IDs must be unique after any domain clipping performed by the model.

### `outlet_loc`

Modeled outlet locations.

### `parcel_out`

Maps parcel IDs to one or more outlet IDs. Referenced outlet IDs must exist in `outlet_loc`.

### `parcel_up`

Optional upstream parcel connectivity. This is used by BMP calculations that require contributing-area relationships.

### `parcel_p`

Optional parcel-selection probability information. Probabilities must be valid for the parcels used by the scenario engine.

### `input_distributions`

Optional reusable numeric distribution catalog configured at the top level:

    input_distributions: ./inputs/input_distributions.csv

Key column: `distribution_id`. Every catalog row must contain one valid fixed-value or distribution specification.

### `bmp_efficiency`

BMP efficiency values and distributions by CPS, pollutant, and, when pathway-aware, pathway.

Statistical mode requires complete coverage for every active pathway. `plet_rusle` requires surface coverage and defaults missing correctly labeled subsurface efficiency to zero with logging.

### `bmp_cost`

Optional BMP cost values and distributions used for cost accounting and, when configured, BMP-selection weighting. The existing `unit` column remains required because cost scaling depends on the cost unit.

### `delivery_ratios`

Optional parcel-to-outlet delivery ratios used to attenuate loads before outlet evaluation.

### `outlet_target`

Optional outlet pollutant reduction targets.

### `outlet_mean`

Optional outlet mean-load reference values.

## Statistical-mode load inputs

### `pollutant_load_rate`

Required in statistical mode. It may contain:

- explicit parcel × pollutant × pathway values or distributions
- one aggregate parcel × pollutant value or distribution that is subsequently split with configured pathway fractions

`pid="*"` may define a default for all parcels, with exact parcel rows overriding the default for the same pollutant and pathway.

See [Statistical load-generation mode](statistical_mode.md).

## `plet_rusle` load inputs

### `load_generation.plet_inputs`

Required long-form PLET parcel parameter table. It supplies climate variables and fixed `land_cover` and `hsg` classifications. Numeric rows use the standardized fixed-value and distribution schema and may use `pid="*"` defaults with parcel-specific overrides.

`cn` and `infiltration_fraction` are **not** supplied in this table.

### `load_generation.hydrology_lookup`

Required long-form land-cover/HSG hydrology input table. It contains exactly one `cn` and one `infiltration_fraction` row for every supported land-cover × HSG pairing.

Both parameters may be supplied as fixed values or distributions. This file replaces the old source-code data file `src/data/plet_hydrology_lookup.csv`.

### `load_generation.rusle_inputs`

Optional long-form RUSLE parameter table. Numeric rows use the common distribution schema and may use `pid="*"` defaults. A parcel with RUSLE data must have a complete factor set and may supply `sdr` to override the default sediment delivery ratio of 1.0.

### `load_generation.pollutant_concentrations`

Runoff concentration values and distributions. Required when TN or TP is modeled. TSS concentration is also needed when RUSLE is not available for a parcel and TSS is modeled. `pid="*"` defaults and parcel-specific overrides are supported.

### `load_generation.groundwater_concentrations`

Required for each modeled non-TSS pollutant. Used with PLET infiltration volume to calculate the `subsurface` pathway. `pid="*"` defaults and parcel-specific overrides are supported.

## Recommended PLET input layout

    inputs/
      plet/
        input_distributions.csv
        plet_inputs.csv
        plet_hydrology_lookup.csv
        rusle_inputs.csv
        pollutant_concentrations.csv
        groundwater_concentrations.csv
        bmp_efficiency.csv
        bmp_cost.csv
        ...spatial and routing inputs...

The model keeps conceptually different variable families in separate files while using the same distribution columns for all numeric quantities.

## Output directory

Current canonical outputs are written below the configured `outputs` directory.

### Per-BMP records

    bmps/s{scenario}.parquet

Contains individual BMP placement records and associated pollutant-treatment and removal information.

### Per-parcel records

    parcels/s{scenario}.parquet

Contains parcel baseline and final pollutant information for each scenario.

### PLET/RUSLE load diagnostics

    load_parameters/s{scenario}.parquet

Written when PLET/RUSLE load-generation diagnostics are available. The production PLET pathways are `surface` and `subsurface`; compatibility diagnostic fields may also be retained for older callers or tests and should not be confused with additional production pathways.

### Scenario metrics

    scenario_metrics/s{scenario}.parquet

Canonical per-scenario metrics.

### Outlet trajectories

    outlet_trajectories/all_scenarios.parquet

Aggregated outlet trajectory data used for downstream plotting and scenario comparison.

### Logs

    log.txt
    logs/s{scenario}.txt

The driver writes the overall log, while scenario workers can write scenario-specific logs. PLET/RUSLE logs also report cases where subsurface BMP efficiency defaults to zero or unexpected pathway labels are ignored.

### Plots

Summary `plot_*` outputs visualize scenario trajectories such as implementation cost or BMP count versus outlet load or target metrics.

## Interpreting outputs

Each scenario is one possible implementation realization. Scientific interpretation should focus on the **distribution** of outcomes across scenarios.

Useful summaries include:

- median and percentile pollutant reductions
- probability of meeting a target
- cost distributions
- BMP portfolio composition
- sensitivity to baseline-load assumptions
- differences between configuration alternatives