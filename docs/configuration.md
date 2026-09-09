# Configuration reference

[← Back to main README](../readme.md)

## Configuration structure

A YAML configuration combines common scenario settings with one of two load-generation configurations:

- default statistical mode
- `plet_rusle` mode

If `load_generation.mode` is omitted, the model uses statistical mode.

## Common required configuration

Typical common requirements are:

- `domain` — watershed boundary file
- `parcels` — parcel polygon file
- `outlet_loc` — outlet location file
- `parcel_out` — CSV mapping parcels to outlet IDs
- `pollutants` — modeled pollutant labels
- `cps` — BMP or conservation-practice CPS codes
- `bmp_efficiency` — BMP efficiency values and distributions
- `n_scenarios` — number of Monte Carlo scenarios
- at least one of `bmp_limit_n` or `bmp_limit_usd`

Common optional settings include:

- `input_distributions` — reusable named numeric distributions referenced from other input tables
- `parcel_up` — upstream parcel relationships
- `parcel_p` — parcel-selection probability weights
- `bmp_cost` — BMP cost values and distributions
- `delivery_ratios` — parcel-to-outlet delivery ratios
- `outlet_target` — outlet pollutant reduction targets
- `outlet_mean` — outlet mean-load reference values
- `buffer_depth_ft` — buffer-depth assumption used by applicable grassed BMP calculations
- `bmp_sel_prob_via_costs` — allow cost to influence BMP-selection probabilities
- `bmp_fail_rate` — probability that a BMP placement fails
- `bmp_fail_reduction` — efficiency multiplier applied after failure
- `random_seed` — base random seed
- `outputs` — output directory
- `verbose` — verbose logging
- `parallel` — parallel execution settings

Pollutant aliases such as `nitrogen`, `phosphorus`, and `sediment` are normalized to canonical pollutant labels where supported.

## Standard numeric input schema

Numeric input tables use a common fixed-value and distribution convention. New files should use canonical columns:

    value, distribution_id, mean, sd, min, p05, p50, p95, max

Only the columns needed for a given row must contain values. Accepted forms and precedence rules are described in [Standardized numeric inputs and distributions](input_distributions.md).

A top-level reusable catalog is optional:

    input_distributions: ./inputs/input_distributions.csv

Example catalog:

    distribution_id,value,mean,sd,min,p05,p50,p95,max,units,notes
    annual_precip_default,,42,3,34,,,,50,in/year,Example bounded normal
    runoff_tn_default,3,,,,,,,,mg/L,Fixed value

A row in another table can then reference `distribution_id` rather than repeat the statistics. A catalog reference reuses the distribution definition; it does not automatically share the same random draw among parcels.

## Statistical mode configuration

Statistical mode can be explicit:

    load_generation:
      mode: statistical

or `load_generation` can be omitted.

A basic configuration is:

    verbose: true
    outputs: ./outputs
    random_seed: 42

    domain: ./inputs/domain.gpkg
    parcels: ./inputs/parcels.gpkg
    outlet_loc: ./inputs/outlet_loc.gpkg
    parcel_out: ./inputs/parcel_out.csv
    parcel_up: ./inputs/parcel_up.csv
    parcel_p: ./inputs/parcel_p.csv

    pollutants: [TN, TP, TSS]
    cps: [340, 329, 590, 412, 656]

    input_distributions: ./inputs/input_distributions.csv

    pollutant_load_rate: ./inputs/pollutant_load_rate.csv
    bmp_efficiency: ./inputs/bmp_efficiency.csv
    bmp_cost: ./inputs/bmp_cost.csv

    outlet_target: ./inputs/outlet_target.csv
    outlet_mean: ./inputs/outlet_mean.csv

    n_scenarios: 1000
    bmp_limit_n: 200

    bmp_fail_rate: 0.25
    bmp_fail_reduction: 0.25

### Large parcel datasets

`pollutant_load_rate.csv` may use `pid="*"` as a default distribution for all parcels. Exact parcel rows override the wildcard row for the same pollutant and pathway.

    pid,pollutant,pathway,value,distribution_id,mean,sd,min,max,units
    *,TN,surface,,tn_surface_default,,,,,kg/ha/yr
    P104,TN,surface,,,9.2,1.4,6,13,kg/ha/yr

This is useful when thousands of parcels share an assumption. When parcel distributions are genuinely unique, supply one row per parcel × pollutant × pathway, or one aggregate row per parcel × pollutant.

### Statistical mode with aggregate parcel inputs

If `pollutant_load_rate.csv` contains one aggregate parcel pollutant quantity per parcel × pollutant while `bmp_efficiency.csv` defines multiple pathways, define the split explicitly:

    pollutant_load_rate_pathway_fractions:
      surface: 0.70
      shallow subsurface: 0.20
      tile: 0.10

The fractions must correspond to active BMP-efficiency pathways and sum to 1.0.

## `plet_rusle` configuration

A typical configuration is:

    verbose: true
    outputs: ./outputs
    random_seed: 42

    domain: ./inputs/plet/domain.gpkg
    parcels: ./inputs/plet/parcels.gpkg
    outlet_loc: ./inputs/plet/outlet_loc.gpkg
    parcel_out: ./inputs/plet/parcel_out.csv
    parcel_up: ./inputs/plet/parcel_up.csv
    parcel_p: ./inputs/plet/parcel_p.csv

    pollutants: [TN, TP, TSS]
    cps: [340, 329, 590, 412, 656]

    input_distributions: ./inputs/plet/input_distributions.csv

    bmp_efficiency: ./inputs/plet/bmp_efficiency.csv
    bmp_cost: ./inputs/plet/bmp_cost.csv

    load_generation:
      mode: plet_rusle
      plet_inputs: ./inputs/plet/plet_inputs.csv
      hydrology_lookup: ./inputs/plet/plet_hydrology_lookup.csv
      rusle_inputs: ./inputs/plet/rusle_inputs.csv
      pollutant_concentrations: ./inputs/plet/pollutant_concentrations.csv
      groundwater_concentrations: ./inputs/plet/groundwater_concentrations.csv

    n_scenarios: 1000
    bmp_limit_n: 200

### Mode-specific rules

In `plet_rusle` mode:

- statistical-mode parcel load-rate input is not used
- production pathways are fixed to `surface` and `subsurface`
- `plet_inputs` is required
- `hydrology_lookup` is required
- every supported land-cover × HSG pairing in `hydrology_lookup` must define both `cn` and `infiltration_fraction` as either a fixed value or a valid distribution
- parcel `plet_inputs` must supply fixed `land_cover` and `hsg` classifications and may not supply `cn` or `infiltration_fraction` directly
- `pollutant_concentrations` is required when TN or TP is modeled
- `groundwater_concentrations` is required for every modeled non-TSS pollutant
- `rusle_inputs` is optional, but a parcel that supplies RUSLE inputs must supply a complete RUSLE factor set and may supply `sdr` to override the default delivery ratio of 1.0
- `pathway_mode` and `watershed_area_mi2` have been removed and are an error if supplied
- statistical pathway-fraction settings do not control PLET/RUSLE pathways

Legacy `groundwater_loads` and `treat_groundwater_with_bmps` settings do not control production PLET/RUSLE pathway generation. The production calculation always estimates the subsurface load from groundwater concentration and the sampled infiltration fraction. BMP treatment is controlled by the `subsurface` BMP efficiency.

## Scenario stopping conditions

The model supports a BMP-count limit, a cost limit, or both.

A scenario stops before adding another BMP as soon as **either** configured stopping condition has been met. In other words:

- if only `bmp_limit_n` is configured, no additional BMPs are added after the BMP-count limit is reached
- if only `bmp_limit_usd` is configured, no additional BMPs are added after the cost limit is reached
- if both are configured, no additional BMPs are added after **either** the BMP-count limit or the cost limit has been reached

This is an **OR** stopping rule, not an **AND** rule.

The stopping check occurs at the start of each BMP-placement iteration. A BMP that causes cumulative cost or BMP count to reach or exceed its configured limit is retained, and the scenario stops before another BMP is added.

## Parallel configuration

    parallel:
      n_jobs: -1
      max_nbytes: "1M"
      temp_folder: "/tmp/bmp-loky"

- `n_jobs` controls worker processes; `-1` uses all available CPUs
- `max_nbytes` controls the memmap threshold for objects passed to workers
- `temp_folder` optionally sets the `loky` temporary directory