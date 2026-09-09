# BASIN-BMP-SCENARIO-SIMulator

`basin-bmp-scenario-sim` is a probabilistic, basin-scale best management practice (BMP) scenario simulator for evaluating how uncertainty in pollutant generation, BMP placement, BMP effectiveness, BMP cost, and BMP failure affects basin and outlet pollutant-load outcomes.

The model is a Monte Carlo scenario framework for annualized parcel and outlet **loads**. In code and some internal data structures, the evolving parcel-scale quantities updated during BMP application are referred to as **`load_rates`**. In this README, **loads** are treated as the primary scientific and output quantities, while **`load_rates`** refers to the current annualized parcel/pathway pollutant quantities tracked through the scenario engine.

The simulator is **not** a continuous hydrologic or water-quality model. It represents annual pollutant generation and BMP effects at the parcel scale and routes those loads to configured outlets.

## Scientific purpose

The model is intended for watershed planning and uncertainty analysis. It uses Monte Carlo simulation to generate many plausible basin-wide BMP implementation scenarios, propagate uncertain inputs through each scenario, and evaluate the resulting parcel- and outlet-scale pollutant loads, costs, and target attainment.

The model is designed to answer questions such as:

- How variable are expected pollutant reductions across plausible BMP portfolios?
- How often does a given implementation strategy meet an outlet load-reduction target?
- How do uncertainty in baseline pollutant generation and BMP performance affect predicted outcomes?
- How do implementation limits such as BMP count or cost change the distribution of outcomes?

The simulator is annualized rather than continuous. It does not directly simulate sub-daily hydrology, channel hydraulics, in-stream nutrient transformation, groundwater travel times, sediment storage or remobilization within the channel network, or mechanistic BMP performance through time. Those processes must be represented indirectly through input assumptions, routing factors, or complementary models where needed.

## Two load-generation modes

The model provides two alternative ways to establish baseline parcel pollutant loads.

| Feature | Default statistical mode | `plet_rusle` mode |
|---|---|---|
| Baseline pollutant generation | User supplies parcel pollutant load-rate values/distributions | Model calculates loads from PLET-style hydrology, concentrations, and optional RUSLE |
| Statistical parcel input | `pollutant_load_rate` | Not used |
| Pollutant pathways | User-defined | Fixed to `surface` and `subsurface` |
| Runoff and infiltration modeled | No | Yes |
| Curve Number | Not used | Required user input by land-cover × HSG pairing |
| Infiltration fraction | Not used | Required user input by land-cover × HSG pairing |
| Sediment/TSS | User-supplied parcel pollutant input | RUSLE when complete RUSLE inputs are supplied, otherwise concentration-based TSS may be used |
| BMP efficiency coverage | Required for every active pathway | `surface` required; missing correctly labeled `subsurface` defaults to 0 with logging |

The default statistical mode is appropriate when parcel pollutant quantities are already available from monitoring, another model, literature, calibration, or expert judgment. The `plet_rusle` mode is appropriate when baseline loads should be generated internally from PLET-style runoff and infiltration calculations and, optionally, RUSLE sediment generation.

## Core terminology

### Concentration

Concentration is pollutant mass per unit water volume, for example `mg/L`. Concentrations are used explicitly by `plet_rusle` to calculate runoff-derived and infiltration-derived nutrient loads.

### Load

A load is pollutant mass over the modeled annual period. Parcel and outlet outcomes are reported as loads. BMP application updates parcel/pathway quantities over time, and later BMPs operate on the remaining load after earlier BMPs.

### `load_rates`

Some code paths and internal representations use `load_rates` for the current annualized parcel/pathway pollutant quantities being updated during scenario execution. In user-facing scientific terms, these are the evolving parcel/pathway pollutant quantities that drive parcel and outlet load accounting.

### Pathway

A pathway partitions a parcel pollutant quantity into components that may respond differently to BMPs.

In **statistical mode**, pathway labels are user-defined bookkeeping categories and may represent concepts such as surface runoff, shallow subsurface flow, tile drainage, groundwater, or another user-defined transport category.

In **`plet_rusle` mode**, pathway meanings are fixed:

- `surface` — runoff-derived load plus sediment-associated pollutant load where applicable
- `subsurface` — infiltration/groundwater-derived non-TSS load

Earlier shallow and deep subsurface splits are **not** part of the production `plet_rusle` pathway representation. Compatibility diagnostic fields may still appear in some outputs for older callers or tests and should not be interpreted as additional production pathways.

## Standardized uncertain-input format

Numeric inputs use a common value/distribution convention. Depending on the table, an input may be specified as:

- a fixed `value`
- `mean` + `sd`, optionally with `min`/`max` bounds
- the legacy `min` + `mean` + `max` form
- `min` + `max` for a uniform distribution
- `min` + percentile columns such as `p05`, `p50`, `p95` + `max`
- a reusable `distribution_id` defined once in an optional `input_distributions.csv` catalog

Parcel-indexed inputs may use `pid: "*"` as a default and provide only parcel-specific overrides. This can greatly reduce duplication when many parcels share assumptions.

## Quick start

A model run requires common spatial and routing inputs, pollutant and BMP definitions, a BMP-efficiency table, a load-generation configuration, and at least one scenario stopping condition.

Example command:

    python run_model.py config.yaml

Statistical mode is the default:

    pollutants: [TN, TP, TSS]
    cps: [340, 329, 590]

    input_distributions: ./inputs/input_distributions.csv

    pollutant_load_rate: ./inputs/pollutant_load_rate.csv
    bmp_efficiency: ./inputs/bmp_efficiency.csv

    n_scenarios: 1000
    bmp_limit_n: 200

PLET/RUSLE mode is enabled explicitly and requires a hydrology table:

    input_distributions: ./inputs/plet/input_distributions.csv

    load_generation:
      mode: plet_rusle
      plet_inputs: ./inputs/plet/plet_inputs.csv
      hydrology_lookup: ./inputs/plet/plet_hydrology_lookup.csv
      rusle_inputs: ./inputs/plet/rusle_inputs.csv
      pollutant_concentrations: ./inputs/plet/pollutant_concentrations.csv
      groundwater_concentrations: ./inputs/plet/groundwater_concentrations.csv

See the docs below for complete examples and mode-specific requirements.

## Documentation

- [Model concepts and scientific formulation](docs/model_overview.md) — loads, pathways, Monte Carlo structure, and outlet routing
- [Configuration reference](docs/configuration.md) — common YAML settings and complete examples for both modes
- [Standardized numeric inputs and distributions](docs/input_distributions.md) — fixed values, distributions, reusable distribution IDs, parcel defaults and overrides, and shared draws
- [Statistical load-generation mode](docs/statistical_mode.md) — direct parcel load-rate inputs, arbitrary pathways, aggregate pathway splitting, and coverage rules
- [PLET/RUSLE load-generation mode](docs/plet_rusle_mode.md) — required land-cover/HSG hydrology inputs, runoff, infiltration, groundwater loads, RUSLE, and two-pathway BMP treatment
- [BMP simulation](docs/bmp_simulation.md) — BMP selection, efficiencies, treatment fractions, failure, signed effects, and serial stacking
- [Input and output reference](docs/input_output_reference.md) — input file roles, output files, and interpretation
- [Reproducibility, testing, and limitations](docs/reproducibility_validation.md) — seeds, parallel execution, validation expectations, assumptions, and appropriate interpretation

## Required inputs at a glance

Inputs shared by both modes generally include:

- watershed/domain geometry
- parcel polygons
- outlet locations
- parcel-to-outlet mapping
- modeled pollutants
- configured BMP or conservation-practice CPS codes
- BMP-efficiency statistics
- number of Monte Carlo scenarios
- a BMP-count limit, cost limit, or both

Load-generation-specific inputs differ substantially between the two modes. See [Configuration reference](docs/configuration.md).

## Outputs

The model writes scenario-level and aggregated results to the configured `outputs` directory. Current canonical outputs include per-BMP records, per-parcel results, scenario metrics, outlet trajectories, logs, plots, and, when `plet_rusle` is used, load-generation diagnostic records.

See [Input and output reference](docs/input_output_reference.md) for details.

## Reproducibility

Set `random_seed` in the YAML configuration or use the CLI `--seed` option. The base seed is used to spawn scenario-specific child seeds so a run can be reproduced when the same code, inputs, configuration, and seed are used.

For reproducible scientific analyses, archive or record:

- the repository commit or release
- the configuration YAML
- all input data files, including any distribution catalog and PLET hydrology lookup
- the random seed
- the Python environment and dependency versions
- the generated logs
- the canonical outputs

## Scientific scope and limitations

The model is a scenario and uncertainty framework, not a substitute for a calibrated process-based watershed model where detailed temporal hydrology, water-quality transformation, or in-stream processes are required. Results depend on the validity of the supplied probability distributions, pathway definitions, BMP efficiencies, routing assumptions, and, in `plet_rusle` mode, the PLET/RUSLE parameterization, including the user-supplied Curve Number and infiltration-fraction assumptions.

BMP efficiencies are applied serially to the current remaining load. Parcel-to-outlet routing may use optional delivery ratios, but the simulator does not independently resolve all physical fate and transport processes between a parcel and an outlet.

When both BMP-count and cost limits are configured, the scenario uses an OR stopping rule: no additional BMPs are added once either limit has been reached or exceeded.

Model outputs should therefore be interpreted as conditional on the selected model structure and input assumptions.

## Citation

When using the model in scientific or technical work, identify the repository and the exact version, release, or Git commit used for the analysis.