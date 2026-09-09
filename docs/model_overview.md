# Model concepts and scientific formulation

[← Back to main README](../readme.md)

## Purpose

`basin-bmp-scenario-sim` is a Monte Carlo framework for connecting uncertainty in parcel-scale pollutant generation and BMP implementation to basin-outlet pollutant outcomes. It operates on annualized pollutant loads and parcel/pathway `load_rates` rather than simulating a continuous hydrograph or water-quality time series.

## Core quantities

The model distinguishes several related quantities.

### Concentration

Concentration is pollutant mass per unit water volume, for example `mg/L`. Concentrations are used explicitly by `plet_rusle` to calculate runoff-derived and infiltration-derived nutrient loads.

### Load

Load is pollutant mass over a modeled period. Parcel and outlet outcomes are tracked and interpreted as loads.

### `load_rates`

The code uses `load_rates` for the current annualized parcel/pathway pollutant quantities being updated as BMPs are applied. These are the evolving parcel/pathway quantities that later determine parcel and outlet loads.

### Pathway

A pathway partitions a parcel's pollutant quantity into components that may respond differently to BMPs.

In **statistical mode**, pathway labels are user-defined bookkeeping categories and may represent concepts such as surface runoff, shallow subsurface flow, tile drainage, groundwater, or another user-defined transport category.

In **`plet_rusle` mode**, pathway meanings are fixed:

- `surface` — runoff-derived load plus sediment-associated pollutant load where applicable
- `subsurface` — infiltration/groundwater-derived non-TSS load

## Monte Carlo representation

Uncertain quantities may be represented using fixed values or statistical information supported by the input tables. Across scenarios, the model samples uncertain values and thereby produces a distribution of implementation outcomes rather than one deterministic answer.

Uncertainty may include:

- parcel selection
- BMP type
- parcel pollutant load-rate inputs or physical load-generation inputs
- BMP pollutant-removal efficiencies
- BMP-specific treatment characteristics
- cost
- BMP failure

The resulting distribution of outlet loads or target attainment can be interpreted as conditional on those user-defined uncertainty distributions and the model's structural assumptions.

## Scenario sequence

For each scenario, the model first establishes baseline parcel/pathway pollutant quantities for selected parcels and pollutants. It then repeatedly selects BMP placements and updates the current pathway `load_rates`. Later BMPs operate on quantities remaining after earlier BMPs.

Conceptually, for current pathway quantity `L`, treated fraction `f`, and sampled BMP efficiency `e`:

    L_new = L_old × (1 - f × e)

If `e` is negative, the same equation represents an increase in load rather than removal.

## Parcel-to-outlet evaluation

`parcel_out` associates parcels with modeled outlets. Optional delivery ratios can attenuate parcel loads before outlet evaluation. Outlet target and mean-load inputs allow scenario trajectories to be expressed relative to decision-relevant reference values.

The load-generation mode determines how baseline parcel loads are established. It does not change the overall parcel-selection, BMP-selection, routing, and outlet-evaluation framework.

## What the model does not represent directly

Unless supplied indirectly through input distributions or delivery factors, the model does not explicitly simulate:

- sub-daily or daily hydrology
- channel hydraulics
- in-stream nutrient transformation
- groundwater travel times
- mechanistic BMP biogeochemistry or hydraulic performance through time
- all sediment storage and remobilization processes between parcel and outlet

The model should therefore be viewed as a probabilistic scenario framework rather than a full process-based watershed simulator.

## Two load-generation modes

### Statistical mode

In statistical mode, the user supplies parcel pollutant load-rate inputs and defines the pathways. The model does not directly compute runoff, infiltration, erosion, or groundwater transport in this mode.

### `plet_rusle` mode

In `plet_rusle`, the model calculates baseline parcel loads from PLET-style runoff and infiltration calculations, pollutant concentrations, and optional RUSLE sediment generation. The pathway meanings are fixed by the hydrologic formulation.

## Interpretation

Model results are conditional statements of the form:

> Given these baseline-load assumptions, BMP-effect distributions, placement rules, costs, failure assumptions, pathway definitions, and routing assumptions, this is the simulated distribution of basin outcomes.

That is different from claiming that the output distribution captures every source of real-world uncertainty.