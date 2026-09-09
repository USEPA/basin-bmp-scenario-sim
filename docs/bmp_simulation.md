# BMP simulation

[← Back to main README](../readme.md)

## Shared scenario engine

After baseline parcel/pathway pollutant quantities are established, both load-generation modes use the same BMP scenario engine.

A scenario repeatedly:

1. selects a parcel according to configured parcel-selection probabilities
2. selects an eligible BMP according to configured BMP-selection probabilities
3. samples pollutant- and pathway-specific BMP efficiencies
4. samples BMP-specific characteristics where applicable
5. optionally simulates BMP failure
6. applies BMP effects to the current pathway `load_rates`
7. calculates BMP cost when cost inputs are configured
8. updates parcel and outlet results
9. continues until the configured stopping condition is reached

## Pathway-specific efficiency

For current pathway quantity `L`, treated fraction `f_t`, and sampled BMP efficiency `e`:

    L_new = L_old × (1 - f_t × e)

The removed amount is the difference between the previous and updated load. Because efficiencies are signed, a negative efficiency can increase the resulting pathway load.

### Statistical mode

Every active pathway requires an explicitly defined BMP efficiency for every configured CPS × pollutant combination.

### `plet_rusle` mode

A correctly labeled `surface` efficiency is required. A missing or unusable correctly labeled `subsurface` efficiency defaults to zero and is logged.

## Treatment fraction

The fraction of load affected by a BMP depends on the BMP type and its sampled or configured geometry or contributing area. An in-field practice may affect a different fraction than a grassed waterway or wetland. The model therefore applies efficiency to the BMP-relevant treated fraction rather than assuming every BMP treats every parcel identically.

## BMP failure

BMP failure is optional. Example:

    bmp_fail_rate: 0.25
    bmp_fail_reduction: 0.25

In this example, each placement has a 25% probability of failure, and a failed placement retains 25% of the sampled BMP efficiency.

The same failure multiplier applies across the sampled pathway efficiencies for that placement.

## Serial stacking

BMP effects are applied serially to the current remaining load. For two fully applied 50% reductions:

    100 → 50 → 25

The total reduction is 75%, not 100%.

This is an important structural assumption. It is most natural where later practices act on pollutant mass remaining after earlier practices. If efficiency estimates were developed under a different interaction assumption, users should account for that when designing input distributions or interpreting results.

## Parcel connectivity and wetlands

Where applicable, `parcel_up` supplies upstream parcel relationships used by BMP logic that depends on contributing drainage area. Cells containing multiple upstream parcel IDs are interpreted as multiple relationships rather than one compound ID.

## Cost and stopping conditions

BMP costs may be sampled from `bmp_cost` when configured. Cost can also optionally influence BMP-selection probabilities when `bmp_sel_prob_via_costs` is enabled.

Scenarios may be limited by BMP count, cost, or both. No additional BMPs are added once a configured stopping condition has been met. If both `bmp_limit_n` and `bmp_limit_usd` are configured, the scenario stops before another BMP is added as soon as **either** limit has been reached.

The stopping check occurs at the start of each BMP-placement iteration. This means a BMP that causes cumulative cost or BMP count to reach or exceed its configured limit is included in the scenario, and the run then stops before the next BMP would be added.

## Interpretation

BMP removal estimates are conditional on:

- the baseline parcel/pathway pollutant quantities
- the sampled efficiency distribution
- the fraction of the relevant load treated
- BMP failure assumptions
- BMP order
- parcel and outlet routing assumptions

The model should therefore be viewed as a probabilistic scenario framework rather than a mechanistic simulation of BMP biogeochemistry or hydraulic performance through time.