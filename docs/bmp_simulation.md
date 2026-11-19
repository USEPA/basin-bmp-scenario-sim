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

With `deterministic_bmp_design_parameters: true`, wetlands occupy 20% of the
selected parcel area and have a 4:1 catchment-to-wetland area ratio, treating
that parcel without adding upstream parcels. Grassed waterways span 25% of
parcel perimeter and treat 25% of parcel area. The default (`false`) retains
sampled design parameters; see [configuration](configuration.md).

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

The required `parcel_down` table stores one immediate *downstream* recipient
per source parcel as `(pid, pid_down)`. Its `pid` key is unique; an outlet or
unresolved parcel has no row, and many upstream parcels may share one
recipient. `parcel_down` is acyclic, and it cannot represent split routing.
The model reverses these links in memory to build immediate-upstream
adjacency, then traverses all reachable ancestors in shortest-distance order.
Legacy `parcel_up` input is not accepted. The graph is a whole-parcel routing
approximation, not a fraction-aware catchment model.

For sampled wetland designs, the simulation traverses the **full transitive
upstream network** from the wetland host, starting with the host itself and
adding upstream parcels by shortest graph distance (number of immediate
connections). Parcels tied in distance are considered in ascending PID order
(numerically for numeric IDs and lexicographically otherwise). Each upstream
parcel can contribute only once, even if reachable by multiple routes.
Traversal stops as soon as the wetland's sampled treatment area is covered or
the upstream network is exhausted; partial treatment of the last selected
parcel is preserved. Parcels are considered from the complete hydrologic
universe, including those excluded from random BMP placement.

The existing deterministic-design assumption is intentionally unchanged:
`deterministic_bmp_design_parameters: true` treats only the wetland-host parcel.
No wetland efficiency, delivery, cost, or cumulative-cost stopping assumptions
are changed by the graph-traversal fix.

## Cost and stopping conditions

BMP costs may be sampled from `bmp_cost` when configured. A CPS may have one
or more named `cost_component` rows. For every individual BMP placement, the
model makes a fresh draw from every component row associated with that CPS,
scales each draw using that row's unit and the realized BMP quantity, and sums
the components into the placement's `cost_usd`. Thus a cover crop can be
represented as seed + establishment + termination, while a structural BMP can
mix area-, length-, or project-based components. Cost can also optionally
influence BMP-selection probabilities when `bmp_sel_prob_via_costs` is enabled;
the representative costs of all components are summed before probabilities are
calculated.

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