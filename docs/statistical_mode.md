# Statistical load-generation mode

[← Back to main README](../readme.md)

## Purpose

Statistical mode is the default load-generation approach.

It is appropriate when parcel pollutant quantities are already available from monitoring, another model, literature, calibration, or expert judgment. In this mode, the user supplies parcel pollutant load-rate inputs directly, and the model does not calculate runoff, infiltration, erosion, or groundwater transport explicitly.

If `load_generation.mode` is omitted, the model uses statistical mode.

## Required parcel pollutant input

Statistical mode requires:

    pollutant_load_rate: ./inputs/pollutant_load_rate.csv

The file may contain either:

- explicit parcel × pollutant × pathway values or distributions
- one aggregate parcel × pollutant value or distribution that is subsequently split with configured pathway fractions

`pid="*"` may define a default for all parcels, with exact parcel rows overriding the default for the same pollutant and pathway.

## Explicit pathway input

An explicit-pathway file may look like:

    pid,pollutant,pathway,value,distribution_id,mean,sd,min,max,units
    P101,TN,surface,,,8.5,1.2,5,12,kg/ha/yr
    P101,TN,shallow subsurface,,,3.0,0.7,1.5,5.0,kg/ha/yr
    P101,TP,surface,1.2,,,,,,kg/ha/yr

Pathway labels are user-defined in this mode. They may represent hydrologic or bookkeeping categories such as:

- `surface`
- `shallow subsurface`
- `deep subsurface`
- `tile`
- `groundwater`

The model does not infer hydrologic meaning from the labels by itself.

## Aggregate parcel input with configured pathway split

If the statistical parcel table contains one aggregate parcel pollutant quantity per parcel × pollutant while `bmp_efficiency.csv` defines multiple pathways, define the pathway split explicitly:

    pollutant_load_rate_pathway_fractions:
      surface: 0.70
      shallow subsurface: 0.20
      tile: 0.10

The fractions must correspond to active BMP-efficiency pathways and sum to 1.0.

An aggregate-input table may look like:

    pid,pollutant,value,distribution_id,mean,sd,min,max,units
    P101,TN,,,11.5,1.8,7,16,kg/ha/yr
    P101,TP,1.4,,,,,,kg/ha/yr

The configured fractions are then used to partition that aggregate parcel pollutant quantity into active pathways.

## Wildcard defaults for large parcel datasets

For large parcel datasets, `pollutant_load_rate.csv` may use `pid="*"` as a default distribution for all parcels. Exact parcel rows override the wildcard row for the same pollutant and pathway.

Example:

    pid,pollutant,pathway,value,distribution_id,mean,sd,min,max,units
    *,TN,surface,,tn_surface_default,,,,,kg/ha/yr
    P104,TN,surface,,,9.2,1.4,6,13,kg/ha/yr

This is useful when many parcels share an assumption. When parcel distributions are genuinely unique, provide one row per parcel × pollutant × pathway, or one aggregate row per parcel × pollutant if pathway splitting is configured separately.

## BMP-efficiency coverage rules

Every active pathway requires an explicitly defined BMP efficiency for every configured CPS × pollutant combination.

If statistical mode uses pathways such as `surface`, `shallow subsurface`, and `tile`, then `bmp_efficiency.csv` must provide coverage for all active CPS × pollutant × pathway combinations that can occur during the scenario simulation.

## What statistical mode does not do

Statistical mode does **not**:

- compute runoff from precipitation
- compute infiltration from Curve Number or hydrologic soil group
- compute sediment generation with RUSLE
- infer pathway fractions automatically unless configured
- attach a fixed hydrologic interpretation to user-defined pathway labels

Those processes must already be represented in the supplied parcel pollutant load-rate inputs or in the assumptions used to create them.

## Interpretation

Statistical mode is best understood as a probabilistic scenario framework for parcel pollutant loads and BMP responses. Results depend on the quality of the supplied parcel load-rate assumptions, the chosen pathway definitions, BMP efficiencies, routing assumptions, and stopping rules.