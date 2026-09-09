# PLET/RUSLE load-generation mode

[← Back to main README](../readme.md)

## Purpose

`plet_rusle` mode calculates baseline parcel pollutant loads internally using PLET-style runoff and infiltration calculations, pollutant concentrations, and optional RUSLE sediment generation.

This mode is appropriate when annual parcel loads should be derived from parcel hydrology and concentration assumptions rather than supplied directly as statistical parcel load-rate inputs.

## Required configuration

A typical configuration is:

    load_generation:
      mode: plet_rusle
      plet_inputs: ./inputs/plet/plet_inputs.csv
      hydrology_lookup: ./inputs/plet/plet_hydrology_lookup.csv
      rusle_inputs: ./inputs/plet/rusle_inputs.csv
      pollutant_concentrations: ./inputs/plet/pollutant_concentrations.csv
      groundwater_concentrations: ./inputs/plet/groundwater_concentrations.csv

## Production pathways

`plet_rusle` has exactly two production pathways:

- `surface`
- `subsurface`

The production calculation does not create separate shallow and deep subsurface pathways. Compatibility diagnostic fields may still appear in some outputs for older callers or tests, but they should not be interpreted as additional production pathways.

## Required PLET hydrology input

`plet_rusle` mode requires:

    load_generation:
      mode: plet_rusle
      hydrology_lookup: ./path/to/plet_hydrology_lookup.csv

The table is a user input, not a source-code lookup. It uses long form:

    land_cover,hsg,parameter,value,distribution_id,mean,sd,min,p05,p50,p95,max,sample_group,units
    cropland,B,cn,78,,,,,,,,,,,dimensionless
    cropland,B,infiltration_fraction,0.30,,,,,,,,,,,fraction

For every supported land-cover/HSG pairing, the file must contain exactly one `cn` row and exactly one `infiltration_fraction` row.

Supported land covers are:

- `urban`
- `cropland`
- `pastureland`
- `forest`
- `user_defined`

Supported HSG values are:

- `A`
- `B`
- `C`
- `D`

This produces 40 required parameter rows: 5 land covers × 4 HSGs × 2 parameters.

CN must remain in `(0, 100]`. Infiltration fraction must remain in `[0, 1]`. A fixed value or a distribution can be supplied for either parameter.

Example stochastic pair:

    land_cover,hsg,parameter,value,distribution_id,mean,sd,min,max,units
    cropland,B,cn,,,78,2,70,86,dimensionless
    cropland,B,infiltration_fraction,,,0.30,0.03,0.20,0.40,fraction

Each parcel classified as cropland/HSG B samples those pair-specific distributions independently unless `sample_group` is supplied.

## `plet_inputs`

`load_generation.plet_inputs` is a required long-form parcel parameter table. It supplies climate variables and fixed `land_cover` and `hsg` classifications.

Numeric rows use the standardized fixed-value/distribution schema and may use `pid="*"` defaults with parcel-specific overrides.

`cn` and `infiltration_fraction` are **not** supplied in this table.

## RUSLE inputs

`load_generation.rusle_inputs` is optional. Numeric rows use the common distribution schema and may use `pid="*"` defaults.

A parcel with RUSLE data must have a complete factor set and may supply `sdr` to override the default sediment delivery ratio of 1.0.

## Pollutant concentrations

### Runoff concentrations

`load_generation.pollutant_concentrations` is required when TN or TP is modeled.

TSS concentration is also needed when RUSLE is not available for a parcel and TSS is modeled. `pid="*"` defaults and parcel-specific overrides are supported.

### Groundwater concentrations

`load_generation.groundwater_concentrations` is required for each modeled non-TSS pollutant. These values are used with PLET infiltration volume to calculate the `subsurface` pathway. `pid="*"` defaults and parcel-specific overrides are supported.

## BMP efficiency expectations

A `surface` efficiency is required for every configured CPS × pollutant combination.

A missing or unusable correctly labeled `subsurface` efficiency defaults to zero and is logged.

Example:

    cps,pollutant,pathway,value,distribution_id,mean,sd,min,max
    340,TN,surface,,,0.35,,0.20,0.50
    340,TN,subsurface,0,,,,,
    340,TP,surface,,,0.25,,0.10,0.40

Unexpected pathway labels such as `shallow subsurface` or `deep subsurface` are not remapped to the PLET `subsurface` pathway. They are ignored in this mode, a warning is logged, and the subsurface efficiency defaults to zero if no valid `subsurface` row remains.

This prevents an incorrectly labeled row from silently changing infiltration-derived nutrient treatment.

## Removed and legacy configuration concepts

`load_generation.pathway_mode` has been removed. `plet_rusle` always derives its two production pathways from PLET and RUSLE inputs.

`watershed_area_mi2` has been removed from `rusle_inputs` and is an error if supplied, along with its `watershed_area_sqmi` and `watershed_area_sq_mi` spellings. No delivery ratio was ever derived from it; parcels that relied on it were silently computed at a delivery ratio of 1.0. Supply `sdr` explicitly instead.

Legacy `groundwater_loads` and `treat_groundwater_with_bmps` keys do not determine production pathway generation. The current production calculation always estimates subsurface load from groundwater concentration and sampled infiltration. Whether a BMP reduces that load is determined by the BMP's `subsurface` efficiency.

Statistical pathway-fraction settings are not used to generate baseline PLET/RUSLE loads.

## Main distinction from statistical mode

In statistical mode, the user supplies the baseline parcel pollutant load-rate inputs and defines the pathways. In `plet_rusle`, the model calculates baseline parcel loads and the pathway meanings are fixed by the hydrologic formulation.

## Recommended file organization

For a PLET/RUSLE project:

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

The organization deliberately separates different scientific variable families while keeping the numeric specification syntax the same in every file.