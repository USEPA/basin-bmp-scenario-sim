# BASIN-BMP-SIMulator 

`basin-bmp-sim` is a watershed BMP scenario simulator for analyzing parcel-level pollutant yields, BMP efficiencies, and outlet delivery outcomes.

This repository contains code for a probabilistic model of best management practice (BMP) (alternatively, 'conservation practices') impacts on basin-scale pollutant loads.

### Contact

evenson.grey@epa.gov

## Configuration

Required configuration keys:

- `domain`: watershed boundary file (`.gpkg`, `.shp`, etc.)
- `parcels`: parcel polygons file
- `outlet_loc`: outlet location file
- `parcel_out`: CSV mapping parcels to outlet IDs
- `pollutants`: list of pollutant labels
- `cps`: list of BMP CPS codes
- `pollutant_yield`: CSV of pollutant yield statistics per parcel
- `bmp_efficiency`: CSV of BMP efficiency statistics per BMP type and pollutant
- `n_scenarios`: number of scenarios to produce
- one of `bmp_limit_n` or `bmp_limit_usd`

Optional configuration keys:

- `parcel_up`: CSV of parcel upstream connectivity
- `parcel_p`: parcel selection probability weights
- `bmp_cost`: CSV of BMP cost statistics
- `delivery_ratios`: CSV of parcel-to-outlet delivery ratios
- `outlet_target`: CSV of outlet pollutant reduction targets
- `outlet_mean`: CSV of outlet mean load metrics
- `buffer_depth_ft`: buffer depth in feet for grassed BMPs

## Outputs

The model writes results to the configured `outputs` directory:

- `bmps.csv` (aggregated across all scenarios, includes `scenario` and `cps_name`)
- `parcels.csv` (aggregated across all scenarios, includes `scenario`)
- `plot_*` files for summary visualizations
- `log_*.txt`

## Notes

- Pollutant labels are normalized from aliases such as `nitrogen`, `phosphorus`, and `sediment`.
- `parcel_out` outlet IDs must exist in `outlet_loc`.
- If both `bmp_limit_n` and `bmp_limit_usd` are specified, the simulation stops when either limit is reached.
