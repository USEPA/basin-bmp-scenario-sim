# BASIN-BMP-SIMulator 

`basin-bmp-sim` is a probabilistic basin-scale BMP scenario simulator to assess the liklihood of cost-effectively meeting basin-scale pollutant load reduction targets

## Description

`basin-bmp-sim` simulates basin-wide best management practice (BMP) implementation scenarios showing aggregate costs and impacts on basin-outlet pollutant loads as a function of random draws from user-described statistical distributions depicting:
- Parcel selection
  - Describe the relative liklihood of BMP implementation across the basin's parcel / agricultural fields
- BMP / conservation practice type
  - Describe the relative liklihood that specific types of BMPs or conservation practices will be implemented, including BMP-specific characteristics such as:
    - Wetland area
    - Wetland catchment-to-area ratio
    - Grassed waterway length
    - Portion of parcel draining to the BMP
- Cost
  - Describe the likely costs (e.g., annualized USD) (inluding opportunity, construction, maintenance) of implementing individual types BMPs
- Parcel pollutant yield
  - Describe the likely yield rates (e.g., kg/ha/yr) for specific pollutant types across basin parcels  
- BMP efficiency
  - Describe the likely effectiveness of specific types of BMPs 

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

### Contact

evenson.grey@epa.gov
