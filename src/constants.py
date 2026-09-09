"""Central constants used throughout the simulation.

This module defines the canonical configuration keys, validated data payload
keys, CSV column names, output labels, pollutant aliases, and BMP names used by
the rest of the application.

Notes
-----
Units are normalized as follows:

* lengths are in meters (m)
* areas are in hectares (ha)
* costs are in USD

Canonical pollutant labels are defined by ``POLLUTANT_CANONICAL`` and aliases
map through ``POLLUTANT_ALIAS_MAP``. Output prefixes such as ``treated_`` and
``removed_`` denote per-pollutant loads, while ``cost_usd`` and
``total_cost_usd`` capture BMP costing.
"""

from pathlib import Path

# Unit conversions
FT_TO_M = 0.3048
M2_PER_HA = 10_000.0
INCH_OVER_HA_TO_LITERS = 254_000.0
TON_PER_ACRE_TO_KG_PER_HA = 907.18474 / 0.40468564224
CURRENT_TIMESTEP_YEARS = 1.0


# Config keys
CFG_DOMAIN = "domain"
CFG_PARCELS = "parcels"
CFG_OUTLET_LOC = "outlet_loc"
CFG_PARCEL_OUT = "parcel_out"
CFG_PARCEL_UP = "parcel_up"
CFG_PARCEL_P = "parcel_p"
CFG_POLLUTANTS = "pollutants"
CFG_CPS = "cps"
CFG_POLLUTANT_LOAD_RATE = "pollutant_load_rate"
CFG_BMP_EFFICIENCY = "bmp_efficiency"
CFG_BMP_COST = "bmp_cost"
CFG_DELIVERY_RATIOS = "delivery_ratios"
CFG_OUTLET_TARGET = "outlet_target"
CFG_OUTLET_MEAN = "outlet_mean"
CFG_N_SCENARIOS = "n_scenarios"
CFG_BMP_LIMIT_N = "bmp_limit_n"
CFG_BMP_LIMIT_USD = "bmp_limit_usd"
CFG_BMP_SEL = "bmp_sel"
CFG_PARALLEL = "parallel"
CFG_RANDOM_SEED = "random_seed"
CFG_OUTPUTS = "outputs"
CFG_VERBOSE = "verbose"
CFG_BUFFER_DEPTH_FT = "buffer_depth_ft"
CFG_BMP_SEL_PROB_VIA_COSTS = "bmp_sel_prob_via_costs"
CFG_INPUT_DISTRIBUTIONS = "input_distributions"

# Canonical output folders/files
DIR_SCENARIO_METRICS = "scenario_metrics"
DIR_OUTLET_TRAJECTORIES = "outlet_trajectories"
FILE_ALL_SCENARIOS_PARQUET = "all_scenarios.parquet"

# Optional PLET/RUSLE load-generation block
CFG_LOAD_GENERATION = "load_generation"
LOAD_MODE_STATISTICAL = "statistical"
LOAD_MODE_PLET_RUSLE = "plet_rusle"
LOAD_PLET_INPUTS = "plet_inputs"
LOAD_HYDROLOGY_LOOKUP = "hydrology_lookup"
LOAD_RUSLE_INPUTS = "rusle_inputs"
LOAD_CONCENTRATIONS = "pollutant_concentrations"
LOAD_GROUNDWATER_CONCENTRATIONS = "groundwater_concentrations"
LOAD_GROUNDWATER_LOADS = "groundwater_loads"
LOAD_TREAT_GROUNDWATER_WITH_BMPS = "treat_groundwater_with_bmps"

# PLET/RUSLE parameter schema and lookup metadata
PLET_CLASSIFICATION_PARAMETERS = ("land_cover", "hsg")
PLET_LAND_COVERS = ("urban", "cropland", "pastureland", "forest", "user_defined")
PLET_HSG_VALUES = ("A", "B", "C", "D")
PLET_HYDROLOGY_LOOKUP_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples" / "east_fork" / "inputs" / "plet" / "plet_hydrology_lookup.csv"
)
PLET_DERIVED_PARAMETERS = ("cn", "infiltration_fraction")
PLET_REQUIRED_INPUTS = (
    "annual_precip_in",
    "rain_days",
    "rain_correction_fraction",
    "runoff_day_fraction",
    "land_cover",
    "hsg",
)
PLET_REQUIRED_RESOLVED_INPUTS = (
    "annual_precip_in",
    "rain_days",
    "rain_correction_fraction",
    "runoff_day_fraction",
    "cn",
    "infiltration_fraction",
)
RUSLE_REQUIRED_INPUTS = ("r", "k", "ls", "c", "p")
PLET_LAND_COVER_ALIASES = {
    "urban": "urban",
    "developed": "urban",
    "cropland": "cropland",
    "crop": "cropland",
    "row_crop": "cropland",
    "row_crops": "cropland",
    "pasture": "pastureland",
    "pastureland": "pastureland",
    "forest": "forest",
    "forested": "forest",
    "woodland": "forest",
    "woods": "forest",
    "user_defined": "user_defined",
    "userdefined": "user_defined",
}
PLET_PARAMETER_ALIASES = {
    "annual_rainfall_in": "annual_precip_in",
    "annual_precipitation_in": "annual_precip_in",
    "ar": "annual_precip_in",
    "rdays": "rain_days",
    "rainfall_correction": "rain_correction_fraction",
    "rcor": "rain_correction_fraction",
    "rain_day_correction": "runoff_day_fraction",
    "rdcor": "runoff_day_fraction",
    "curve_number": "cn",
    "land_use": "land_cover",
    "landuse": "land_cover",
    "land_cover_class": "land_cover",
    "land_cover_classification": "land_cover",
    "hydrologic_soil_group": "hsg",
    "soil_hydrologic_group": "hsg",
    "soil_group": "hsg",
    "hsg_classification": "hsg",
    "shg": "hsg",
    "initial_abstraction_ratio": "ia_ratio",
    "alpha": "ia_ratio",
    "rusle_r": "r",
    "rusle_k": "k",
    "rusle_ls": "ls",
    "rusle_c": "c",
    "rusle_p": "p",
    "delivery_ratio": "sdr",
    "sediment_delivery_ratio": "sdr",
    "soil_n_percent": "sediment_n_pct",
    "soil_p_percent": "sediment_p_pct",
    "enrichment": "enrichment_ratio",
    "infiltration_frac": "infiltration_fraction",
    "infiltration_factor": "infiltration_fraction",
    "gw_infiltration_fraction": "infiltration_fraction",
    "groundwater_infiltration_fraction": "infiltration_fraction",
    "shallow_subsurface_fraction": "fraction_subsurface_shallow",
    "fraction_shallow_subsurface": "fraction_subsurface_shallow",
    "subsurface_shallow_fraction": "fraction_subsurface_shallow",
}

# New: BMP failure configuration
CFG_BMP_FAIL_RATE = "bmp_fail_rate"            # probability [0,1] a BMP fails
CFG_BMP_FAIL_REDUCTION = "bmp_fail_reduction"  # efficiency scale [0,1] on failure

# Fractions to split parcel load rates by pathway
CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE = "pollutant_load_rate_frac_surface"
CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW = "pollutant_load_rate_frac_shallow"
CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS = "pollutant_load_rate_pathway_fractions"

# Data payload keys (used in the validated data dict passed to Model)
DATA_PARCELS = "parcels"
DATA_PARCEL_P = "parcel_p"
DATA_PARCEL_UP_MAP = "parcel_up_map"
DATA_PARCEL_OUT_MAP = "parcel_out_map"
DATA_POLLUTANTS = "pollutants"
DATA_CPS = "cps"
DATA_OUTLET_LOC = "outlet_loc"
DATA_OUTLET_TARGET = "outlet_target"
DATA_OUTLET_MEAN = "outlet_mean"
DATA_BMP_EFFICIENCY = "bmp_eff"
DATA_BMP_COST = "bmp_cost"
DATA_POLLUTANT_LOAD_RATE = "pollutant_load_rate"
DATA_DELIVERY_RATIOS = "delivery_ratios"
DATA_BMP_LIMIT_N = "bmp_limit_n"
DATA_BMP_LIMIT_USD = "bmp_limit_usd"
DATA_N_SCENARIOS = "n_scenarios"
DATA_RANDOM_SEED = "random_seed"
DATA_AVG_AREA_HA = "avg_area_ha"
DATA_AVG_PERIM_M = "avg_perim_m"
DATA_LOAD_GENERATION = "load_generation"
DATA_PLET_INPUTS = "plet_inputs"
DATA_RUSLE_INPUTS = "rusle_inputs"
DATA_POLLUTANT_CONCENTRATIONS = "pollutant_concentrations"
DATA_GROUNDWATER_CONCENTRATIONS = "groundwater_concentrations"
DATA_PATHWAYS = "pathways"
DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS = "pollutant_load_rate_pathway_fractions"
DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE = "pollutant_load_rate_is_aggregate"

# Common column names
COL_PID = "pid"
COL_OID = "oid"
COL_CPS = "cps"
COL_POLLUTANT = "pollutant"
COL_OIDS = "oids"
COL_PID_UP = "pid_up"
COL_PROBABILITY = "probability"
COL_UNIT = "unit"
COL_AREA_M2 = "area_m2"
COL_AREA_HA = "area_ha"
COL_PERIM_M = "perim_m"
COL_TARGET = "target"
COL_MEAN = "mean"
COL_SD = "sd"
COL_MIN = "min"
COL_MAX = "max"
COL_SDR_F_TO_S = "sdr_f_to_s"
COL_SDR_S_TO_O = "sdr_s_to_o"
COL_NDR_F_TO_S = "ndr_f_to_s"
COL_NDR_S_TO_O = "ndr_s_to_o"
PERCENTILE_PREFIX = "p"

# New: optional pathway column
COL_PATHWAY = "pathway"
COL_DISTRIBUTION_ID = "distribution_id"
COL_SAMPLE_GROUP = "sample_group"
COL_MASS_TIMESTEP_YEARS = "mass_timestep_years"

# Standardized uncertain-input schema
DISTRIBUTION_STAT_ALIASES = {
    "average": "mean",
    "avg": "mean",
    "std": "sd",
    "minimum": "min",
    "maximum": "max",
    "p0": "min",
    "p100": "max",
}
DISTRIBUTION_NAMED_STATS = ("value", "mean", "sd", "min", "max")

# Output and axis constants
XAXIS_COST = "cost"
XAXIS_COUNT = "count"
YAXIS_TOTAL = "total"
YAXIS_TARGET = "target"
YAXIS_MEAN = "mean"

# Default values
DEFAULT_BUFFER_DEPTH_FT = 35.0
DEFAULT_BMP_FAIL_REDUCTION = 0.25  # used when a failure occurs but reduction not provided

# BMP identifiers and model heuristics
CPS_GRASSED_WATERWAY = 412
CPS_CONSTRUCTED_WETLAND = 656
WETLAND_AREA_COST_CPS = (656, 657)
WETLAND_AREA_HA_STATS = {"min": 0.1, "p25": 0.4, "p50": 0.81, "p75": 2.0, "max": 4.0}
WETLAND_CATCHMENT_RATIO_STATS = {"min": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "max": 100.0}
GRASSED_WATERWAY_PERIMETER_FRACTION_STATS = {"min": 0.1, "max": 0.3, "mean": 0.2}
GRASSED_WATERWAY_TREATED_FRACTION_STATS = {"min": 0.2, "max": 0.4, "mean": 0.3}
PROB_EST_WETLAND_MAX_AREA_HA = 0.8
PROB_EST_BUFFER_PERIM_FRACTION = 0.2
MIN_BMP_SELECTION_COST_USD = 0.01

OUTPUT_BASELINE_PREFIX = "baseline_"
OUTPUT_FINAL_PREFIX = "final_"

# Mass-based summary column naming
BASELINE_MASS_PREFIX = "baseline_mass_"
TREATED_BASELINE_MASS_PREFIX = "treated_baseline_mass_"
REMOVED_MASS_PREFIX = "removed_mass_"
MASS_SUFFIX = "_kg"
TREATMENT_EXPOSURE_PREFIX = "treatment_exposure_fraction_"
REALIZED_EFFICIENCY_PREFIX = "realized_efficiency_"
OVERALL_REDUCTION_PREFIX = "overall_reduction_fraction_"

# Output record suffixes
OUTPUT_EFFICIENCY_JSON = "efficiency_json"
OUTPUT_LINEAR_LENGTH = "linear_length_m"
OUTPUT_BUFFER_AREA = "buffer_area_ha"
OUTPUT_PORTION_TREATED = "portion_treated"
OUTPUT_WETLAND_AREA = "wetland_area_ha"
OUTPUT_CATCHMENT_RATIO = "catchment_to_wetland_ratio"
OUTPUT_IMPACTED_PIDS = "impacted_pids"
OUTPUT_TREATED = "treated"
OUTPUT_REMOVED = "removed"
OUTPUT_COST_USD = "cost_usd"
OUTPUT_TOTAL_COST_USD = "total_cost_usd"
OUTPUT_EFFICIENCY = "efficiency"
OUTPUT_BMP_FAILED = "failed"  # New: per-BMP failure flag in bmps CSVs

# Pollutant canonical labels and alias mapping
POLLUTANT_CANONICAL = ("TN", "TP", "TSS")
POLLUTANT_ALIAS_MAP = {
    "tn": "TN",
    "tp": "TP",
    "tss": "TSS",
    "nitrogen": "TN",
    "phosphorus": "TP",
    "sediment": "TSS",
}

# BMP CPS code name mapping
BMP_CPS_NAME_MAP = {
    329: "Residue Management (No-Till)",
    340: "Cover Crop",
    412: "Grassed Waterway",
    590: "Nutrient Management",
    656: "Constructed Wetland",
}

# Canonical pathway labels
PATHWAY_VALUES = ("surface", "shallow subsurface", "deep subsurface")
PLET_PATHWAY_VALUES = ("surface", "subsurface")