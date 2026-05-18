"""Central constants for config keys, CSV columns, and axis labels."""

# Config keys
CFG_DOMAIN = "domain"
CFG_PARCELS = "parcels"
CFG_OUTLET_LOC = "outlet_loc"
CFG_PARCEL_OUT = "parcel_out"
CFG_PARCEL_UP = "parcel_up"
CFG_PARCEL_P = "parcel_p"
CFG_POLLUTANTS = "pollutants"
CFG_CPS = "cps"
CFG_POLLUTANT_YIELD = "pollutant_yield"
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

# Data payload keys (used in the validated data dict passed to Simulator)
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
DATA_POLLUTANT_YIELD = "pollutant_yield"
DATA_DELIVERY_RATIOS = "delivery_ratios"
DATA_BMP_LIMIT_N = "bmp_limit_n"
DATA_BMP_LIMIT_USD = "bmp_limit_usd"
DATA_N_SCENARIOS = "n_scenarios"
DATA_RANDOM_SEED = "random_seed"
DATA_AVG_AREA_HA = "avg_area_ha"
DATA_AVG_PERIM_M = "avg_perim_m"

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

# Output and axis constants
XAXIS_COST = "cost"
XAXIS_COUNT = "count"
YAXIS_TOTAL = "total"
YAXIS_TARGET = "target"
YAXIS_MEAN = "mean"

# Default values
DEFAULT_BUFFER_DEPTH_FT = 35.0
OUTPUT_TREATED_PREFIX = "treated_"
OUTPUT_REMOVED_PREFIX = "removed_"
OUTPUT_BASELINE_PREFIX = "baseline_"
OUTPUT_FINAL_PREFIX = "final_"

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
