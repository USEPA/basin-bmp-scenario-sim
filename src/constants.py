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

from dataclasses import dataclass
import math
import re
from typing import Dict, Mapping, Optional, Tuple

# Unit conversions
FT_TO_M = 0.3048
M2_PER_HA = 10_000.0
INCH_OVER_HA_TO_LITERS = 254_000.0
TON_PER_ACRE_TO_KG_PER_HA = 907.18474 / 0.40468564224
CURRENT_TIMESTEP_YEARS = 1.0

# AFO manure and feedlot assumptions
AFO_FEEDLOT_CURVE_NUMBERS = (91.0, 92.0, 93.0, 94.0)
AFO_FEEDLOT_MAX_CONCENTRATIONS = {"TN": 1500.0, "TP": 300.0, "BOD": 2000.0}
AFO_SEASONAL_P_WEIGHTS = {"winter": 0.20, "spring": 0.15, "summer": 0.10, "fall": 0.05}
AFO_SPECIES_ALIASES = {
    "beef_steer": "beef_finishing",
    "dairy_cow": "dairy_lactating",
    "swine": "swine_grow_finish",
    "broiler": "poultry_broiler",
}

# Custom logging level
VERBOSE_LEVEL_NUM = 15

# Input schema metadata
INPUT_SCHEMA_TABLE = "model_input_schema"
INPUT_SCHEMA_VERSION = 3
PLET_REQUIRED_PARAMETERS = frozenset({"land_cover"})

# Config keys
CFG_DOMAIN = "domain"
CFG_PARCELS = "parcels"
CFG_PLET_FORCING = "plet_forcing"
CFG_HUC_PARAMETERS = "huc_parameters"
CFG_OUTLETS = "outlets"
CFG_POLLUTANTS = "pollutants"
CFG_CPS = "cps"
CFG_POLLUTANT_LOAD_RATE = "pollutant_load_rate"
CFG_BMP_EFFICIENCY = "bmp_efficiency"
CFG_BMP_COST = "bmp_cost"
CFG_DELIVERY_RATIOS = "delivery_ratios"
CFG_N_SCENARIOS = "n_scenarios"
CFG_BMP_LIMIT_N = "bmp_limit_n"
CFG_BMP_LIMIT_USD = "bmp_limit_usd"
CFG_BMP_SEL = "bmp_sel"
CFG_PARALLEL = "parallel"
CFG_DETERMINISTIC_PARCEL_ORDER = "deterministic_parcel_order"
CFG_DETERMINISTIC_BMP_TYPE_ORDER = "deterministic_bmp_type_order"
CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS = "deterministic_bmp_design_parameters"
CFG_RANDOM_SEED = "random_seed"
CFG_OUTPUTS = "outputs"
CFG_VERBOSE = "verbose"
CFG_BUFFER_DEPTH_FT = "buffer_depth_ft"
CFG_BMP_SEL_PROB_VIA_COSTS = "bmp_sel_prob_via_costs"

# Configuration path resolution
PLET_HYDROLOGY_LABEL = "plet_hydrology_inputs"
CONFIG_PATH_KEYS = (
    CFG_DOMAIN,
    CFG_PARCELS,
    CFG_PLET_FORCING,
    CFG_HUC_PARAMETERS,
    CFG_OUTLETS,
    "animal_operations",
    CFG_BMP_EFFICIENCY,
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_OUTPUTS,
)
LOAD_GENERATION_PATH_KEYS: tuple[str, ...] = ()

# Canonical output folders/files
DIR_SCENARIO_METRICS = "scenario_metrics"
DIR_OUTLET_TRAJECTORIES = "outlet_trajectories"
FILE_ALL_SCENARIOS_PARQUET = "all_scenarios.parquet"

# Optional PLET/RUSLE load-generation block
CFG_LOAD_GENERATION = "load_generation"
LOAD_MODE_STATISTICAL = "statistical"
LOAD_MODE_PLET_RUSLE = "plet_rusle"
LOAD_PLET_INPUTS = "plet_inputs"
LOAD_RUSLE_INPUTS = "rusle_inputs"
LOAD_CONCENTRATIONS = "pollutant_concentrations"
LOAD_GROUNDWATER_CONCENTRATIONS = "groundwater_concentrations"
LOAD_GROUNDWATER_LOADS = "groundwater_loads"
LOAD_TREAT_GROUNDWATER_WITH_BMPS = "treat_groundwater_with_bmps"

# PLET/RUSLE parameter schema and lookup metadata
PLET_CLASSIFICATION_PARAMETERS = ("land_cover", "hsg")
PLET_LAND_COVERS = ("urban", "cropland", "pastureland", "forest", "user_defined")
PLET_HSG_VALUES = ("A", "B", "C", "D")
PLET_DERIVED_PARAMETERS = ("cn", "infiltration_fraction")
PLET_REQUIRED_INPUTS = (
    "rain_days",
    "runoff_day_fraction",
    "land_cover",
    "hsg",
)
PLET_PRECIPITATION_INPUT_ALTERNATIVES = (
    ("avg_rain_in",),
    ("annual_precip_in", "rain_correction_fraction"),
)
PLET_REQUIRED_RESOLVED_INPUTS = (
    "rain_days",
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
    "avg_rain": "avg_rain_in",
    "average_rain": "avg_rain_in",
    "average_rain_in": "avg_rain_in",
    "average_rainfall": "avg_rain_in",
    "average_rainfall_in": "avg_rain_in",
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


# Canonical tables/layers in consolidated GeoPackage inputs.
# Structural tables have no ``input_`` prefix. Every user-supplied parcel-side
# model variable is stored in its own ``input_*`` table.
GPKG_PARCELS_LAYER = "parcels"
GPKG_PARCEL_UP_TABLE = "parcel_up"
GPKG_PARCEL_OUTLETS_TABLE = "parcel_outlets"
GPKG_PARCEL_HUC12_TABLE = "parcel_huc12"
GPKG_PLET_HUC12_LAYER = "huc12"
GPKG_OUTLETS_LAYER = "outlets"
GPKG_OUTLET_STATS_TABLE = "outlet_stats"

# Standard per-variable parcel input tables. Values in these tables all use the
# same fixed-value/distribution columns; only their key columns differ.
PARCEL_PARAMETER_INPUT_TABLES = {
    "annual_precip_in": "input_annual_precip_in",
    "avg_rain_in": "input_avg_rain_in",
    "rain_days": "input_rain_days",
    "rain_correction_fraction": "input_rain_correction_fraction",
    "runoff_day_fraction": "input_runoff_day_fraction",
    "land_cover": "input_land_cover",
    "hsg": "input_hsg",
    "ia_ratio": "input_ia_ratio",
    "r": "input_rusle_r",
    "k": "input_rusle_k",
    "ls": "input_rusle_ls",
    "c": "input_rusle_c",
    "p": "input_rusle_p",
    "sdr": "input_sediment_delivery_ratio",
    "sediment_n_pct": "input_sediment_n_pct",
    "sediment_p_pct": "input_sediment_p_pct",
    "enrichment_ratio": "input_enrichment_ratio",
}
GPKG_INPUT_SELECTION_WEIGHT = "input_selection_weight"
GPKG_INPUT_CURVE_NUMBER = "input_curve_number"
GPKG_INPUT_INFILTRATION_FRACTION = "input_infiltration_fraction"
GPKG_INPUT_POLLUTANT_LOAD_RATE = "input_pollutant_load_rate"
GPKG_INPUT_SURFACE_CONCENTRATION = "input_surface_concentration"
GPKG_INPUT_SUBSURFACE_CONCENTRATION = "input_subsurface_concentration"
GPKG_DELIVERY_RATIO_TABLES = {
    "sdr_f_to_s": "input_sdr_f_to_s",
    "sdr_s_to_o": "input_sdr_s_to_o",
    "ndr_f_to_s": "input_ndr_f_to_s",
    "ndr_s_to_o": "input_ndr_s_to_o",
}


# HUC12-scale PLET forcing uses the same one-variable-per-table pattern as
# parcels.gpkg, but rows are keyed by huc12 instead of pid.  The spatial
# ``huc12`` layer contains only WBD geometry/metadata; model forcing belongs in
# these attribute tables so every value may be fixed or distributional.
HUC12_PLET_PARAMETER_INPUT_TABLES = {
    "avg_rain_in": "input_avg_rain_in",
    "annual_precip_in": "input_annual_precip_in",
    "rain_days": "input_rain_days",
    "rain_correction_fraction": "input_rain_correction_fraction",
    "runoff_day_fraction": "input_runoff_day_fraction",
    "hsg": "input_hsg",
    "r": "input_rusle_r",
    "k": "input_rusle_k",
    "ls": "input_rusle_ls",
    "c": "input_rusle_c",
    "p": "input_rusle_p",
    "sediment_n_pct": "input_sediment_n_pct",
    "sediment_p_pct": "input_sediment_p_pct",
}

RUSLE_PARAMETER_NAMES = (
    "r", "k", "ls", "c", "p", "sdr",
    "sediment_n_pct", "sediment_p_pct", "enrichment_ratio",
)

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
COL_HUC12 = "huc12"
COL_AREA_FRACTION = "area_fraction"
COL_OID = "oid"
COL_CPS = "cps"
COL_COST_COMPONENT = "cost_component"
COL_POLLUTANT = "pollutant"
COL_OIDS = "oids"
COL_PID_UP = "pid_up"
COL_PROBABILITY = "probability"
COL_SELECTION_WEIGHT = "selection_weight"
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
    313: "Waste Storage Facility",
    329: "Residue Management (No-Till)",
    340: "Cover Crop",
    412: "Grassed Waterway",
    590: "Nutrient Management",
    656: "Constructed Wetland",
}

# Canonical pathway labels
PATHWAY_VALUES = ("surface", "shallow subsurface", "deep subsurface")
PLET_PATHWAY_VALUES = ("surface", "subsurface")


# Unit validation specifications
@dataclass(frozen=True)
class UnitSpec:
    """Describe one canonical unit and accepted conversion aliases."""

    canonical: str
    aliases: Mapping[str, float]


def _unit_key(value: object) -> str:
    """Normalize a human-authored unit label for comparison."""
    text = str(value).strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = text.replace("²", "2").replace("^2", "2")
    text = text.replace("·", "*").replace("×", "*")
    text = text.replace("$", "usd")
    text = text.replace("_", " ")
    text = re.sub(r"\bper\b", "/", text)
    return re.sub(r"\s+", "", text)


def _spec(canonical: str, aliases: Mapping[str, float]) -> UnitSpec:
    normalized = {_unit_key(canonical): 1.0}
    normalized.update({_unit_key(label): float(factor) for label, factor in aliases.items()})
    return UnitSpec(canonical=canonical, aliases=normalized)


FRACTION_SPEC = _spec(
    "fraction",
    {
        "ratio": 1.0, "decimal": 1.0, "dimensionless": 1.0, "unitless": 1.0, "1": 1.0,
        "percent": 0.01, "pct": 0.01, "%": 0.01,
    },
)
PERCENT_SPEC = _spec(
    "percent",
    {
        "pct": 1.0, "%": 1.0, "fraction": 100.0, "ratio": 100.0, "decimal": 100.0,
        "dimensionless": 100.0, "unitless": 100.0, "1": 100.0,
    },
)
DIMENSIONLESS_SPEC = _spec(
    "dimensionless", {"unitless": 1.0, "1": 1.0, "ratio": 1.0, "index": 1.0}
)
PRECIP_SPEC = _spec(
    "in/year",
    {
        "in/yr": 1.0, "inch/year": 1.0, "inches/year": 1.0, "inch/yr": 1.0,
        "inches/yr": 1.0, "mm/year": 1.0 / 25.4, "mm/yr": 1.0 / 25.4,
        "cm/year": 1.0 / 2.54, "cm/yr": 1.0 / 2.54, "m/year": 100.0 / 2.54,
        "m/yr": 100.0 / 2.54,
    },
)
EVENT_PRECIP_SPEC = _spec(
    "in/event",
    {
        "in": 1.0, "inch": 1.0, "inches": 1.0, "inch/event": 1.0,
        "inches/event": 1.0, "in/storm": 1.0, "inch/storm": 1.0,
        "inches/storm": 1.0, "mm": 1.0 / 25.4, "mm/event": 1.0 / 25.4,
        "mm/storm": 1.0 / 25.4, "cm": 1.0 / 2.54, "cm/event": 1.0 / 2.54,
        "cm/storm": 1.0 / 2.54,
    },
)
DAYS_PER_YEAR_SPEC = _spec(
    "days/year",
    {"day/year": 1.0, "days/yr": 1.0, "day/yr": 1.0, "d/year": 1.0, "d/yr": 1.0},
)
CONCENTRATION_SPEC = _spec(
    "mg/L",
    {
        "mg/liter": 1.0, "mg/litre": 1.0, "ug/L": 1.0e-3, "ug/liter": 1.0e-3,
        "ug/litre": 1.0e-3, "g/L": 1.0e3, "g/liter": 1.0e3, "g/litre": 1.0e3,
        "mg/m3": 1.0e-3, "g/m3": 1.0, "kg/m3": 1.0e3,
    },
)
LOAD_RATE_SPEC = _spec(
    "kg/ha/yr",
    {
        "kg/ha/year": 1.0, "kg/hectare/yr": 1.0, "kg/hectare/year": 1.0,
        "kg/ac/yr": 1.0 / 0.40468564224, "kg/acre/yr": 1.0 / 0.40468564224,
        "lb/ha/yr": 0.45359237, "lb/hectare/yr": 0.45359237,
        "lb/ac/yr": 0.45359237 / 0.40468564224,
        "lb/acre/yr": 0.45359237 / 0.40468564224,
        "lb/ac/year": 0.45359237 / 0.40468564224,
        "lb/acre/year": 0.45359237 / 0.40468564224, "g/m2/yr": 10.0,
        "g/m2/year": 10.0, "mg/m2/yr": 0.01, "mg/m2/year": 0.01,
        "kg/km2/yr": 0.01, "kg/km2/year": 0.01,
    },
)
MASS_RATE_SPEC = _spec(
    "kg/yr",
    {
        "kg/year": 1.0, "g/yr": 1.0e-3, "g/year": 1.0e-3, "mg/yr": 1.0e-6,
        "mg/year": 1.0e-6, "lb/yr": 0.45359237, "lb/year": 0.45359237,
        "ton/yr": 907.18474, "ton/year": 907.18474, "tons/yr": 907.18474,
        "tons/year": 907.18474, "metric ton/yr": 1000.0, "metric ton/year": 1000.0,
        "tonne/yr": 1000.0, "tonne/year": 1000.0,
    },
)
COST_AREA_SPEC = _spec(
    "usd/ha",
    {
        "usd/hectare": 1.0, "usd/ac": 1.0 / 0.40468564224,
        "usd/acre": 1.0 / 0.40468564224, "usd/m2": 10000.0,
        "usd per unit area": 1.0,
    },
)
COST_LENGTH_SPEC = _spec(
    "usd/m",
    {
        "usd/meter": 1.0, "usd/metre": 1.0, "usd/ft": 1.0 / 0.3048,
        "usd/foot": 1.0 / 0.3048, "usd/km": 0.001, "usd per unit length": 1.0,
    },
)
COST_PROJECT_SPEC = _spec(
    "usd/project",
    {"usd/bmp": 1.0, "usd/practice": 1.0, "usd/each": 1.0, "usd/unit": 1.0},
)
RUSLE_R_SPEC = _spec(
    "rusle-r-us-customary",
    {
        "us customary rusle r": 1.0, "us-customary-rusle-r": 1.0, "plet-r": 1.0,
        "index": 1.0,
    },
)
RUSLE_K_SPEC = _spec(
    "rusle-k-us-customary",
    {
        "us customary rusle k": 1.0, "us-customary-rusle-k": 1.0, "plet-k": 1.0,
        "ton acre hour/(acre foot ton inch)": 1.0,
    },
)

_KIND_SPECS: Dict[str, Tuple[UnitSpec, ...]] = {
    "fraction": (FRACTION_SPEC,),
    "percent": (PERCENT_SPEC,),
    "dimensionless": (DIMENSIONLESS_SPEC,),
    "precip": (PRECIP_SPEC,),
    "event_precip": (EVENT_PRECIP_SPEC,),
    "days_per_year": (DAYS_PER_YEAR_SPEC,),
    "concentration": (CONCENTRATION_SPEC,),
    "load_rate": (LOAD_RATE_SPEC,),
    "mass_rate": (MASS_RATE_SPEC,),
    "cost": (COST_AREA_SPEC, COST_LENGTH_SPEC, COST_PROJECT_SPEC),
    "rusle_r": (RUSLE_R_SPEC,),
    "rusle_k": (RUSLE_K_SPEC,),
}
_PARAMETER_KINDS: Dict[str, str] = {
    "annual_precip_in": "precip",
    "avg_rain_in": "event_precip",
    "rain_days": "days_per_year",
    "rain_correction_fraction": "fraction",
    "runoff_day_fraction": "fraction",
    "cn": "dimensionless",
    "ia_ratio": "fraction",
    "infiltration_fraction": "fraction",
    "runoff_multiplier": "dimensionless",
    "groundwater_multiplier": "dimensionless",
    "r": "rusle_r",
    "k": "rusle_k",
    "ls": "dimensionless",
    "c": "dimensionless",
    "p": "dimensionless",
    "sdr": "fraction",
    "sediment_multiplier": "dimensionless",
    "sediment_delivery_multiplier": "dimensionless",
    "sediment_n_pct": "percent",
    "sediment_p_pct": "percent",
    "enrichment_ratio": "dimensionless",
    "fraction_subsurface_shallow": "fraction",
}


# Physical input validation domains
@dataclass(frozen=True)
class PhysicalDomain:
    """Allowed physical domain for a user-supplied numeric quantity."""

    low: Optional[float] = None
    high: Optional[float] = None
    low_inclusive: bool = True
    high_inclusive: bool = True

    def contains(self, value: float) -> bool:
        numeric = float(value)
        if not math.isfinite(numeric):
            return False
        if self.low is not None:
            if self.low_inclusive and numeric < self.low:
                return False
            if not self.low_inclusive and numeric <= self.low:
                return False
        if self.high is not None:
            if self.high_inclusive and numeric > self.high:
                return False
            if not self.high_inclusive and numeric >= self.high:
                return False
        return True

    def describe(self) -> str:
        if self.low is None and self.high is None:
            return "all finite values"
        if self.low is None:
            operator = "<=" if self.high_inclusive else "<"
            return f"{operator} {self.high:g}"
        if self.high is None:
            operator = ">=" if self.low_inclusive else ">"
            return f"{operator} {self.low:g}"
        left = "[" if self.low_inclusive else "("
        right = "]" if self.high_inclusive else ")"
        return f"{left}{self.low:g}, {self.high:g}{right}"


NONNEGATIVE_DOMAIN = PhysicalDomain(low=0.0)
POSITIVE_DOMAIN = PhysicalDomain(low=0.0, low_inclusive=False)
FRACTION_DOMAIN = PhysicalDomain(low=0.0, high=1.0)
EFFICIENCY_DOMAIN = PhysicalDomain(high=1.0)
PERCENT_DOMAIN = PhysicalDomain(low=0.0, high=100.0)
CN_DOMAIN = PhysicalDomain(low=0.0, high=100.0, low_inclusive=False)
IA_RATIO_DOMAIN = PhysicalDomain(low=0.0, high=0.20)
_TABLE_PHYSICAL_DOMAINS: Dict[str, PhysicalDomain] = {
    CFG_POLLUTANT_LOAD_RATE: NONNEGATIVE_DOMAIN,
    CFG_BMP_EFFICIENCY: EFFICIENCY_DOMAIN,
    CFG_BMP_COST: NONNEGATIVE_DOMAIN,
    LOAD_CONCENTRATIONS: NONNEGATIVE_DOMAIN,
    LOAD_GROUNDWATER_CONCENTRATIONS: NONNEGATIVE_DOMAIN,
}
_PARAMETER_PHYSICAL_DOMAINS: Dict[str, PhysicalDomain] = {
    "annual_precip_in": NONNEGATIVE_DOMAIN,
    "avg_rain_in": NONNEGATIVE_DOMAIN,
    "rain_days": NONNEGATIVE_DOMAIN,
    "rain_correction_fraction": FRACTION_DOMAIN,
    "runoff_day_fraction": FRACTION_DOMAIN,
    "cn": CN_DOMAIN,
    "ia_ratio": IA_RATIO_DOMAIN,
    "infiltration_fraction": FRACTION_DOMAIN,
    "runoff_multiplier": NONNEGATIVE_DOMAIN,
    "groundwater_multiplier": NONNEGATIVE_DOMAIN,
    "r": NONNEGATIVE_DOMAIN,
    "k": NONNEGATIVE_DOMAIN,
    "ls": NONNEGATIVE_DOMAIN,
    "c": NONNEGATIVE_DOMAIN,
    "p": NONNEGATIVE_DOMAIN,
    "sdr": FRACTION_DOMAIN,
    "sediment_multiplier": NONNEGATIVE_DOMAIN,
    "sediment_delivery_multiplier": NONNEGATIVE_DOMAIN,
    "sediment_n_pct": PERCENT_DOMAIN,
    "sediment_p_pct": PERCENT_DOMAIN,
    "enrichment_ratio": NONNEGATIVE_DOMAIN,
    "fraction_subsurface_shallow": FRACTION_DOMAIN,
}
_SAMPLING_KIND_PHYSICAL_DOMAINS: Dict[str, PhysicalDomain] = {
    "efficiency": EFFICIENCY_DOMAIN,
    "load_rate": NONNEGATIVE_DOMAIN,
    "nonnegative": NONNEGATIVE_DOMAIN,
    "fraction": FRACTION_DOMAIN,
    "cn": CN_DOMAIN,
    "ia_ratio": IA_RATIO_DOMAIN,
    "percent": PERCENT_DOMAIN,
    "positive": POSITIVE_DOMAIN,
}


# Consolidated input schema
@dataclass(frozen=True)
class InputVariableSpec:
    """Description of one user-facing ``input_*`` table."""

    name: str
    table: str
    key_columns: Tuple[str, ...]
    kind: str
    required_modes: Tuple[str, ...] = ()
    canonical_units: str | None = None
    default_row_allowed: bool = False


def _build_input_variable_specs() -> Dict[str, InputVariableSpec]:
    registry: Dict[str, InputVariableSpec] = {}
    for parameter, table in PARCEL_PARAMETER_INPUT_TABLES.items():
        kind = (
            "classification"
            if parameter in PLET_CLASSIFICATION_PARAMETERS
            else "numeric_distribution"
        )
        required_modes = ("plet_rusle",) if parameter in PLET_REQUIRED_PARAMETERS else ()
        registry[table] = InputVariableSpec(
            name=parameter,
            table=table,
            key_columns=("pid",),
            kind=kind,
            required_modes=required_modes,
            default_row_allowed=True,
        )
    registry[GPKG_INPUT_SELECTION_WEIGHT] = InputVariableSpec(
        name="selection_weight",
        table=GPKG_INPUT_SELECTION_WEIGHT,
        key_columns=("pid",),
        kind="fixed_numeric",
        default_row_allowed=True,
    )
    registry[GPKG_INPUT_CURVE_NUMBER] = InputVariableSpec(
        name="cn",
        table=GPKG_INPUT_CURVE_NUMBER,
        key_columns=("land_cover", "hsg"),
        kind="numeric_distribution",
        required_modes=("plet_rusle",),
    )
    registry[GPKG_INPUT_INFILTRATION_FRACTION] = InputVariableSpec(
        name="infiltration_fraction",
        table=GPKG_INPUT_INFILTRATION_FRACTION,
        key_columns=("land_cover", "hsg"),
        kind="numeric_distribution",
        required_modes=("plet_rusle",),
    )
    registry[GPKG_INPUT_POLLUTANT_LOAD_RATE] = InputVariableSpec(
        name="pollutant_load_rate",
        table=GPKG_INPUT_POLLUTANT_LOAD_RATE,
        key_columns=("pid", "pollutant"),
        kind="numeric_distribution",
        required_modes=("statistical",),
        default_row_allowed=True,
    )
    registry[GPKG_INPUT_SURFACE_CONCENTRATION] = InputVariableSpec(
        name="surface_concentration",
        table=GPKG_INPUT_SURFACE_CONCENTRATION,
        key_columns=("pid", "pollutant"),
        kind="numeric_distribution",
        default_row_allowed=True,
    )
    registry[GPKG_INPUT_SUBSURFACE_CONCENTRATION] = InputVariableSpec(
        name="subsurface_concentration",
        table=GPKG_INPUT_SUBSURFACE_CONCENTRATION,
        key_columns=("pid", "pollutant"),
        kind="numeric_distribution",
        default_row_allowed=True,
    )
    for name, table in GPKG_DELIVERY_RATIO_TABLES.items():
        registry[table] = InputVariableSpec(
            name=name,
            table=table,
            key_columns=("pid", "oid"),
            kind="fixed_numeric",
            default_row_allowed=True,
        )
    return registry


INPUT_VARIABLE_SPECS: Dict[str, InputVariableSpec] = _build_input_variable_specs()
KNOWN_INPUT_TABLES = frozenset(INPUT_VARIABLE_SPECS)