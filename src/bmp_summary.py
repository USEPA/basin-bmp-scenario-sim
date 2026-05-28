"""Generate statistical summaries of BMPs by type for each scenario."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import defaultdict

from .constants import (
    BMP_CPS_NAME_MAP,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_REMOVED_PREFIX,
    OUTPUT_TREATED_PREFIX,
    OUTPUT_WETLAND_AREA,
)


class BMPSummaryCollector:
    """Collect statistics from BMPs as they are generated during scenario execution."""
    
    def __init__(self, pollutants: List[str], scenario_id: int) -> None:
        """Initialize collector for a scenario."""
        self.pollutants = pollutants
        self.scenario_id = scenario_id
        
        # Group data by CPS type
        # Structure: {cps: {"records": [...], "attributes": {...}}}
        self.bmp_by_cps: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "records": [],
            "attributes": defaultdict(list),
        })
    
    def add_bmp_record(
        self,
        bmp_record: Dict[str, Any],
        pid_baseline_yields: Dict[str, float],
    ) -> None:
        """
        Add a BMP record to the collector.
        
        Parameters
        ----------
        bmp_record : Dict[str, Any]
            The BMP record dictionary from scenario simulation
        pid_baseline_yields : Dict[str, float]
            Baseline yield for each pollutant for this parcel
        """
        cps = int(bmp_record["cps"])
        
        self.bmp_by_cps[cps]["records"].append(bmp_record)
        
        # Collect type-specific attributes
        if cps == 656:  # Constructed Wetland
            wetland_area = bmp_record.get(OUTPUT_WETLAND_AREA)
            catchment_ratio = bmp_record.get(OUTPUT_CATCHMENT_RATIO)
            if wetland_area is not None:
                self.bmp_by_cps[cps]["attributes"]["wetland_area_ha"].append(float(wetland_area))
            if catchment_ratio is not None:
                self.bmp_by_cps[cps]["attributes"]["catchment_ratio"].append(float(catchment_ratio))
                
        elif cps == 412:  # Grassed Waterway
            buffer_area = bmp_record.get(OUTPUT_BUFFER_AREA)
            linear_length = bmp_record.get(OUTPUT_LINEAR_LENGTH)
            if buffer_area is not None:
                self.bmp_by_cps[cps]["attributes"]["buffer_area_ha"].append(float(buffer_area))
            if linear_length is not None:
                self.bmp_by_cps[cps]["attributes"]["linear_length_m"].append(float(linear_length))
        
        # Store PID and baseline yields for efficiency computation
        self.bmp_by_cps[cps]["attributes"]["pid"].append(str(bmp_record.get("pid", "")))
        for pol in self.pollutants:
            baseline = pid_baseline_yields.get(pol, 0.0)
            self.bmp_by_cps[cps]["attributes"][f"baseline_{pol}"].append(baseline)
    
    def generate_summary_dataframe(self) -> pd.DataFrame:
        """Generate summary statistics DataFrame."""
        summaries: List[Dict[str, Any]] = []
        
        for cps in sorted(self.bmp_by_cps.keys()):
            data = self.bmp_by_cps[cps]
            bmp_records = data["records"]
            attrs = data["attributes"]
            
            summary: Dict[str, Any] = {
                "scenario": self.scenario_id,
                "cps": cps,
                "cps_name": BMP_CPS_NAME_MAP.get(cps, f"CPS {cps}"),
                "bmp_count": len(bmp_records),
            }
            
            # Type-specific attributes (e.g., wetland area, buffer area)
            for attr_name in ["wetland_area_ha", "catchment_ratio", "buffer_area_ha", "linear_length_m"]:
                if attr_name in attrs and len(attrs[attr_name]) > 0:
                    values = np.array(attrs[attr_name])
                    stats = _compute_statistics(values)
                    for stat_name, stat_val in stats.items():
                        summary[f"{attr_name}_{stat_name}"] = stat_val
            
            # Treated loads per pollutant
            for pol in self.pollutants:
                treated_col = f"{OUTPUT_TREATED_PREFIX}{pol}"
                treated_loads = []
                for rec in bmp_records:
                    treated = rec.get(treated_col)
                    if treated is not None:
                        treated_loads.append(float(treated))
                
                if treated_loads:
                    values = np.array(treated_loads)
                    stats = _compute_statistics(values)
                    for stat_name, stat_val in stats.items():
                        summary[f"treated_{pol}_{stat_name}"] = stat_val
            
            # Removed loads per pollutant
            for pol in self.pollutants:
                removed_col = f"{OUTPUT_REMOVED_PREFIX}{pol}"
                removed_loads = []
                for rec in bmp_records:
                    removed = rec.get(removed_col)
                    if removed is not None:
                        removed_loads.append(float(removed))
                
                if removed_loads:
                    values = np.array(removed_loads)
                    stats = _compute_statistics(values)
                    for stat_name, stat_val in stats.items():
                        summary[f"removed_{pol}_{stat_name}"] = stat_val
            
            # Efficiency coefficients per pollutant
            for pol in self.pollutants:
                treated_col = f"{OUTPUT_TREATED_PREFIX}{pol}"
                efficiencies = []
                
                for idx, rec in enumerate(bmp_records):
                    treated = rec.get(treated_col)
                    if treated is not None:
                        treated = float(treated)
                        baseline = attrs[f"baseline_{pol}"][idx]
                        eff = _compute_efficiency(treated, baseline)
                        if eff is not None:
                            efficiencies.append(eff)
                
                if efficiencies:
                    values = np.array(efficiencies)
                    stats = _compute_statistics(values)
                    for stat_name, stat_val in stats.items():
                        summary[f"efficiency_{pol}_{stat_name}"] = stat_val
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def _compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive statistics for a distribution of values."""
    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    
    # Filter out NaN values
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    
    return {
        "count": len(valid_values),
        "mean": float(np.mean(valid_values)),
        "std": float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else np.nan,
        "min": float(np.min(valid_values)),
        "p25": float(np.percentile(valid_values, 25)),
        "p50": float(np.percentile(valid_values, 50)),
        "p75": float(np.percentile(valid_values, 75)),
        "max": float(np.max(valid_values)),
    }


def _compute_efficiency(
    treated: float, 
    baseline_yield: float
) -> Optional[float]:
    """Compute efficiency coefficient from treated load and baseline yield."""
    if baseline_yield <= 0:
        return None
    return min(1.0, treated / baseline_yield)
