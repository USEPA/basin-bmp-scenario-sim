import pandas as pd
from typing import Dict, List, Union, TYPE_CHECKING
if TYPE_CHECKING: from model import Model


from .constants import (
    COL_AREA_HA,
    COL_CPS,
    COL_PERIM_M,
    COL_PID,
    DATA_PARCELS,
    DATA_PARCEL_UP_MAP,
)


def _sample_yield(
    self: Model,
    parcel_idx: int, 
    pol_idx: int
    ) -> float:
    """Sample baseline pollutant yield for a parcel and pollutant index."""
    stats = self.pollutant_yield_stats[parcel_idx][pol_idx]
    if stats is None:
        raise KeyError(
            f"No pollutant yield stats found for pid={self.parcel_ids[parcel_idx]}, pollutant={self.pollutants[pol_idx]}"
        )
    return self._sample_from_stats(
        stats,
        kind="yield",
    )


def _sample_parcel_index(
    self: Model,
    ) -> int:
    """Choose a parcel index randomly from parcel selection probabilities."""
    idx = self.rng.choice(len(self.parcel_selection_ids), 
                          p=self.parcel_selection_probs)
    self.logger.debug(f"selected parcel idx={idx} with pid={self.parcel_selection_ids[idx]}")
    return idx


def _get_parcel_metadata(
    self: Model, 
    pid: Union[int, str]
    ) -> pd.Series:
    """Return parcel metadata for a given parcel ID, raising if missing."""
    sub = self.data[DATA_PARCELS]
    match = sub[sub[COL_PID].astype(str) == str(pid)]
    if match.empty:
        raise KeyError(
            f"Selected pid {pid} not found in parcels after clipping. "
            f"Ensure parcel_p PIDs exist in parcels and are within the domain."
        )
    return match.iloc[0]


def _get_parcel_up_list(
    self: Model, 
    pid: Union[int, str]
    ) -> List[str]:
    """Return the ordered list of up-gradient parcel IDs for the given parcel."""
    return list(self.data[DATA_PARCEL_UP_MAP].get(str(pid), []))


def _get_parcel_out_oids(
    self: Model, 
    parcel_idx: int
    ) -> List[str]:
    """Return the outlet IDs associated with a parcel index."""
    return list(self.parcel_out_oids[parcel_idx])


def _get_delivery_coeffs(
    self: Model, 
    pid: Union[int, str], 
    oid: Union[int, str]
    ) -> Dict[str, float]:
    """Get delivery coefficients for a parcel-to-outlet pair.

    Defaults to 1.0 when no explicit delivery ratios are supplied.
    """
    return self.delivery_coeffs.get(
        (str(pid), str(oid)),
        dict(sdr_f_to_s=1.0, sdr_s_to_o=1.0, ndr_f_to_s=1.0, ndr_s_to_o=1.0),
    )