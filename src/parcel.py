"""Parcel lookup and routing helpers.

This module provides small helper functions used to sample parcel-level loads,
select parcels for BMP placement, and resolve routing and delivery metadata.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

from .constants import (
    COL_AREA_HA,
    COL_CPS,
    COL_PERIM_M,
    COL_PID,
    DATA_PARCELS,
    DATA_PARCEL_UP_MAP,
)


def _sample_load_rate(self: "Model", parcel_idx: int, pol_idx: int) -> float:
    """Sample a baseline areal pollutant load rate for one parcel/pollutant.

        In the current annual model the returned value has units of kg/ha/yr.
        Dynamic implementations can retain this helper name while changing the
        rate's time basis explicitly in the timestep state.

        Parameters
        ----------
        parcel_idx : int
            Zero-based index of the parcel.
        pol_idx : int
            Zero-based index of the pollutant.

        Returns
        -------
        float
            Sampled baseline areal pollutant load rate.

        Raises
        ------
        KeyError
            If no load-rate statistics are available for the requested parcel and pollutant.
        
    """
    stats = self.pollutant_load_rate_stats[parcel_idx][pol_idx]
    if stats is None:
        raise KeyError(
            "No pollutant load-rate stats found for "
            f"pid={self.parcel_ids[parcel_idx]}, pollutant={self.pollutants[pol_idx]}"
        )
    return self._sample_from_stats(stats, kind="load_rate")




def _sample_parcel_index(self: "Model") -> int:
    """Sample the next parcel to receive a BMP attempt.

        Returns
        -------
        int
            Index of the selected parcel.
        
    """
    idx = self.rng.choice(len(self.parcel_selection_ids), p=self.parcel_selection_probs)
    self.logger.verbose(f"selected parcel idx={idx} with pid={self.parcel_selection_ids[idx]}")
    return idx


def _get_parcel_metadata(self: "Model", pid: Union[int, str]) -> pd.Series:
    """Return metadata for one parcel.

        Parameters
        ----------
        pid : Union[int, str]
            Parcel identifier.

        Returns
        -------
        pd.Series
            Metadata row for the requested parcel.

        Raises
        ------
        KeyError
            If the requested parcel is not present after domain clipping.
        
    """
    sub = self.data[DATA_PARCELS]
    match = sub[sub[COL_PID].astype(str) == str(pid)]
    if match.empty:
        raise KeyError(
            f"Selected pid {pid} not found in parcels after clipping. "
            f"Ensure parcel_p PIDs exist in parcels and are within the domain."
        )
    return match.iloc[0]


def _get_parcel_up_list(self: "Model", pid: Union[int, str]) -> List[str]:
    """Return upstream parcel IDs for one parcel.

        Parameters
        ----------
        pid : Union[int, str]
            Parcel identifier.

        Returns
        -------
        List[str]
            Upstream parcel identifiers.
        
    """
    return list(self.data[DATA_PARCEL_UP_MAP][str(pid)])


def _get_parcel_out_oids(self: "Model", parcel_idx: int) -> List[str]:
    """Return outlet IDs connected to one parcel.

        Parameters
        ----------
        parcel_idx : int
            Zero-based index of the parcel.

        Returns
        -------
        List[str]
            Outlet identifiers connected to the parcel.
        
    """
    return list(self.parcel_out_oids[parcel_idx])


def _get_delivery_coeffs(self: "Model", pid: Union[int, str], oid: Union[int, str]) -> Dict[str, float]:
    """Return delivery coefficients for a parcel-to-outlet path.

        Parameters
        ----------
        pid : Union[int, str]
            Parcel identifier.
        oid : Union[int, str]
            Outlet identifier.

        Returns
        -------
        Dict[str, float]
            Delivery coefficients keyed by pollutant.
        
    """
    return self.delivery_coeffs[(str(pid), str(oid))]
