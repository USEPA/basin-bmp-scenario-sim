"""Shared parcel/BMP eligibility for deterministic and stochastic placement.

Eligibility is a property of a parcel/CPS combination, independent of the
algorithm that chooses among eligible combinations. Keep additional siting
rules in ``eligible_indices`` so both selection modes use the same rules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


class BMPEligibilityPool:
    """Track feasible placements and exclude exhausted parcels from sampling.

    Parameters
    ----------
    selection_pids : Sequence[str]
        The BMP-placement universe (not the full hydrologic parcel universe).
    bmp_cps : Sequence[int]
        Configured CPS codes, in the same order as ``bmp_probabilities``.
    bmp_probabilities : Sequence[float]
        Relative conditional weights for stochastic BMP selection.
    feedlot_bmp_cps : Iterable[int]
        CPS codes restricted to AFO host parcels.
    afo_host_pids : Iterable[str]
        Parcel IDs hosting AFOs.
    parcel_probabilities : Sequence[float]
        Relative weights for stochastic parcel selection.
    deterministic_parcels, deterministic_bmps : bool
        Selection strategies; eligibility rules are shared between them.
    """

    def __init__(
        self,
        selection_pids: Sequence[str],
        bmp_cps: Sequence[int],
        bmp_probabilities: Sequence[float],
        feedlot_bmp_cps: Iterable[int],
        afo_host_pids: Iterable[str],
        parcel_probabilities: Sequence[float],
        *,
        deterministic_parcels: bool = False,
        deterministic_bmps: bool = False,
    ) -> None:
        self.selection_pids = tuple(str(pid) for pid in selection_pids)
        self.bmp_cps = tuple(int(cps) for cps in bmp_cps)
        self.bmp_probabilities = np.asarray(bmp_probabilities, dtype=float)
        self.parcel_probabilities = np.asarray(parcel_probabilities, dtype=float)
        if len(self.bmp_probabilities) != len(self.bmp_cps):
            raise ValueError("BMP CPS codes and selection probabilities must align")
        if len(self.parcel_probabilities) != len(self.selection_pids):
            raise ValueError("Parcel IDs and selection probabilities must align")
        self.feedlot_bmp_cps = {int(cps) for cps in feedlot_bmp_cps}
        self.afo_host_pids = {str(pid) for pid in afo_host_pids}
        self.deterministic_parcels = bool(deterministic_parcels)
        self.deterministic_bmps = bool(deterministic_bmps)
        self._deterministic_cps_order = tuple(sorted(set(self.bmp_cps)))
        self.applied_by_pid: dict[str, set[int]] = defaultdict(set)
        self._selection_indices_by_pid: dict[str, list[int]] = defaultdict(list)
        for idx, pid in enumerate(self.selection_pids):
            self._selection_indices_by_pid[pid].append(idx)
        self._active = np.asarray(
            [
                self._parcel_has_selection_weight(idx)
                and bool(self.selectable_indices(pid))
                for idx, pid in enumerate(self.selection_pids)
            ],
            dtype=bool,
        )

    def eligible_indices(self, pid: str) -> list[int]:
        """Indices of CPS codes physically eligible on ``pid``.

        A CPS cannot be repeated on a parcel, and feedlot-only BMPs require
        an AFO host parcel. This is the *only* place for siting rules.
        """
        pid = str(pid)
        applied = self.applied_by_pid[pid]
        return [
            idx for idx, cps in enumerate(self.bmp_cps)
            if cps not in applied
            and (cps not in self.feedlot_bmp_cps or pid in self.afo_host_pids)
        ]

    def selectable_indices(self, pid: str) -> list[int]:
        """Eligible CPS indices with positive probability if BMPs are random.

        A zero BMP probability does not make a practice physically ineligible:
        deterministic BMP order may still select it. In stochastic mode it
        cannot be drawn, so it cannot keep a parcel active.
        """
        eligible = self.eligible_indices(pid)
        if self.deterministic_bmps:
            return eligible
        return [idx for idx in eligible if np.isfinite(self.bmp_probabilities[idx])
                and self.bmp_probabilities[idx] > 0.0]

    def _parcel_has_selection_weight(self, selection_index: int) -> bool:
        if self.deterministic_parcels:
            return True
        weight = self.parcel_probabilities[selection_index]
        return bool(np.isfinite(weight) and weight > 0.0)

    def active_selection_indices(self) -> np.ndarray:
        """Indices of selectable parcels, in original input-table order."""
        return np.flatnonzero(self._active)

    def select_bmp(self, pid: str, rng: np.random.Generator, cursor: int) -> tuple[int, int]:
        """Select a BMP after parcel selection; return CPS and updated cursor."""
        indices = self.selectable_indices(pid)
        if not indices:
            raise ValueError(f"Parcel {pid} has no selectable BMPs")
        if self.deterministic_bmps:
            eligible = {self.bmp_cps[i] for i in indices}
            for offset in range(len(self._deterministic_cps_order)):
                idx = (cursor + offset) % len(self._deterministic_cps_order)
                cps = self._deterministic_cps_order[idx]
                if cps in eligible:
                    return cps, idx + 1
            raise AssertionError("Active BMP eligibility pool has no deterministic choice")
        probabilities = self.bmp_probabilities[indices]
        probabilities = probabilities / probabilities.sum()
        selected = int(rng.choice(len(indices), p=probabilities))
        return self.bmp_cps[indices[selected]], cursor

    def record_placement(self, pid: str, cps: int) -> None:
        """Consume a parcel/CPS combination after successful implementation."""
        pid, cps = str(pid), int(cps)
        if cps not in {self.bmp_cps[i] for i in self.selectable_indices(pid)}:
            raise ValueError(f"CPS {cps} is not selectable on parcel {pid}")
        self.applied_by_pid[pid].add(cps)
        still_active = bool(self.selectable_indices(pid))
        for idx in self._selection_indices_by_pid.get(pid, ()):
            self._active[idx] = self._parcel_has_selection_weight(idx) and still_active
