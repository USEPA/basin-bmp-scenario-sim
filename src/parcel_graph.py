"""Graph helpers for normalized parcel-to-upstream drainage relationships.

The ``parcel_down`` table records each source parcel (`pid`) and its immediate
downstream parcel (`pid_down`). The model reverses its direction in memory; it is
preserved: watershed traversal follows those links away from a wetland host.
"""

from __future__ import annotations

from typing import Iterator, Mapping, Sequence


def pid_sort_key(pid: object) -> tuple[int, int | str, str]:
    """Sort integer-like parcel IDs numerically; others lexicographically."""
    text = str(pid)
    if text.lstrip("-").isdigit():
        return (0, int(text), text)
    return (1, text, text)


def validate_acyclic_upstream_map(upstream_by_pid: Mapping[str, Sequence[str]]) -> None:
    """Reject a directed cycle in modeled immediate upstream connections.

    Uses iterative depth-first traversal rather than recursion so very long
    upstream chains do not hit the Python recursion limit. The error identifies
    an actual cycle (rather than a downstream parcel that depends on one).
    """
    # 0 = unseen; 1 = in the current traversal path; 2 = complete.
    state: dict[str, int] = {}
    for start in sorted(upstream_by_pid, key=pid_sort_key):
        if state.get(start, 0):
            continue
        state[start] = 1
        path = [start]
        positions = {start: 0}
        stack = [(start, iter(sorted(upstream_by_pid[start], key=pid_sort_key)))]
        while stack:
            pid, upstream = stack[-1]
            try:
                neighbor = next(upstream)
            except StopIteration:
                state[pid] = 2
                stack.pop()
                path.pop()
                positions.pop(pid)
                continue
            if neighbor not in upstream_by_pid:
                raise ValueError(
                    f"parcel_down references unknown modeled upstream pid {neighbor!r}"
                )
            kind = state.get(neighbor, 0)
            if kind == 1:
                cycle = path[positions[neighbor]:] + [neighbor]
                raise ValueError(
                    "parcel_down contains a directed cycle: " + " -> ".join(cycle)
                )
            if kind == 0:
                state[neighbor] = 1
                positions[neighbor] = len(path)
                path.append(neighbor)
                stack.append((neighbor, iter(sorted(upstream_by_pid[neighbor], key=pid_sort_key))))


def iter_full_upstream_by_distance(
    parcel_idx: int,
    immediate_upstream_indices: Sequence[Sequence[int]],
    parcel_ids: Sequence[str],
) -> Iterator[int]:
    """Yield every unique upstream parcel in increasing shortest-hop distance.

    For equal distances, increasing parcel ID breaks ties (numeric ordering for
    integer-like PIDs, lexical ordering otherwise). A shared upstream parcel is
    yielded once, including when two drainage branches converge. The wetland
    host is never returned. This iterates the modeled hydrologic parcel
    universe, not the BMP-selection subset.
    """
    if len(immediate_upstream_indices) != len(parcel_ids):
        raise ValueError("upstream parcel indices and hydrologic parcel IDs must have equal lengths")
    if not (0 <= parcel_idx < len(parcel_ids)):
        raise IndexError(f"Wetland parcel index {parcel_idx} is outside the hydrologic universe")
    visited = {parcel_idx}
    frontier = [parcel_idx]
    while frontier:
        next_layer: set[int] = set()
        for idx in frontier:
            for upstream_idx in immediate_upstream_indices[idx]:
                if upstream_idx < 0 or upstream_idx >= len(parcel_ids):
                    raise ValueError(f"upstream parcel index {upstream_idx} is outside the hydrologic universe")
                if upstream_idx not in visited:
                    next_layer.add(upstream_idx)
        frontier = sorted(next_layer, key=lambda i: (pid_sort_key(parcel_ids[i]), i))
        visited.update(frontier)
        yield from frontier


def validate_acyclic_downstream_map(downstream: dict[str, str | None]) -> None:
    """Validate a functional source→downstream drainage graph iteratively."""
    complete: set[str] = set()
    for source in downstream:
        if source in complete:
            continue
        path: list[str] = []
        positions: dict[str, int] = {}
        at: str | None = source
        while at is not None and at not in complete:
            if at not in downstream:
                raise ValueError(f"parcel_down references unknown downstream pid {at}")
            if at in positions:
                cycle = path[positions[at]:] + [at]
                raise ValueError('parcel_down contains a directed cycle: ' + ' -> '.join(cycle))
            positions[at] = len(path)
            path.append(at)
            at = downstream[at]
        complete.update(path)
