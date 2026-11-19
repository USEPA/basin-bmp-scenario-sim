"""Graph helpers for normalized parcel-to-upstream drainage relationships.

The ``parcel_up`` table records *immediate* links from each downstream parcel
(``pid``) to an upstream parcel (``pid_up``). Its direction is intentionally
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
                    f"parcel_up references unknown modeled upstream pid {neighbor!r}"
                )
            kind = state.get(neighbor, 0)
            if kind == 1:
                cycle = path[positions[neighbor]:] + [neighbor]
                raise ValueError(
                    "parcel_up contains a directed cycle: " + " -> ".join(cycle)
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
        raise ValueError("parcel_up indices and hydrologic parcel IDs must have equal lengths")
    if not (0 <= parcel_idx < len(parcel_ids)):
        raise IndexError(f"Wetland parcel index {parcel_idx} is outside the hydrologic universe")
    visited = {parcel_idx}
    frontier = [parcel_idx]
    while frontier:
        next_layer: set[int] = set()
        for idx in frontier:
            for upstream_idx in immediate_upstream_indices[idx]:
                if upstream_idx < 0 or upstream_idx >= len(parcel_ids):
                    raise ValueError(f"parcel_up index {upstream_idx} is outside the hydrologic universe")
                if upstream_idx not in visited:
                    next_layer.add(upstream_idx)
        frontier = sorted(next_layer, key=lambda i: (pid_sort_key(parcel_ids[i]), i))
        visited.update(frontier)
        yield from frontier
