"""Immediate D8 parcel edges and transactional ``parcel_up`` GeoPackage I/O.

``parcel_up(pid, pid_up)`` records *one direct upstream connection per row*.
The complete upstream catchment is deliberately NOT stored: the watershed
model traverses the immediate upstream network when it needs ancestors.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping

import numpy as np


def _integer_id(value: object) -> int:
    """Convert a parcel ID into the integer key required by the model schema."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid boolean parcel ID: {value!r}")
    text = str(value).strip()
    if not text or not (text.isdecimal() or (text.startswith('-') and text[1:].isdecimal())):
        raise ValueError(f"Model parcel_up requires integer parcel IDs; received {value!r}")
    return int(text)


def direct_edges_from_d8(
    pid_raster: np.ndarray,
    next_r: np.ndarray,
    next_c: np.ndarray,
    code_to_pid: Mapping[int, object],
) -> list[tuple[int, int]]:
    """Derive ``(downstream pid, directly upstream pid)`` from D8 flow paths.

    A parcel connects to the *first different parcel encountered downstream*.
    Gaps of unlabeled DEM cells are followed, but an intervening parcel stops
    the trace: A -> B -> C produces (B,A) and (C,B), **not** (C,A).
    Multiple parcel boundary outlets are allowed; edges are deduplicated.
    D8 loops in unlabeled background terminate without creating spurious edges.
    """
    if (pid_raster.shape != next_r.shape or pid_raster.shape != next_c.shape
            or pid_raster.ndim != 2):
        raise ValueError("Parcel labels and D8 pointer rasters must be matching 2D arrays")
    nrows, ncols = pid_raster.shape
    flat_labels = pid_raster.ravel()
    nr = next_r.ravel()
    nc = next_c.ravel()
    gap_cache: dict[int, int] = {}
    edges: set[tuple[int, int]] = set()

    def next_parcel_in_gap(index: int) -> int:
        """First nonzero parcel downstream from a background cell, or zero."""
        path: list[int] = []
        positions: set[int] = set()
        while True:
            if index in gap_cache:
                result = gap_cache[index]
                break
            code = int(flat_labels[index])
            if code:
                result = code
                break
            if index in positions:
                result = 0  # an accidental flat-cell D8 cycle
                break
            positions.add(index)
            path.append(index)
            r, c = int(nr[index]), int(nc[index])
            if r < 0 or r >= nrows or c < 0 or c >= ncols:
                result = 0
                break
            index = r * ncols + c
        for visited in path:
            gap_cache[visited] = result
        return result

    # Row-wise array operations avoid materializing another giant 10m DEM-sized
    # cell-index array. Only actual parcel exit cells need Python traversal.
    for r in range(nrows):
        labels = pid_raster[r]
        candidates = np.flatnonzero((labels > 0) & (next_r[r] >= 0)
                                        & (next_r[r] < nrows)
                                        & (next_c[r] >= 0)
                                        & (next_c[r] < ncols))
        if not candidates.size:
            continue
        target_r = next_r[r, candidates]
        target_c = next_c[r, candidates]
        downstream_codes = pid_raster[target_r, target_c]
        differing = downstream_codes != labels[candidates]
        for c, downstream in zip(candidates[differing], downstream_codes[differing]):
            source_code = int(labels[c])
            downstream_code = int(downstream)
            if downstream_code == 0:
                target_index = int(next_r[r, c]) * ncols + int(next_c[r, c])
                downstream_code = next_parcel_in_gap(target_index)
            if downstream_code and downstream_code != source_code:
                if source_code not in code_to_pid or downstream_code not in code_to_pid:
                    raise ValueError("D8 raster includes a parcel code without a parcel ID")
                upstream_id = _integer_id(code_to_pid[source_code])
                downstream_id = _integer_id(code_to_pid[downstream_code])
                edges.add((downstream_id, upstream_id))

    return sorted(edges)


def validate_direct_edges(
    edges: Iterable[tuple[int, int]], known_parcel_ids: Iterable[object]
) -> list[tuple[int, int]]:
    """Validate identifiers, remove duplicate edges, and reject directed cycles.

    Uses an iterative graph traversal so long drainage chains cannot exceed
    Python's recursion depth. Source and target parcel IDs must be known.
    """
    ids = [_integer_id(pid) for pid in known_parcel_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Parcel IDs are not unique after integer normalization")
    known = set(ids)
    relations: set[tuple[int, int]] = set()
    adjacency: dict[int, list[int]] = {pid: [] for pid in known}
    for downstream, upstream in edges:
        downstream, upstream = _integer_id(downstream), _integer_id(upstream)
        if downstream not in known or upstream not in known:
            raise ValueError(
                f"parcel_up references unknown parcel IDs: {downstream}, {upstream}"
            )
        if downstream == upstream:
            raise ValueError(f"parcel_up contains a self-link / directed cycle: {downstream}")
        if (downstream, upstream) not in relations:
            relations.add((downstream, upstream))
            adjacency[downstream].append(upstream)

    # DFS matching the model's downstream -> immediate-upstream orientation.
    state: dict[int, int] = {}  # 1 = current traversal, 2 = completed
    for start in sorted(known):
        if state.get(start):
            continue
        state[start] = 1
        path = [start]
        positions = {start: 0}
        stack = [(start, iter(sorted(adjacency[start])))]
        while stack:
            pid, neighbors = stack[-1]
            try:
                up = next(neighbors)
            except StopIteration:
                state[pid] = 2
                stack.pop()
                path.pop()
                positions.pop(pid)
                continue
            if state.get(up) == 1:
                cycle = path[positions[up]:] + [up]
                raise ValueError("parcel_up contains a directed cycle: " +
                                 " -> ".join(map(str, cycle)))
            if state.get(up, 0) == 0:
                state[up] = 1
                positions[up] = len(path)
                path.append(up)
                stack.append((up, iter(sorted(adjacency[up]))))
    return sorted(relations)


def _parcels_pid_is_unique(conn: sqlite3.Connection) -> bool:
    """Whether SQLite permits FOREIGN KEY references to ``parcels(pid)``."""
    info = conn.execute('PRAGMA table_info("parcels")').fetchall()
    if any(col[1] == 'pid' and col[5] for col in info):
        return True
    # Other polygon GeoPackages sometimes use fid as primary key and store pid
    # as an attribute without a uniqueness constraint; still support these.
    for index in conn.execute('PRAGMA index_list("parcels")'):
        if not index[2]:
            continue
        idx_name = index[1].replace('"', '""')
        columns = conn.execute(f'PRAGMA index_info("{idx_name}")').fetchall()
        if [col[2] for col in columns] == ['pid']:
            return True
    return False


def write_gpkg_parcel_up(
    gpkg: str | Path, edges: Iterable[tuple[int, int]], known_parcel_ids: Iterable[object]
) -> int:
    """Atomically replace ``parcel_up`` with the model's normalized edge table.

    The transaction rolls back the old table, content registration and rows if
    a create, insert or metadata write fails. Other GeoPackage layers and tables
    are not modified. Use an external copy/backup if the previous graph must
    remain available after a successful replacement.
    """
    gpkg = Path(gpkg)
    if gpkg.suffix.lower() != '.gpkg' or not gpkg.is_file():
        raise ValueError("A local .gpkg parcel file is required to update parcel_up")
    rows = validate_direct_edges(edges, known_parcel_ids)
    with sqlite3.connect(gpkg, timeout=30) as conn:
        conn.execute('PRAGMA foreign_keys=ON')
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='parcels'").fetchone():
            raise ValueError("The GeoPackage requires a 'parcels' geometry layer")
        info = conn.execute('PRAGMA table_info("parcels")').fetchall()
        if 'pid' not in {col[1] for col in info}:
            raise ValueError("The GeoPackage parcels layer must have a pid column")
        contents = conn.execute("SELECT data_type FROM gpkg_contents WHERE table_name='parcel_up'").fetchone()
        if contents is not None and contents[0] != 'attributes':
            raise ValueError("Cannot replace parcel_up: gpkg_contents does not describe an attributes table")
        existing = conn.execute("SELECT type FROM sqlite_master WHERE name='parcel_up'").fetchone()
        if existing is not None and existing[0] != 'table':
            raise ValueError("Cannot replace parcel_up: existing object is not a table")
        # A DROP TABLE could otherwise cascade into an unrelated table if it
        # declares a foreign key to parcel_up. Fail rather than mutate others.
        if existing:
            for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                if table == 'parcel_up' or table.startswith('sqlite_'):
                    continue
                safe_table = table.replace('"', '""')
                refs = conn.execute(f'PRAGMA foreign_key_list("{safe_table}")').fetchall()
                if any(str(fk[2]).lower() == 'parcel_up' for fk in refs):
                    raise ValueError(
                        f"Cannot replace parcel_up: table {table!r} has a foreign key "
                        "referencing parcel_up"
                    )
        use_fk = _parcels_pid_is_unique(conn)
        if not use_fk:
            logging.warning(
                "parcels.pid is not declared unique; writing normalized parcel_up "
                "without SQL foreign keys (the model's identifier and graph "
                "validation still applies)."
            )
        fk_clause = (', FOREIGN KEY(pid) REFERENCES parcels(pid), '
                     'FOREIGN KEY(pid_up) REFERENCES parcels(pid)' if use_fk else '')
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('DROP TABLE IF EXISTS "parcel_up"')
        conn.execute('CREATE TABLE "parcel_up" ('
                     'id INTEGER PRIMARY KEY AUTOINCREMENT, '
                     'pid INTEGER NOT NULL, pid_up INTEGER NOT NULL, '
                     'UNIQUE(pid, pid_up)' + fk_clause + ')')
        conn.executemany('INSERT INTO "parcel_up" (pid, pid_up) VALUES (?, ?)', rows)
        if contents is None:
            conn.execute(
                "INSERT INTO gpkg_contents (table_name, data_type, identifier, description, last_change) "
                "VALUES ('parcel_up', 'attributes', 'parcel_up', '', "
                "strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
            )
        else:
            conn.execute(
                "UPDATE gpkg_contents SET last_change=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE table_name='parcel_up'"
            )
        if use_fk:
            errors = conn.execute('PRAGMA foreign_key_check("parcel_up")').fetchall()
            if errors:
                raise ValueError(f"parcel_up failed foreign-key checks: {errors[:5]}")
        # Context manager commits on success; rolls back on any exception.
    return len(rows)


def write_normalized_csv(path: str | Path, edges: Iterable[tuple[int, int]]) -> None:
    """Optionally export the same one-edge-per-row model table to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(('pid', 'pid_up'))
        writer.writerows(sorted(set(edges)))
