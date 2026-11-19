"""WhiteboxTools D8 routing and evidence-based, acyclic parcel connectivity.

The DEM is conditioned using WhiteboxTools, not custom uphill-flow code.
Each raster cell originating inside a parcel contributes exactly once to the
first *different* parcel encountered downstream (or to the unmodeled outlet).
Thus area weights never include upstream parcels' throughflow. The resulting
single-outlet-per-parcel graph is a documented simplification, not a statement
that a parcel cannot physically drain in several directions.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
import rasterio

try:
    from .rasterio_compat import read_band, read_mask
except ImportError:  # Direct execution of utils/create_parcel_down.py
    from rasterio_compat import read_band, read_mask

# Clockwise from NE, as documented for the default, non-ESRI Whitebox D8
# pointer convention. 0 means no downstream cell, not north/south/etc.
WHITEBOX_D8 = {
    1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
    16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0),
}


def whitebox_d8_to_next(pointer: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Decode Whitebox (NOT ESRI) pointers into flat downstream cell indices.

    Invalid values or directions into masked/no-data cells cause an error;
    a flow pointing outside the raster becomes an outlet (-1). Index output
    uses int32 and therefore requires fewer than 2**31 raster cells.
    """
    pointer = np.asarray(pointer)
    valid = np.asarray(valid, dtype=bool)
    if pointer.ndim != 2 or pointer.shape != valid.shape:
        raise ValueError('Whitebox D8 pointers and valid-cell mask must be matching 2D grids')
    nrows, ncols = pointer.shape
    if pointer.size >= np.iinfo(np.int32).max:
        raise ValueError('Whitebox D8 grid exceeds int32 routing capacity')
    codes = np.unique(pointer[valid])
    bad = [float(v) for v in codes if not np.isfinite(v) or v not in (0, *WHITEBOX_D8)]
    if bad:
        raise ValueError(f'Invalid Whitebox D8 pointer code(s) {bad[:8]}; '
                         'check Whitebox vs ESRI pointer encoding and NoData values')
    out = np.full(pointer.shape, -1, dtype=np.int32)
    for code, (dr, dc) in WHITEBOX_D8.items():
        r, c = np.where((pointer == code) & valid)
        rr, cc = r + dr, c + dc
        inside = (rr >= 0) & (rr < nrows) & (cc >= 0) & (cc < ncols)
        if np.any(inside):
            # Internal flow into NoData is invalid (a raster-hole artifact).
            if not np.all(valid[rr[inside], cc[inside]]):
                raise ValueError(f'Whitebox D8 pointer {code} routes into NoData')
            out[r[inside], c[inside]] = rr[inside] * ncols + cc[inside]
    return out.reshape(-1)


def validate_cell_dag(next_flat: np.ndarray) -> None:
    """Topologically verify every raster pointer; fail on a cell-level cycle.

    Iterative, O(cells) and recursion-free; the graph can contain many
    different headwaters and outlets, but no directed cycles.
    """
    nxt = np.asarray(next_flat).reshape(-1)
    n = nxt.size
    if np.any((nxt < -1) | (nxt >= n)):
        raise ValueError('Raster D8 pointer leads outside its cell index domain')
    indegree = np.bincount(nxt[nxt >= 0], minlength=n).astype(np.int32)
    heads = np.flatnonzero(indegree == 0)
    queue = np.empty(n, dtype=np.int32)
    tail = heads.size
    queue[:tail] = heads
    head = 0
    while head < tail:
        node = int(queue[head])
        head += 1
        down = int(nxt[node])
        if down >= 0:
            indegree[down] -= 1
            if indegree[down] == 0:
                queue[tail] = down
                tail += 1
    if head < n:
        first = int(np.flatnonzero(indegree > 0)[0])
        path, seen = [], {}
        at = first
        while at not in seen and at >= 0:
            seen[at] = len(path)
            path.append(at)
            at = int(nxt[at])
        cycle = path[seen.get(at, 0):] + ([at] if at >= 0 else [])
        raise ValueError('Whitebox D8 raster contains a directed cell-level cycle: '
                         + ' -> '.join(map(str, cycle[:12])))


def _check_wbt_result(result: Any, label: str, output: Path) -> None:
    # The WhiteboxTools Python interface returns 0 on success.
    if result not in (0, None) or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(
            f'WhiteboxTools {label} failed (status={result!r}); expected {output}. '
            'Inspect WhiteboxTools output and raster geometry.'
        )


def run_whitebox_d8(
    dem_file: str | Path,
    output_dir: str | Path,
    *,
    method: str = 'breach',
    breach_dist: int = 100,
    exe: str | None = None,
) -> Path:
    """Run WhiteboxTools depression conditioning followed by D8Pointer.

    ``breach`` uses BreachDepressionsLeastCost with fill=True for any unbreached
    sinks; ``fill`` uses FillDepressions. Both generate the default Whitebox
    pointer convention (``esri_pntr=False``). No custom uphill fallback.

    ``exe`` accepts an existing WhiteboxTools executable when the Python
    frontend is not installed. Without ``exe``, the ``whitebox`` Python
    frontend must be installed (``python -m pip install whitebox``).
    """
    dem_file = Path(dem_file).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dem_file.is_file():
        raise ValueError(f'DEM not found: {dem_file}')
    if method not in {'breach', 'fill'}:
        raise ValueError(f'Unknown Whitebox conditioning method: {method!r}')
    if not (1 <= int(breach_dist) <= 100000):
        raise ValueError('--whitebox-breach-dist must be between 1 and 100000 cells')
    conditioned = output_dir / 'whitebox_conditioned_dem.tif'
    pointer = output_dir / 'whitebox_d8_pointer.tif'
    # Never let an old output from a previous successful invocation mask a
    # new WhiteboxTools failure that returned no file.
    for previous in (conditioned, pointer):
        previous.unlink(missing_ok=True)
    logging.info('WhiteboxTools conditioning (%s): %s', method, dem_file)
    if exe:
        binary = Path(exe).expanduser().resolve()
        if not binary.is_file():
            raise ValueError(f'WhiteboxTools executable not found: {binary}')
        def run(cmd: list[str], output: Path) -> None:
            cp = subprocess.run([str(binary), *cmd], cwd=str(output_dir),
                                capture_output=True, text=True, check=False)
            if cp.returncode != 0:
                raise RuntimeError('WhiteboxTools failure: ' + (cp.stderr or cp.stdout)[-1200:])
            _check_wbt_result(0, cmd[0], output)
        if method == 'breach':
            run(['-r=BreachDepressionsLeastCost', f'--dem={dem_file}',
                 f'--output={conditioned}', f'--dist={int(breach_dist)}',
                 '--min_dist', '--fill'], conditioned)
        else:
            run(['-r=FillDepressions', f'--dem={dem_file}',
                 f'--output={conditioned}', '--fix_flats'], conditioned)
        logging.info('WhiteboxTools D8Pointer...')
        run(['-r=D8Pointer', f'--dem={conditioned}',
             f'--output={pointer}'], pointer)
    else:
        try:
            from whitebox import WhiteboxTools
        except ImportError as exc:
            raise RuntimeError(
                'WhiteboxTools is required. Install the Python frontend with '
                '`python -m pip install whitebox`, or pass --whitebox-exe PATH '
                'to a local whitebox_tools executable.'
            ) from exc
        wbt = WhiteboxTools()
        wbt.set_verbose_mode(False)
        if method == 'breach':
            result = wbt.breach_depressions_least_cost(
                str(dem_file), str(conditioned), dist=int(breach_dist),
                min_dist=True, fill=True,
            )
        else:
            result = wbt.fill_depressions(str(dem_file), str(conditioned), fix_flats=True)
        _check_wbt_result(result, 'depression conditioning', conditioned)
        logging.info('WhiteboxTools D8Pointer...')
        result = wbt.d8_pointer(str(conditioned), str(pointer), esri_pntr=False)
        _check_wbt_result(result, 'D8 pointer', pointer)
    return pointer


def read_whitebox_pointers(
    pointer_path: str | Path, dem_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and verify georeferenced pointer alignment with the source DEM."""
    with rasterio.open(dem_path) as dem_ds, rasterio.open(pointer_path) as pn_ds:
        if (dem_ds.shape != pn_ds.shape or dem_ds.crs != pn_ds.crs or
                not dem_ds.transform.almost_equals(pn_ds.transform)):
            raise ValueError('Whitebox D8 pointer grid is not aligned with input DEM')
        dem_valid = read_mask(dem_ds) > 0
        pointer_valid = read_mask(pn_ds) > 0
        if np.any(dem_valid & ~pointer_valid):
            raise ValueError('Whitebox D8 pointer grid has NoData within the valid DEM')
        valid = dem_valid & pointer_valid
        # The D8 decoder ignores pointer codes outside the valid mask. Read
        # the raw band without allocating an unnecessary NumPy masked array.
        pointer = read_band(pn_ds)
        return whitebox_d8_to_next(pointer, valid), valid


def compute_parcel_origin_destinations(
    parcel_codes: np.ndarray, next_flat: np.ndarray,
) -> np.ndarray:
    """Find the first *different* parcel for each parcel cell's D8 path.

    Gaps (code 0) can be crossed; if a path leaves and reenters its source
    parcel it continues tracing. Unmodeled outlets receive destination 0.
    Traversal is memoized and O(number of raster cells), with no recursion.
    ``validate_cell_dag`` must be called before this function.
    """
    labels = np.asarray(parcel_codes, dtype=np.int32).reshape(-1)
    nxt = np.asarray(next_flat, dtype=np.int32).reshape(-1)
    if labels.size != nxt.size or np.any(labels < 0):
        raise ValueError('Invalid parcel raster or pointer shape')
    # -2 = not computed; -1 = no parcel reached downstream, >=0 = index
    gap_first = np.full(labels.size, -2, dtype=np.int32)
    dest = np.full(labels.size, -1, dtype=np.int32)

    def first_parcel_after_gap(start: int) -> int:
        path = []
        at = start
        while at >= 0 and labels[at] == 0 and gap_first[at] == -2:
            path.append(at)
            at = int(nxt[at])
        if at < 0:
            found = -1
        elif labels[at]:
            found = at
        else:
            found = int(gap_first[at])
        for item in path:
            gap_first[item] = found
        return found

    for original in np.flatnonzero(labels):
        original = int(original)
        if dest[original] >= 0:
            continue
        origin_label = int(labels[original])
        path = []
        at = original
        while True:
            if dest[at] >= 0:
                found_code = int(dest[at]); break
            path.append(at)
            down = int(nxt[at])
            if down < 0:
                found_code = 0; break
            down_label = int(labels[down])
            if down_label == origin_label:
                at = down
            elif down_label > 0:
                found_code = down_label; break
            else:
                after = first_parcel_after_gap(down)
                if after < 0:
                    found_code = 0; break
                if int(labels[after]) != origin_label:
                    found_code = int(labels[after]); break
                at = after
        for index in path:
            dest[index] = found_code
    return dest.reshape(np.asarray(parcel_codes).shape)


def score_candidate_connections(
    parcel_codes: np.ndarray,
    destinations: np.ndarray,
    code_to_pid: Mapping[int, object],
    *,
    cell_area_m2: float,
) -> list[dict]:
    """Count each source-parcel cell once, not upstream stream throughflow."""
    labels = np.asarray(parcel_codes, dtype=np.int32).reshape(-1)
    dest = np.asarray(destinations, dtype=np.int32).reshape(-1)
    if labels.shape != dest.shape or not np.isfinite(cell_area_m2) or cell_area_m2 <= 0:
        raise ValueError('Bad raster alignment or cell size for routing scores')
    orig = labels > 0
    if not np.any(orig):
        raise ValueError('No raster cells correspond to parcels at this DEM resolution')
    if np.any(dest[orig] < 0) or np.any(dest[orig] == labels[orig]):
        raise ValueError('Parcel-cell destinations contain unresolved or self links')
    scale = int(labels.max()) + 1
    pair_codes, counts = np.unique(labels[orig].astype(np.int64) * scale +
                                    dest[orig].astype(np.int64), return_counts=True)
    sources = pair_codes // scale
    totals = np.bincount(sources.astype(np.int32), weights=counts, minlength=scale)
    results = []
    for pair, cnt in zip(pair_codes, counts):
        source_code = int(pair // scale)
        to_code = int(pair % scale)
        if source_code not in code_to_pid or (to_code and to_code not in code_to_pid):
            raise ValueError('Raster references a parcel without a matching parcel ID')
        upstream = int(str(code_to_pid[source_code]))
        downstream = int(str(code_to_pid[to_code])) if to_code else None
        results.append({
            'upstream_pid': upstream,
            'downstream_pid': downstream,
            'origin_cells': int(cnt),
            'origin_area_m2': float(cnt) * cell_area_m2,
            'parcel_cells': int(totals[source_code]),
            'share': float(cnt / totals[source_code]),
            'decision': 'not_evaluated',
            'reason': '',
        })
    return sorted(results, key=lambda x: (x['upstream_pid'], -x['origin_cells'],
                                          -1 if x['downstream_pid'] is None else x['downstream_pid']))


def select_acyclic_connections(
    candidates: list[dict],
) -> tuple[list[tuple[int, int]], list[dict], dict]:
    """Choose the strongest available downstream destination for each parcel.

    Each candidate is supported by cells originating inside its source parcel.
    No minimum share is imposed: even a minority destination can represent the
    parcel, but the selected edge is a WHOLE-parcel routing approximation.
    Exact ties are broken by lower numeric pid; OUTSIDE follows numeric pids.
    The globally strongest candidates are installed first. If a link would
    form a directed cycle, try the next candidate for its source parcel.
    """
    by_source: dict[int, list[dict]] = {}
    for row in candidates:
        row['decision'] = 'not_selected'
        row['reason'] = ''
        by_source.setdefault(int(row['upstream_pid']), []).append(row)
    allowed = []
    outside_sources: set[int] = set()
    for source, rows in by_source.items():
        ordered = sorted(rows, key=lambda row: (-int(row['origin_cells']),
                                                float('inf') if row['downstream_pid'] is None
                                                else int(row['downstream_pid'])))
        if ordered[0]['downstream_pid'] is None:
            outside_sources.add(source)
            ordered[0]['decision'] = 'outside_parcel_domain'
            ordered[0]['reason'] = 'largest_drainage_share_outside_domain'
            for other in ordered[1:]:
                other['decision'] = 'rejected_secondary'
                other['reason'] = 'largest_drainage_share_outside_domain'
            continue
        for row in ordered:
            if row['downstream_pid'] is not None:
                allowed.append(row)
            else:
                row['decision'] = 'rejected_secondary'
                row['reason'] = 'outside_not_dominant'
    allowed.sort(key=lambda row: (-row['share'], -row['origin_cells'],
                                  row['upstream_pid'], row['downstream_pid']))
    down: dict[int, int] = {}
    rejected_cycle = 0
    for row in allowed:
        source, target = int(row['upstream_pid']), int(row['downstream_pid'])
        if source in down:
            row['decision'] = 'rejected_secondary'
            row['reason'] = 'stronger_connection_selected'
            continue
        if source == target:
            row['decision'] = 'rejected_cycle'
            row['reason'] = 'self_link'
            rejected_cycle += 1
            continue
        at = target
        path = [source, target]
        while at in down:
            at = down[at]
            path.append(at)
            if at == source:
                break
        if at == source:
            row['decision'] = 'rejected_cycle'
            row['reason'] = 'would_form_cycle:' + '->'.join(map(str, path))
            rejected_cycle += 1
            continue
        down[source] = target
        row['decision'] = 'selected'
        row['reason'] = 'largest_noncyclic_connection'
    unresolved_cycles = sorted({int(r['upstream_pid']) for r in allowed
                                if r['decision'] == 'rejected_cycle'} - set(down))
    edges = sorted(down.items())
    summary = {'ambiguous_parcels': 0, 'outside_parcels': len(outside_sources),
               'rejected_cycle_candidates': rejected_cycle,
               'cycle_unresolved_pids': unresolved_cycles,
               'selected_edges': len(edges)}
    return edges, sorted(candidates, key=lambda row: (row['upstream_pid'],
                  -row['origin_cells'], float('inf') if row['downstream_pid'] is None
                  else row['downstream_pid'])), summary


def write_candidate_diagnostics(path: str | Path, rows: list[dict]) -> None:
    """Write auditable full candidate evidence atomically as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ('upstream_pid', 'downstream_pid', 'origin_cells', 'origin_area_m2',
              'parcel_cells', 'share', 'decision', 'reason')
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8',
                                         dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, 'downstream_pid':
                                'OUTSIDE' if row['downstream_pid'] is None
                                else row['downstream_pid']})
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()
