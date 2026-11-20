#!/usr/bin/env python3
"""Explicit, one-time parcel_up -> required parcel_down GeoPackage migration.

The normal model loader has no legacy parcel_up fallback. This utility is an
explicit data-migration operation, not an alternate input reader. It validates
source/target references and acyclicity before modifying any GeoPackage.

Example:
    python utils/migrate_parcel_down.py --east-fork
    python utils/migrate_parcel_down.py parcels.gpkg --reference-csv drainage.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sqlite3
import sys

try:
    from .parcel_down_graph import write_gpkg_parcel_down, validate_direct_edges
except ImportError:
    from parcel_down_graph import write_gpkg_parcel_down, validate_direct_edges


def read_links_from_csv(path: Path) -> list[tuple[int,int]]:
    with path.open(newline='',encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    cols = set(rows[0])
    if {'pid','pid_down'} <= cols:
        return [(int(row['pid']), int(row['pid_down'])) for row in rows]
    if {'pid','pid_up'} <= cols:
        # Explicit migration: legacy upstream table becomes source->receiver.
        return [(int(row['pid_up']), int(row['pid'])) for row in rows]
    raise ValueError('Reference CSV must have pid,pid_down or legacy pid,pid_up')


def migrate(path: Path, links: list[tuple[int,int]], *, backup: bool = True,
            dry_run: bool = False) -> int:
    if path.suffix.lower() != '.gpkg' or not path.is_file():
        raise ValueError(f'Parcel GeoPackage not found: {path}')
    with sqlite3.connect(path) as conn:
        ids = [row[0] for row in conn.execute('SELECT pid FROM parcels')]
    valid = validate_direct_edges(links, ids)
    if dry_run:
        print(f'VALID {path}: {len(valid)} parcel_down links, {len(ids)} parcel IDs')
        return len(valid)
    if backup:
        dest = path.with_name(path.name + '.before_parcel_down.bak')
        if dest.exists():
            raise FileExistsError(f'Backup already exists: {dest}. Remove or rename it explicitly.')
        # SQLite online backup obtains a consistent snapshot, unlike a raw copy
        # when WAL journaling has outstanding changes.
        with sqlite3.connect(path) as source, sqlite3.connect(dest) as target:
            source.backup(target)
        print('Backup:',dest)
    count = write_gpkg_parcel_down(path, valid, ids)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM parcel_down').fetchone()[0] == count
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='parcel_up'").fetchone() is None
    print(f'Updated {path}: {count} parcel_down links')
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parcels',nargs='?',type=Path)
    parser.add_argument('--reference-csv',type=Path)
    parser.add_argument('--east-fork',action='store_true',
                        help='Migrate the four East Fork parcel GeoPackages using the audited Whitebox graph.')
    parser.add_argument('--no-backup',action='store_true',
                        help='Skip the default .before_parcel_down.bak safety copy')
    parser.add_argument('--dry-run',action='store_true',
                        help='Validate graph and parcel IDs without writing')
    args=parser.parse_args()
    if args.east_fork:
        if args.parcels or args.reference_csv:
            parser.error('--east-fork cannot be combined with positional parcels or --reference-csv')
        repo=Path(__file__).resolve().parents[1]
        source=repo/'utils/data/east_fork_whitebox_parcel_down.csv'
        targets=sorted((repo/'examples/east_fork/inputs/parcels').glob('*.gpkg'))
        if len(targets)!=4:
            raise ValueError(f'Expected exactly 4 East Fork parcels GeoPackages, found {len(targets)}')
    else:
        if args.parcels is None or args.reference_csv is None:
            parser.error('Provide parcels.gpkg and --reference-csv, or use --east-fork')
        source=args.reference_csv
        targets=[args.parcels]
    links=read_links_from_csv(source)
    # Validate *all* files before modifying the first one.
    for path in targets:
        migrate(path, links, dry_run=True)
    if not args.dry_run:
        for path in targets:
            migrate(path, links, backup=not args.no_backup)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
