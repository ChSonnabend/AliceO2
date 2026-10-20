"""Exact, cached CSV record counts for streaming O2 CSV tools."""
import csv
import gzip
import io
import itertools
import json
from pathlib import Path
import sys
import time


def count_csv_rows(filename, max_rows=None):
    path = Path(filename).resolve()
    stat = path.stat()
    identity = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    cache = Path(str(path)+'.rows.json')
    try:
        saved = json.loads(cache.read_text())
        if saved['identity'] == identity and isinstance(saved['rows'], int) and saved['rows'] >= 0:
            total = saved['rows']
            print(f'[count cached] {path.name}: {total:,} data rows', file=sys.stderr, flush=True)
            return min(total, max_rows) if max_rows is not None else total
    except (OSError, ValueError, KeyError, TypeError):
        pass
    print(f'[count] {path}: counting CSV records before processing...', file=sys.stderr, flush=True)
    started = last = time.monotonic()
    count = 0
    with path.open('rb') as raw:
        binary = gzip.GzipFile(fileobj=raw) if path.suffix == '.gz' else raw
        with io.TextIOWrapper(binary, encoding='utf-8', newline='') as stream:
            reader = csv.reader(stream, strict=True)
            next(reader)  # Header is not a data record.
            records = reader if max_rows is None else itertools.islice(reader, max_rows)
            for _ in records:
                count += 1
                if count % 1000000 == 0 or time.monotonic()-last >= 10:
                    last = time.monotonic()
                    percent = 100*raw.tell()/max(1, stat.st_size)
                    print(f'[count] {count:,} rows; approximately {percent:.1f}% of file bytes read; '
                          f'{last-started:.0f}s elapsed', file=sys.stderr, flush=True)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
        raise RuntimeError('CSV changed while counting; use a completed, stable input')
    if max_rows is None:
        try:
            cache.write_text(json.dumps(dict(identity=identity, rows=count))+'\n')
        except OSError as error:
            print(f'[count] Could not save count cache: {error}', file=sys.stderr, flush=True)
    print(f'[count done] {count:,} data rows to process', file=sys.stderr, flush=True)
    return count


def million_rows(value):
    """argparse conversion: millions of rows, allowing exact fractional millions."""
    import argparse
    from decimal import Decimal, InvalidOperation
    try:
        rows = Decimal(value) * 1000000
        if not rows.is_finite() or rows <= 0 or rows != rows.to_integral_value():
            raise ValueError
        return int(rows)
    except (InvalidOperation, ValueError, OverflowError):
        raise argparse.ArgumentTypeError('Expected positive millions representing a whole number of rows')
