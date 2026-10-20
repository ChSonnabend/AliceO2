#!/usr/bin/env python3
"""Stream an O2 shadow CSV to a roots-only CSV, optionally gzip-compressed.

Uses the standard library and holds one CSV record at a time. Existing subtree
truth is copied unchanged; never recompute it after removing secondary rows.
Use completed input files. Output is published only after a successful scan.
"""
import argparse
import csv
import gzip
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from csvProgress import count_csv_rows, million_rows


def open_csv(path, mode):
    opener = gzip.open if str(path).endswith('.gz') else open
    return opener(path, mode, encoding='utf-8', newline='')


def filter_roots(source, output, max_rows=None, progress_every=1000000):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Input and output must differ')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if max_rows is not None and max_rows <= 0:
        raise ValueError('max_rows must be positive')
    if progress_every < 0:
        raise ValueError('progress_every must be nonnegative')
    total_rows = count_csv_rows(source, max_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Compression is selected from the final filename, not the temporary suffix.
    fd, temporary = tempfile.mkstemp(prefix=output.name+'.', suffix='.partial', dir=output.parent)
    os.close(fd)
    scanned = kept = 0
    start = time.monotonic()
    before = source.stat()
    try:
        opener = gzip.open if str(output).endswith('.gz') else open
        with open_csv(source, 'rt') as incoming, opener(temporary, 'wt', encoding='utf-8', newline='') as outgoing:
            reader = csv.reader(incoming, strict=True)
            header = next(reader)
            if len(header) != len(set(header)):
                raise ValueError('Duplicate CSV columns')
            required = ['csv_schema_version', 'shadow_mode', 'is_simulation_root']
            indices = {name: header.index(name) for name in required}
            writer = csv.writer(outgoing, lineterminator='\n')
            writer.writerow(header)
            records = reader if max_rows is None else itertools.islice(reader, max_rows)
            for row in records:
                scanned += 1
                if len(row) != len(header):
                    raise ValueError(f'Malformed/truncated CSV record {scanned}: expected {len(header)} fields, got {len(row)}')
                if row[indices['csv_schema_version']] != '1' or row[indices['shadow_mode']] != '1':
                    raise ValueError(f'Record {scanned}: expected schema-1 shadow CSV')
                root = row[indices['is_simulation_root']]
                if root not in ('0', '1'):
                    raise ValueError(f'Record {scanned}: invalid is_simulation_root')
                if root == '1':
                    writer.writerow(row)
                    kept += 1
                if progress_every and scanned % progress_every == 0:
                    elapsed = time.monotonic()-start
                    print(f'{source.name}: scanned {scanned:,}/{total_rows:,} ({100*scanned/max(1,total_rows):.2f}%); kept {kept:,}; {scanned/max(elapsed,1):,.0f} rows/s', file=sys.stderr, flush=True)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Input changed during filtering; use a completed, stable CSV')
        # Hard-link publication is atomic and cannot overwrite a racing output.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    print(f'[filter done] {scanned:,}/{total_rows:,} rows (100.00%); kept {kept:,}', file=sys.stderr, flush=True)
    return dict(input=str(source), output=str(output), scanned_rows=scanned,
                root_rows=kept, max_rows=max_rows, limited_scan=max_rows is not None,
                elapsed_seconds=round(time.monotonic()-start, 3), output_bytes=output.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Existing .csv or .csv.gz')
    parser.add_argument('--output', required=True, help='New .csv or .csv.gz; existing files are never overwritten')
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument('--max-rows', type=int, help='Read at most this many input data records for a partial sample')
    limit.add_argument('--max-million-rows', dest='max_rows', type=million_rows, help='Stop after this many million input rows (e.g. 10 or 0.5)')
    parser.add_argument('--progress-every', type=int, default=1000000, help='Input rows between progress updates; 0 disables')
    args = parser.parse_args()
    print(json.dumps(filter_roots(args.input, args.output, args.max_rows, args.progress_every), indent=2))


if __name__ == '__main__':
    main()
