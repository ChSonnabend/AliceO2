#!/usr/bin/env python3
"""Primary-pruning QA directly from existing full or roots-only shadow CSVs.

Accepts .csv and .csv.gz; reads chunks, retaining only scored input-root metrics
for exact ROC and event-separated threshold calibration/test. Memory scales with
chunk size plus the number of scored roots, never with all secondary rows.
Requires the dependencies of analyzeTransportShadow.py in the same directory.
"""
import argparse
import json
from pathlib import Path
from csvProgress import million_rows
from analyzeTransportShadow import main as analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', nargs='+', help='One model per invocation; full or filtered CSVs')
    source.add_argument('--cache', help='selected_tracks.npz from an earlier run; no original CSV access')
    p.add_argument('--out', required=True)
    p.add_argument('--chunk-size', type=int, default=50000)
    limit = p.add_mutually_exclusive_group()
    limit.add_argument('--max-rows', type=int, help='Partial sample: maximum input rows per file, possibly a partial event')
    limit.add_argument('--max-million-rows', dest='max_rows', type=million_rows, help='Maximum million input rows per file (e.g. 10 or 0.5)')
    p.add_argument('--target-hit-loss', type=float, default=0.)
    p.add_argument('--calibration-fraction', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    source_args = ['--cache', a.cache] if a.cache else ['--csv', *a.csv]
    args = [*source_args, '--out', a.out, '--scope', 'roots',
            '--csv-chunk-size', str(a.chunk_size), '--target-hit-loss', str(a.target_hit_loss),
            '--calibration-fraction', str(a.calibration_fraction), '--seed', str(a.seed)]
    if a.max_rows is not None:
        args += ['--max-csv-rows', str(a.max_rows)]
    analyze(args)
    from plotPrimaryPruningQA import run_qa
    report = json.loads((Path(a.out)/'summary.json').read_text())
    cut = report.get('threshold')
    if cut is None and len(report.get('configured_thresholds', [])) == 1:
        cut = report['configured_thresholds'][0]
    run_qa(Path(a.out)/'selected_tracks.npz', Path(a.out)/'extra_qa', cut, report.get('operating_points'))


if __name__ == '__main__':
    main()
