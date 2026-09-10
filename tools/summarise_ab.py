#!/usr/bin/env python3
"""Summarise a two-arm A/B run, with and without the first (cold) repeat.

    python3 tools/summarise_ab.py <results-dir> [<results-dir> ...]

Each directory must hold control_timings.csv and treatment_timings.csv as written by
bridge_runner.sh (columns: test,repeat,elapsed_ms).

WHY DROP THE FIRST REPEAT. The first timed run of a case pays costs the later ones do not --
context creation, module load, allocator warm-up, clocks still ramping. Measured here, the first
repeat of the smallest case ran up to 3x the steady-state value while the largest case was
reproducible to 0.03%. Averaging it in therefore inflates small-case timings by an amount that
depends on machine state rather than on the code, and a *ratio* of two such means is not a
speed-up -- it is mostly a measurement of how cold each arm happened to start.

Both columns are printed rather than one, because the difference between them is the honest
uncertainty on the small cases.
"""
import csv, statistics, sys, pathlib


def load(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["test"], []).append(float(row["elapsed_ms"]))
    return out


def summarise(d):
    d = pathlib.Path(d)
    c, t = load(d / "control_timings.csv"), load(d / "treatment_timings.csv")
    print(f"\n{d.name}")
    print(f"  {'test':<5} {'control':>9} {'treat':>9} {'all':>7}   "
          f"{'control':>9} {'treat':>9} {'drop1':>7}   {'spread':>7}")
    print(f"  {'':<5} {'(mean)':>9} {'(mean)':>9} {'':>7}   "
          f"{'(drop1)':>9} {'(drop1)':>9} {'':>7}   {'trt':>7}")
    for k in sorted(set(c) & set(t)):
        cm, tm = statistics.mean(c[k]), statistics.mean(t[k])
        cd, td = statistics.mean(c[k][1:]), statistics.mean(t[k][1:])
        spread = (max(t[k]) - min(t[k])) / min(t[k]) * 100
        print(f"  {k:<5} {cm:>9.1f} {tm:>9.1f} {cm/tm:>6.2f}x   "
              f"{cd:>9.1f} {td:>9.1f} {cd/td:>6.2f}x   {spread:>6.0f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        summarise(d)
