#!/usr/bin/env python3
"""Tabulate a thread-count sweep of toggle_bench.sh runs: one row per (directory, test).

    python3 tools/summarise_sweep.py <results dir> [<results dir> ...] [--phase atoms_u.sort_by_z]

Each directory is one toggle_bench.sh job (toggle_timings.csv + stdout.log). The thread count is
read from the job's own log -- the "env" line records MULTEM_SORT_THREADS -- rather than from the
directory name, so a mislabelled directory cannot put a number in the wrong row. Medians, because a
single disturbed repeat moves a mean of four but not a median.
"""
import argparse
import csv
import pathlib
import re
import statistics
import sys


def threads_of(d):
    log = (d / "stdout.log").read_text(errors="replace")
    m = re.search(r"MULTEM_SORT_THREADS=(\d+)", log)
    return int(m.group(1)) if m else None


def medians(d, phase):
    vals = {}
    for r in csv.DictReader(open(d / "toggle_timings.csv")):
        if r["phase"].strip() == phase:
            vals.setdefault((r["test"], r["toggle"]), []).append(float(r["ms"]))
    return {k: statistics.median(v) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--phase", default="atoms_u.sort_by_z")
    a = ap.parse_args()

    rows = []
    for d in a.dirs:
        t = threads_of(d)
        if t is None:
            print(f"SKIP {d}: no MULTEM_SORT_THREADS in its stdout.log", file=sys.stderr)
            continue
        ph, wall = medians(d, a.phase), medians(d, "WALL")
        for test in sorted({k[0] for k in ph}):
            if (test, "off") in ph and (test, "on") in ph:
                rows.append((test, t, ph[(test, "off")], ph[(test, "on")], wall[(test, "off")], wall[(test, "on")], d.name))

    print(f"  median {a.phase} and WALL, ms (off = sequential, on = the toggle)")
    print(f"  {'test':<5}{'threads':>8}{'phase off':>11}{'phase on':>10}{'speed-up':>10}{'wall off':>10}{'wall on':>9}{'wall':>8}")
    for test, t, po, pn, wo, wn, name in sorted(rows):
        print(f"  {test:<5}{t:>8}{po:>11.1f}{pn:>10.1f}{po / pn:>9.2f}x{wo:>10.1f}{wn:>9.1f}{(wn - wo) / wo * 100:>+7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
