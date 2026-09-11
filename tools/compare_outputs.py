#!/usr/bin/env python3
"""Compare two MULTEM output arrays (raw float32 .bin, as tests/test_multem writes them).

    python3 tools/compare_outputs.py A.bin B.bin [--floor A2.bin]

Prints the maximum difference relative to the largest magnitude in A. With --floor, A2 is a second
run of A's own settings, and A vs A2 is printed too: this program is not deterministic run to run,
so "B agrees with A to 1e-6" means nothing until it is set beside how well A agrees with itself.

A missing file is reported rather than raised -- a run that crashed before writing its output is
itself the result.
"""
import argparse
import math
import pathlib
import struct
import sys


def load(path):
    raw = pathlib.Path(path).read_bytes()
    return struct.unpack(f"{len(raw) // 4}f", raw)


def max_rel(a, b, scale):
    return max(abs(x - y) for x, y in zip(a, b)) / scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference output")
    ap.add_argument("b", help="output to compare against it")
    ap.add_argument("--floor", help="a second run of the reference's settings")
    args = ap.parse_args()

    missing = [p for p in [args.a, args.b] + ([args.floor] if args.floor else []) if not pathlib.Path(p).exists()]
    for p in missing:
        print(f"MISSING {p}")
    if missing:
        return 2

    a, b = load(args.a), load(args.b)
    if len(a) != len(b):
        print(f"SIZE MISMATCH {len(a)} vs {len(b)} elements")
        return 1
    scale = max(max(abs(v) for v in a), 1e-30)
    nonfinite = sum(1 for v in b if not math.isfinite(v))
    print(f"B vs A      : max rel {max_rel(a, b, scale):.3e}  ({len(a)} elements, {nonfinite} non-finite in B)")
    if args.floor:
        c = load(args.floor)
        if len(c) != len(a):
            print(f"FLOOR SIZE MISMATCH {len(c)} vs {len(a)} elements")
            return 1
        print(f"A2 vs A     : max rel {max_rel(a, c, scale):.3e}  (determinism floor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
