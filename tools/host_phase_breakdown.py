#!/usr/bin/env python3
"""Apportion a run's wall time from an nsys trace: GPU, CUDA API, and host phases.

    python3 tools/host_phase_breakdown.py <trace.sqlite> [atom_count]

A kernel profile says how fast the kernels are. It cannot say what fraction of the run they are,
and that fraction is what decides whether optimising them is worth anything. This walks the CUDA
runtime-API timeline and treats the GAPS between calls as host compute -- then reports the largest
gaps with the calls that bracket them, which is usually enough to name the phase without a rebuild.

Give it an atom count and it normalises the specimen-setup block per atom, which is the number that
reveals whether a cost is O(atoms) or fixed.
"""
import sqlite3, sys, pathlib


def main(db, atoms=None):
    c = sqlite3.connect(db)
    rows = c.execute("""select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r
                        join StringIds s on r.nameId=s.id order by r.start""").fetchall()
    if not rows:
        sys.exit("no CUDA runtime records in this trace")
    k = c.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] or 0
    t0, tN = rows[0][0], max(r[1] for r in rows)
    span = tN - t0

    merged = []
    for s, e, _ in rows:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    covered = sum(e - s for s, e in merged)

    gaps = []
    for i in range(len(rows) - 1):
        g = rows[i + 1][0] - rows[i][1]
        if g > 0:
            gaps.append((g, rows[i][2], rows[i + 1][2], rows[i][1] - t0))
    gaps.sort(reverse=True)

    print(f"\n  span                : {span/1e6:9.1f} ms")
    print(f"  GPU kernels         : {k/1e6:9.1f} ms  ({100*k/span:4.1f}%)")
    print(f"  inside a CUDA call  : {covered/1e6:9.1f} ms  ({100*covered/span:4.1f}%)")
    print(f"  host compute (gaps) : {(span-covered)/1e6:9.1f} ms  ({100*(span-covered)/span:4.1f}%)")
    print(f"\n  {'gap ms':>9} {'at':>5}  {'preceded by':<28}{'followed by':<24}")
    for g, a, b, pos in gaps[:5]:
        print(f"  {g/1e6:>9.1f} {100*pos/span:>4.0f}%  {a[:27]:<28}{b[:23]:<24}")

    spec = max((g for g, a, b, _ in gaps
                if a.startswith("cudaStreamSynchronize") and b.startswith("cudaMalloc")), default=0)
    if spec:
        line = f"\n  specimen-setup block: {spec/1e6:9.1f} ms  ({100*spec/span:4.1f}% of the run)"
        if atoms:
            line += f"   = {spec/int(atoms):.0f} ns/atom"
        print(line)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(pathlib.Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
