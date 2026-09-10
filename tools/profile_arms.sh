#!/usr/bin/env bash
# profile_arms.sh -- where does the time actually go, per arm?
#
#   bash tools/profile_arms.sh <control_binary> <treatment_binary> [TEST] [outdir]
#
# A wall-clock A/B says how much faster the branch is. It does not say WHY, and in particular it
# cannot distinguish "the optimisation is weak" from "the optimisation is strong but the thing it
# optimises is a minority of the runtime". Those need different responses, so measure them apart:
# this profiles each arm with nsys and reports GPU kernel time, memcpy time, and the per-kernel
# breakdown. Amdahl does the rest.
set -uo pipefail
CTL="${1:?control binary}"; TRT="${2:?treatment binary}"
TEST="${3:-T3}"; OUT="${4:-$(mktemp -d)}"
command -v nsys >/dev/null || { echo "FATAL: nsys not on PATH"; exit 1; }
mkdir -p "${OUT}/scratch"

for pair in "control:${CTL}" "treatment:${TRT}"; do
    ARM="${pair%%:*}"; BIN="${pair#*:}"
    nsys profile --force-overwrite=true -o "${OUT}/${TEST}_${ARM}" --stats=false \
        "${BIN}" --test "${TEST}" --device device --precision float \
        --output "${OUT}/scratch" > "${OUT}/${ARM}_run.log" 2>&1
    nsys stats --report cuda_gpu_kern_sum "${OUT}/${TEST}_${ARM}.nsys-rep" \
        > "${OUT}/${ARM}_kernels.txt" 2>/dev/null
done

python3 - "${OUT}" "${TEST}" <<'PY'
import sqlite3, sys, pathlib, re
out, test = pathlib.Path(sys.argv[1]), sys.argv[2]
print(f"\n{'arm':<11}{'kernels(ms)':>13}{'memcpy(ms)':>12}{'wall(ms)':>11}{'GPU % of wall':>15}")
tot = {}
for arm in ("control", "treatment"):
    db = out / f"{test}_{arm}.sqlite"
    c = sqlite3.connect(db)
    k = c.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] or 0
    try:
        m = c.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0] or 0
    except sqlite3.Error:
        m = 0
    log = (out / f"{arm}_run.log").read_text()
    w = re.search(r"Time:\s*([0-9.]+)", log)
    wall = float(w.group(1)) if w else float("nan")
    gpu = (k + m) / 1e6
    tot[arm] = (gpu, wall)
    print(f"{arm:<11}{k/1e6:>13.1f}{m/1e6:>12.1f}{wall:>11.1f}{100*gpu/wall:>14.1f}%")

(gc, wc), (gt, wt) = tot["control"], tot["treatment"]
print(f"\n  GPU-side speed-up      : {gc/gt:.2f}x")
# Amdahl on the profiled control: hold the non-GPU remainder fixed, swap in the treatment's GPU time
nongpu = wc - gc
print(f"  non-GPU remainder      : {nongpu:.1f} ms of the control's {wc:.1f} ms  ({100*nongpu/wc:.0f}%)")
print(f"  predicted end-to-end   : {wc/(nongpu+gt):.2f}x   <- what a wall-clock A/B should show")
print("\n  NB the profiler inflates wall time; the RATIO and the GPU fractions are the point,")
print("  not the absolute wall figures. Re-run unprofiled for headline timings.")
PY
echo
echo "artefacts in ${OUT}"
