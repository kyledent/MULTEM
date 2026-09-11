#!/usr/bin/env bash
# toggle_bench.sh -- time one binary against itself under different environment settings.
#
#   TOGGLES="MULTEM_FAST_SETUP" REPEATS=4 TESTS="T3" bash tools/toggle_bench.sh [archive]
#   bash tools/toggle_bench.sh --summarise toggle_timings.csv      # re-summarise a saved run
# (as a batch-job body the archive is found in the job directory; the argument is for local use)
#
# The A/B runner compares two SOURCE TREES. This compares two CODE PATHS inside one build, which
# is the right instrument when an optimisation is selected at runtime: one compile, one process
# image, alternating runs. A rebuild-and-compare on a thermally variable machine moved every
# phase by 16% -- including code that had not been touched -- which is enough to invent or hide a
# result of the size usually being looked for.
#
# Runs are INTERLEAVED (off, on, off, on ...) rather than grouped, so a machine that drifts during
# the job degrades both arms equally instead of whichever ran second.
set -uo pipefail

# Per-phase means, off vs on. A phase name can occur more than once per run -- the set_atoms
# sub-phases fire once for each of the two set_atoms calls -- so repeats of a name are numbered
# (#1, #2) rather than pooled. Pooling averaged a 52 ms call with a 13 ms call and reported the
# difference between them as a 312% "spread".
# The numbering is decided per TEST, not per run: a toggle can remove a call (skipping the second
# get_statistic leaves one occurrence where there were two), and numbering per run then gave the
# two arms different names and dropped the row from both. An arm with no such call shows "--".
summarise() {
python3 - "$1" <<'PY'
import csv, statistics, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
run = lambda r: (r["test"], r["toggle"], r["repeat"])
count = collections.Counter((run(r), r["phase"].strip()) for r in rows)
repeats = collections.defaultdict(int)
for (rk, p), n in count.items():
    repeats[(rk[0], p)] = max(repeats[(rk[0], p)], n)
seen = collections.Counter()
agg = collections.defaultdict(list)
for r in rows:
    p = r["phase"].strip()
    seen[(run(r), p)] += 1
    if repeats[(r["test"], p)] > 1:
        p = f"{p} #{seen[(run(r), p)]}"
    agg[(r["test"], p, r["toggle"])].append(float(r["ms"]))
# Medians beside the means: one disturbed repeat (a neighbour's burst of work) moves a mean of four
# but not a median, and the spread column says when that has happened.
mean = lambda v: f"{statistics.mean(v):>10.1f}" if v else f"{'--':>10}"
med = lambda v: f"{statistics.median(v):>10.1f}" if v else f"{'--':>10}"
spread = lambda v: f"{(max(v) - min(v)) / min(v) * 100 if min(v) else 0:>10.0f}%" if v else f"{'':>11}"
keys = sorted({(t, p) for (t, p, _) in agg})
print(f"  {'test':<6}{'phase':<34}{'off':>10}{'on':>10}{'saved':>10}"
      f"{'spread off':>12}{'spread on':>11}{'med off':>10}{'med on':>10}{'med saved':>11}")
for t, p in keys:
    off, on = agg.get((t, p, "off")), agg.get((t, p, "on"))
    saved = f"{statistics.mean(off) - statistics.mean(on):>+10.1f}" if off and on else f"{'':>10}"
    msaved = f"{statistics.median(off) - statistics.median(on):>+11.1f}" if off and on else f"{'':>11}"
    print(f"  {t:<6}{p[:33]:<34}{mean(off)}{mean(on)}{saved} {spread(off)}{spread(on)}{med(off)}{med(on)}{msaved}")
PY
}

if [ "${1:-}" = "--summarise" ]; then
    [ -f "${2:-}" ] || { echo "usage: $0 --summarise toggle_timings.csv"; exit 2; }
    summarise "$2"; exit 0
fi

TOGGLES="${TOGGLES:-MULTEM_FAST_SETUP}"
REPEATS="${REPEATS:-4}"
TESTS="${TESTS:-T3}"

JOB_DIR="$(pwd)"
# A job system that stages inputs copies them INTO the job directory; it does not pass them as arguments.
ARCHIVE="${1:-${JOB_DIR}/multem_toggle_src.tar.gz}"
[ -f "${ARCHIVE}" ] || { echo "FATAL: no source archive at ${ARCHIVE}"; ls -la "${JOB_DIR}"; exit 91; }
WORK="${WORK_ROOT:-/scratch/$(id -un)}/toggle_$(basename "${JOB_DIR}")"
export HOME="${WORK}/home" TMPDIR="${WORK}/tmp"
mkdir -p "${WORK}" "${HOME}" "${TMPDIR}"

echo "=== toggle_bench ==="
echo "host    : $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.used,memory.total --format=csv,noheader || true
echo "toggles : ${TOGGLES}"

tar -xzf "${ARCHIVE}" -C "${WORK}" || { echo "FATAL: extract failed"; exit 92; }
for d in src tests python-multem; do
    [ -d "${WORK}/${d}" ] || { echo "FATAL: ${d}/ missing from the archive"; exit 93; }
done

# The image's %environment puts /usr/local/cuda/bin on PATH, yet one job's configure reported
# "Looking for a CUDA compiler - NOTFOUND" from check_language(CUDA) and built host-only. Whether
# nvcc was off PATH or the probe itself failed was not recorded, so record it here -- and name
# the compiler explicitly, which skips check_language's probe instead of depending on it.
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VIA="PATH"
else
    NVCC_VIA="fallback search"
    for C in /usr/local/cuda/bin /usr/local/cuda-12.8/bin /usr/local/cuda-12/bin; do
        [ -x "${C}/nvcc" ] && { export PATH="${C}:${PATH}"; break; }
    done
fi
NVCC="$(command -v nvcc || true)"
echo "nvcc    : ${NVCC:-NOT FOUND} (via ${NVCC_VIA})"

ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
[ -n "${ARCH}" ] || { echo "FATAL: no GPU visible -- submitted with --gpus 1?"; exit 94; }

B="${WORK}/build"; rm -rf "${B}"
cmake -S "${WORK}/tests" -B "${B}" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
      -DMULTEM_SOURCE_ROOT="${WORK}/python-multem" \
      ${NVCC:+-DCMAKE_CUDA_COMPILER="${NVCC}"} \
      -DMULTEM_UPSTREAM_SRC="${WORK}/src" > "${JOB_DIR}/cmake.log" 2>&1 \
    || { echo "FATAL: configure failed"; tail -25 "${JOB_DIR}/cmake.log"; exit 95; }
cmake --build "${B}" -j "$(nproc)" > "${JOB_DIR}/build.log" 2>&1 \
    || { echo "FATAL: build failed"; tail -40 "${JOB_DIR}/build.log"; exit 96; }
if grep -q 'CUDA not found\|CUDA disabled' "${JOB_DIR}/cmake.log"; then
    echo "FATAL: cmake built a HOST-ONLY binary -- these timings would not be GPU timings."
    grep -E 'CUDA' "${JOB_DIR}/cmake.log" | head -5
    exit 97
fi
echo "built   : $(ls -lh "${B}/test_multem" | awk '{print $5}')"
echo "cpus    : $(nproc) in this process's affinity mask"
echo "checks  : ${CHECKS:-none}"

CSV="${JOB_DIR}/toggle_timings.csv"; echo "test,toggle,repeat,phase,ms" > "${CSV}"
for T in ${TESTS}; do
    OUT="${WORK}/out_${T}"; mkdir -p "${OUT}"

    # Correctness first, once per case: a toggle that changes the answer is not worth timing.
    # Each CHECKS entry is an env var that makes the binary print "[check] ... N differ ...".
    # No such line at all is also a failure -- a check that never ran has shown nothing.
    for C in ${CHECKS:-}; do
        LINES="$(env "${C}=1" "${B}/test_multem" --test "${T}" --device device --precision float \
                   --output "${OUT}" 2>&1 | grep -E '\[check\]' || true)"
        [ -n "${LINES}" ] || { echo "FATAL: ${C}=1 printed no [check] line for ${T}"; exit 98; }
        echo "${LINES}" | sed "s/^ */  ${T} /" | tee -a "${JOB_DIR}/checks.txt"
        if echo "${LINES}" | grep -qvE ' 0 differ '; then
            echo "FATAL: ${C} found differences for ${T} -- not timing a changed result"; exit 99
        fi
    done
    # one untimed warm-up per case, as in the A/B runner and for the same reason. Keep its
    # "Atoms:" line: it records the atom count and the order they were presented in, which is
    # otherwise only inferable from the timings (e.g. whether MULTEM_TEST_SHUFFLE took effect)
    echo "  ${T} $("${B}/test_multem" --test "${T}" --device device --precision float \
        --output "${OUT}" 2>&1 | grep -m1 -E 'Atoms:' | sed -E 's/^ +//')"

    for R in $(seq 1 "${REPEATS}"); do
        for STATE in off on; do
            ENVSET=""
            [ "${STATE}" = "on" ] && for v in ${TOGGLES}; do ENVSET="${ENVSET} ${v}=1"; done
            # shellcheck disable=SC2086
            env MULTEM_PHASE_TIMING=1 ${ENVSET} "${B}/test_multem" --test "${T}" \
                --device device --precision float --output "${OUT}" 2>&1 \
              | grep -E '\[phase\]|Time:' \
              | sed -E "s/^  \[phase\] +(.+[^ ]) +([0-9.]+) ms$/${T},${STATE},${R},\1,\2/; \
                        s/^  Time: ([0-9.]+) ms.*$/${T},${STATE},${R},WALL,\1/" >> "${CSV}"
            echo "  ${T} repeat ${R} ${STATE}: $(tail -1 "${CSV}" | cut -d, -f5) ms wall"
        done
    done
done

echo; echo "───────── per-phase means ─────────"
summarise "${CSV}"
echo; echo "=== done ==="
