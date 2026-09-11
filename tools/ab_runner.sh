#!/usr/bin/env bash
# ab_runner.sh -- build both arms and time them. Designed to run inside a container or
# batch job: it takes a source archive, builds each arm with identical flags, and reports.
#
# Two arms, one variable. Both are compiled from the SAME python-multem wrapper with the SAME
# toolchain and flags; the only difference is which MULTEM src/ tree is compiled in:
#
#   control    = master, pristine upstream
#   treatment  = this branch, with the optimisations
#
# That is deliberately not the same thing the per-optimisation suites do. Those isolate one
# optimisation each, but only because the source was edited between runs -- they carry no
# compile-time switches, so today every one of them would build the whole optimised tree. A
# two-arm A/B is what the source can still honestly support: the total effect of the branch.
set -uo pipefail

REPEATS="${REPEATS:-3}"     # TIMED repeats; each case also gets one untimed warm-up first
TESTS="${TESTS:-T1 T2 T3 T4 T5 T6}"

# Derive scratch from the JOB directory. /scratch is shared between concurrent jobs and a
# fixed name is a cross-stream data race that surfaces as failures in files neither job touched.
JOB_DIR="$(pwd)"
# WORK_ROOT is overridable so the identical runner can be driven locally as a pre-flight
# (run_bench_bridge.sh --local); on a station it stays /scratch.
WORK="${WORK_ROOT:-/scratch/$(id -un)}/multem_$(basename "${JOB_DIR}")"
export HOME="${WORK}/home" TMPDIR="${WORK}/tmp"     # set here, not with --env: apptainer
mkdir -p "${WORK}" "${HOME}" "${TMPDIR}"            # refuses --env under --containall

echo "=== multem-bench ==="
echo "job dir : ${JOB_DIR}"
echo "work    : ${WORK}"
echo "host    : $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true
echo "gcc     : $(g++ --version | head -1)"
echo "nvcc    : $(nvcc --version 2>/dev/null | tail -2 | head -1 || echo "not on PATH yet")"
echo "cmake   : $(cmake --version | head -1)"

tar -xzf "${JOB_DIR}/multem_bench_src.tar.gz" -C "${WORK}" || { echo "FATAL: extract failed"; exit 92; }
for d in control treatment python-multem tests; do
    [ -d "${WORK}/${d}" ] || { echo "FATAL: ${d}/ missing from the input archive"; exit 93; }
done

# nvcc is on PATH inside the image; on a bare host it may not be. The harness's own build
# scripts auto-detect it, so do the same here rather than failing at configure time.
if ! command -v nvcc >/dev/null 2>&1; then
    for C in /usr/local/cuda/bin /usr/local/cuda-12.8/bin /usr/local/cuda-12/bin; do
        [ -x "${C}/nvcc" ] && { export PATH="${C}:${PATH}"; echo "nvcc     : found at ${C}"; break; }
    done
fi
NVCC="$(command -v nvcc || true)"

ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
[ -n "${ARCH}" ] || { echo "FATAL: no GPU visible -- was the job submitted with --gpus 1?"; exit 94; }
echo "arch    : sm_${ARCH}"

RESULTS="${JOB_DIR}/results"; mkdir -p "${RESULTS}"

for ARM in control treatment; do
    echo; echo "───────── ${ARM} ─────────"
    # Clean build. A CMakeCache from an earlier attempt pins the old paths, so a fixed fault
    # survives the fix that was supposed to clear it -- and reads as the same failure twice.
    B="${WORK}/build_${ARM}"; rm -rf "${B}"
    cmake -S "${WORK}/tests" -B "${B}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
        ${NVCC:+-DCMAKE_CUDA_COMPILER="${NVCC}"} \
        -DMULTEM_SOURCE_ROOT="${WORK}/python-multem" \
        -DMULTEM_UPSTREAM_SRC="${WORK}/${ARM}/src" \
        > "${RESULTS}/${ARM}_cmake.log" 2>&1 \
        || { echo "FATAL: ${ARM} configure failed"; tail -25 "${RESULTS}/${ARM}_cmake.log"; exit 95; }
    cmake --build "${B}" -j "$(nproc)" > "${RESULTS}/${ARM}_build.log" 2>&1 \
        || { echo "FATAL: ${ARM} build failed"; tail -40 "${RESULTS}/${ARM}_build.log"; exit 96; }
    echo "built: $(ls -lh "${B}/test_multem" | awk '{print $5}')"

    CSV="${RESULTS}/${ARM}_timings.csv"; echo "test,repeat,elapsed_ms" > "${CSV}"
    PROV="${RESULTS}/${ARM}_provenance.txt"; : > "${PROV}"
    for T in ${TESTS}; do
        OUT="${WORK}/out_${ARM}_${T}"; mkdir -p "${OUT}"

        # ONE UNTIMED WARM-UP per case. The first run of a case pays costs the later ones do not
        # -- context creation, module load, allocator warm-up, clocks ramping. Measured: the first
        # repeat of the smallest case ran up to 3x the steady state while the largest reproduced to
        # 0.03%, so averaging it in inflates small cases by an amount that depends on machine state
        # rather than on the code. A ratio of two such means measures how cold each arm started.
        # This run is discarded; its full output is kept as provenance (see below).
        "${B}/test_multem" --test "${T}" --device device --precision float \
            --output "${OUT}" > "${WORK}/warmup_${ARM}_${T}.txt" 2>&1 || true

        # Record what the binary itself says it did -- device, grid, atom count, slices. Without
        # this the run artefact cannot answer "was this the GPU path?" except by inference from
        # the build log -- which is exactly the question that gets asked when a result looks wrong.
        { echo "--- ${T} ---"; grep -E 'Device:|Grid:|Atoms:|Slices:|Specimen:' \
            "${WORK}/warmup_${ARM}_${T}.txt" 2>/dev/null; } >> "${PROV}"

        for R in $(seq 1 "${REPEATS}"); do
            MS=$("${B}/test_multem" --test "${T}" --device device --precision float \
                    --output "${OUT}" 2>&1 | grep -oP 'Time:\s*\K[0-9.]+' | head -1)
            if [ -z "${MS}" ]; then echo "  ${T} run ${R}: NO TIMING (skipped)"; continue; fi
            echo "${T},${R},${MS}" >> "${CSV}"
            echo "  ${T} run ${R}: ${MS} ms"
        done
        # keep one output per test for the cross-arm correctness check
        cp -f "${OUT}/${T}_real.bin" "${RESULTS}/${ARM}_${T}_real.bin" 2>/dev/null || true
    done
done

echo; echo "───────── comparison ─────────"
python3 - "${RESULTS}" <<'PY'
import csv, statistics, sys, pathlib
r = pathlib.Path(sys.argv[1])
def load(name):
    d = {}
    f = r / f"{name}_timings.csv"
    if not f.exists(): return d
    for row in csv.DictReader(f.open()):
        d.setdefault(row["test"], []).append(float(row["elapsed_ms"]))
    return d
c, t = load("control"), load("treatment")
lines = ["", f"{'Test':<6}  {'Control(ms)':<13}{'Treatment(ms)':<15}{'Speedup':<9}",
         f"{'-'*6}  {'-'*11:<13}{'-'*13:<15}{'-'*7:<9}"]
for k in sorted(set(c) & set(t)):
    cm, tm = statistics.mean(c[k]), statistics.mean(t[k])
    lines.append(f"{k:<6}  {cm:<13.1f}{tm:<15.1f}{cm/tm if tm else 0:<9.2f}x")
out = "\n".join(lines)
print(out)
(r / "comparison.txt").write_text(out + "\n")
PY

# Correctness, with a floor. This program is not deterministic run to run -- measured at ~6.3e-06
# max relative, same binary and settings -- so a bare control-vs-treatment difference cannot show
# that an optimisation preserves results: the arms can differ by less than the program differs from
# itself. Re-run the control once more and report that self-difference as the floor, so the arm
# comparison is interpretable instead of merely reassuring.
echo; echo "───────── determinism floor (control run twice) ─────────"
FLOOR="${WORK}/floor"; mkdir -p "${FLOOR}"
for T in ${TESTS}; do
    "${WORK}/build_control/test_multem" --test "${T}" --device device --precision float \
        --output "${FLOOR}" > /dev/null 2>&1 || true
    cp -f "${FLOOR}/${T}_real.bin" "${RESULTS}/floor_${T}_real.bin" 2>/dev/null || true
done

echo; echo "───────── correctness (control vs treatment output) ─────────"
python3 - "${RESULTS}" <<'PY'
import sys, pathlib, struct
r = pathlib.Path(sys.argv[1]); worst = 0.0; n = 0
for cf in sorted(r.glob("control_T*_real.bin")):
    tf = r / cf.name.replace("control_", "treatment_", 1)
    if not tf.exists(): continue
    a = struct.unpack(f"{cf.stat().st_size//4}f", cf.read_bytes())
    b = struct.unpack(f"{tf.stat().st_size//4}f", tf.read_bytes())
    if len(a) != len(b):
        print(f"  {cf.name}: SIZE MISMATCH"); worst = float("inf"); continue
    scale = max(max(abs(x) for x in a), 1e-30)
    d = max(abs(x - y) for x, y in zip(a, b)) / scale
    ff = r / f"floor_{cf.name[8:]}"
    fd = None
    if ff.exists():
        c2 = struct.unpack(f"{ff.stat().st_size//4}f", ff.read_bytes())
        if len(c2) == len(a):
            fd = max(abs(x - y) for x, y in zip(a, c2)) / scale
    print(f"  {cf.name[8:-9]}: arms differ {d:.3e}" +
          (f" | determinism floor {fd:.3e}" + ("  <- at/below the floor" if fd and d <= fd * 1.5 else "")
           if fd is not None else ""))
    worst = max(worst, d); n += 1
print(f"  -> {n} cases, worst {worst:.3e}", "OK" if worst < 1e-4 else "*** EXCEEDS 1e-4 ***")
PY

tar -czf "${JOB_DIR}/results.tar.gz" -C "${JOB_DIR}" results 2>/dev/null || true
echo; echo "=== done: results in ${JOB_DIR}/results/ ==="
