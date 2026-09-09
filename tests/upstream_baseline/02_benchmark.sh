#!/usr/bin/env bash
# 02_benchmark.sh -- Benchmark upstream (unmodified) MULTEM and compare
#                    against the optimised build.
#
# Runs both upstream and optimized binaries with matched test cases,
# produces a side-by-side comparison table with speedup factors.
#
# PASS always (timing-only, no correctness checks).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

require_gpu

report_header "Upstream vs Optimised Benchmark"

RESULTS=$(setup_results_dir upstream_baseline benchmark)

# Locate binaries
UPSTREAM_BIN="${TESTS_DIR}/build_upstream/test_multem_upstream"
OPTIMISED_BIN="${TESTS_DIR}/build/test_multem"

if [[ ! -x "${UPSTREAM_BIN}" ]]; then
    report_fail "Upstream binary not found at ${UPSTREAM_BIN}"
    report_info "Run 01_build.sh first."
    exit 1
fi

if [[ ! -x "${OPTIMISED_BIN}" ]]; then
    report_fail "Optimised binary not found at ${OPTIMISED_BIN}"
    report_info "Build the optimised version first: ./run_suite.sh async_streaming 00"
    exit 1
fi

SUMMARY="${RESULTS}/comparison.txt"
TIMING_CSV="${RESULTS}/timings.csv"
N_REPEATS="${BENCHMARK_REPEATS:-3}"

# Record machine info
{
    print_machine_info
    print_gpu_info
    echo "Upstream binary:  ${UPSTREAM_BIN}"
    echo "Optimised binary: ${OPTIMISED_BIN}"
    echo "Repeats: ${N_REPEATS}"
    echo ""
} > "${SUMMARY}"

# CSV header
echo "binary,test,repeat,elapsed_ms" > "${TIMING_CSV}"

# Determine which tests to run based on GPU memory
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' \r' || echo 0)

TESTS_TO_RUN="T1 T2 T3 T4"
if [[ ${GPU_MEM_MB} -ge 4000 ]]; then
    TESTS_TO_RUN="T1 T2 T3 T4 T5"
fi
if [[ ${GPU_MEM_MB} -ge 6000 ]]; then
    TESTS_TO_RUN="T1 T2 T3 T4 T5 T6"
fi

report_info "GPU memory: ${GPU_MEM_MB} MB"
report_info "Tests: ${TESTS_TO_RUN}"
report_info "Repeats: ${N_REPEATS}"
echo ""

# ── Run benchmarks ──────────────────────────────────────────────────────────

run_benchmark() {
    local label="$1"
    local binary="$2"
    local test_name="$3"
    local out_dir="$4"
    local repeat="$5"

    mkdir -p "${out_dir}"

    "${binary}" \
        --test "${test_name}" \
        --device device \
        --precision float \
        --output "${out_dir}" \
        --repeats 1 \
        > "${out_dir}/run_${repeat}.txt" 2>&1

    # Extract elapsed_ms from metadata file
    local elapsed
    elapsed=$(grep "elapsed_ms=" "${out_dir}/${test_name}_meta.txt" 2>/dev/null | cut -d= -f2)
    if [[ -z "${elapsed}" ]]; then
        elapsed="0"
    fi

    echo "${label},${test_name},${repeat},${elapsed}" >> "${TIMING_CSV}"
    echo "${elapsed}"
}

# Warm up GPU with a quick run of each binary
report_info "Warming up GPU..."
"${UPSTREAM_BIN}" --test T1 --device device --precision float --output "${RESULTS}/warmup" --repeats 1 > /dev/null 2>&1 || true
"${OPTIMISED_BIN}" --test T1 --device device --precision float --output "${RESULTS}/warmup" --repeats 1 > /dev/null 2>&1 || true
rm -rf "${RESULTS}/warmup"

declare -A UPSTREAM_TIMINGS
declare -A OPTIMISED_TIMINGS

for TEST in ${TESTS_TO_RUN}; do
    report_info "--- ${TEST} ---"

    # Run upstream
    UPSTREAM_TIMES=()
    for r in $(seq 1 "${N_REPEATS}"); do
        elapsed=$(run_benchmark "upstream" "${UPSTREAM_BIN}" "${TEST}" "${RESULTS}/upstream_${TEST}" "${r}")
        UPSTREAM_TIMES+=("${elapsed}")
        report_info "  Upstream  run ${r}: ${elapsed} ms"
    done

    # Run optimised
    OPTIMISED_TIMES=()
    for r in $(seq 1 "${N_REPEATS}"); do
        elapsed=$(run_benchmark "optimised" "${OPTIMISED_BIN}" "${TEST}" "${RESULTS}/optimised_${TEST}" "${r}")
        OPTIMISED_TIMES+=("${elapsed}")
        report_info "  Optimised run ${r}: ${elapsed} ms"
    done

    # Store for summary table
    UPSTREAM_TIMINGS[${TEST}]="${UPSTREAM_TIMES[*]}"
    OPTIMISED_TIMINGS[${TEST}]="${OPTIMISED_TIMES[*]}"

    echo ""
done

# ── Comparison table ────────────────────────────────────────────────────────

echo "--- Upstream vs Optimised ---" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"

printf "%-6s  %-14s  %-14s  %-14s  %-10s\n" \
    "Test" "Upstream(ms)" "Optimised(ms)" "Diff(ms)" "Speedup" >> "${SUMMARY}"
printf "%-6s  %-14s  %-14s  %-14s  %-10s\n" \
    "------" "-----------" "-----------" "-----------" "-------" >> "${SUMMARY}"

python3 -c "
import csv, statistics

# Parse timings
upstream = {}
optimised = {}
with open('${TIMING_CSV}') as f:
    for row in csv.DictReader(f):
        label = row['binary']
        test = row['test']
        elapsed = float(row['elapsed_ms'])
        bucket = upstream if label == 'upstream' else optimised
        bucket.setdefault(test, []).append(elapsed)

# Comparison
tests = sorted(set(list(upstream.keys()) + list(optimised.keys())))
total_up = 0
total_opt = 0

for test in tests:
    if test in upstream and test in optimised:
        u_mean = statistics.mean(upstream[test])
        o_mean = statistics.mean(optimised[test])
        diff = u_mean - o_mean
        speedup = u_mean / o_mean if o_mean > 0 else 0
        total_up += u_mean
        total_opt += o_mean
        print(f'{test:<6}  {u_mean:<14.1f}  {o_mean:<14.1f}  {diff:<14.1f}  {speedup:<.2f}x')

if total_opt > 0:
    print(f'{\"\":<6}  {\"\":<14}  {\"\":<14}  {\"\":<14}  {\"\":<10}')
    total_diff = total_up - total_opt
    total_speedup = total_up / total_opt
    print(f'{\"Total\":<6}  {total_up:<14.1f}  {total_opt:<14.1f}  {total_diff:<14.1f}  {total_speedup:<.2f}x')
" >> "${SUMMARY}" 2>/dev/null || true

echo "" >> "${SUMMARY}"
echo "Generated: $(date -Iseconds)" >> "${SUMMARY}"

# Display
report_info ""
report_info "Comparison table:"
echo ""
cat "${SUMMARY}"

report_info ""
report_info "CSV: ${TIMING_CSV}"
report_info "Summary: ${SUMMARY}"

report_pass "Upstream vs Optimised benchmark complete"
