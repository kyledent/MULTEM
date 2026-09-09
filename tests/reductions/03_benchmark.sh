#!/usr/bin/env bash
# 03_benchmark.sh -- Timed benchmarks for CUB reductions + __ldg.
#
# Runs each test case with N repeats, records wall-clock time per run,
# writes a summary table, and compares against baseline timings.
#
# PASS always (timing-only, no correctness checks).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

require_binary
require_gpu

report_header "GPU Benchmarks (CUB Reductions + __ldg)"

RESULTS=$(setup_results_dir reductions benchmark)
TIMING_LOG="${RESULTS}/timings.csv"
SUMMARY="${RESULTS}/benchmark.txt"

N_REPEATS="${BENCHMARK_REPEATS:-3}"

# Record machine info
{
    print_machine_info
    print_gpu_info
    echo "Optimisation: CUB BlockReduce + __ldg() (on top of coalescing coalescing)"
} > "${SUMMARY}"

echo "Repeats: ${N_REPEATS}" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"

# CSV header
echo "test,repeat,elapsed_ms" > "${TIMING_LOG}"

# Determine which tests to run based on GPU memory
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' \r' || echo 0)

TESTS_TO_RUN="T1 T2 T3 T4"
if [[ ${GPU_MEM_MB} -ge 4000 ]]; then
    TESTS_TO_RUN="T1 T2 T3 T4 T5"
fi
if [[ ${GPU_MEM_MB} -ge 6000 ]]; then
    TESTS_TO_RUN="T1 T2 T3 T4 T5 T6"
fi
# T7/T8 removed: OOM on an 8 GB GPU. Restore for >=16 GB GPUs.

report_info "Tests: ${TESTS_TO_RUN}"
report_info "Repeats: ${N_REPEATS}"
echo ""

# Table header
printf "%-6s  " "Test" >> "${SUMMARY}"
for r in $(seq 1 "${N_REPEATS}"); do
    printf "%-12s  " "Run${r}(ms)" >> "${SUMMARY}"
done
printf "%-12s  %-12s\n" "Mean(ms)" "Std(ms)" >> "${SUMMARY}"

printf "%-6s  " "------" >> "${SUMMARY}"
for r in $(seq 1 "${N_REPEATS}"); do
    printf "%-12s  " "----------" >> "${SUMMARY}"
done
printf "%-12s  %-12s\n" "----------" "----------" >> "${SUMMARY}"

for TEST in ${TESTS_TO_RUN}; do
    report_info "--- ${TEST} ---"

    TIMINGS=()
    TEST_DIR="${RESULTS}/${TEST}"
    mkdir -p "${TEST_DIR}"

    for r in $(seq 1 "${N_REPEATS}"); do
        # Run test (suppress output, capture timing from metadata)
        "${MULTEM_TEST_BIN}" \
            --test "${TEST}" \
            --device device \
            --precision float \
            --output "${TEST_DIR}" \
            --repeats 1 \
            > "${TEST_DIR}/run_${r}.txt" 2>&1

        # Extract elapsed_ms from metadata file
        ELAPSED=$(grep "elapsed_ms=" "${TEST_DIR}/${TEST}_meta.txt" 2>/dev/null | cut -d= -f2)
        if [[ -z "${ELAPSED}" ]]; then
            ELAPSED="0"
        fi

        TIMINGS+=("${ELAPSED}")
        echo "${TEST},${r},${ELAPSED}" >> "${TIMING_LOG}"
        report_info "  Run ${r}: ${ELAPSED} ms"
    done

    # Compute mean and std via Python
    STATS=$(python3 -c "
import statistics
t = [float(x) for x in '${TIMINGS[*]}'.split()]
mean = statistics.mean(t)
std = statistics.stdev(t) if len(t) > 1 else 0
print(f'{mean:.1f} {std:.1f}')
")

    MEAN=$(echo "${STATS}" | cut -d' ' -f1)
    STD=$(echo "${STATS}" | cut -d' ' -f2)

    # Write table row
    printf "%-6s  " "${TEST}" >> "${SUMMARY}"
    for t in "${TIMINGS[@]}"; do
        printf "%-12s  " "${t}" >> "${SUMMARY}"
    done
    printf "%-12s  %-12s\n" "${MEAN}" "${STD}" >> "${SUMMARY}"

    echo ""
done

echo "" >> "${SUMMARY}"

# Compare against baseline if available
BASELINE_TIMING="${RESULTS_DIR}/baseline/benchmark/timings.csv"
if [[ -f "${BASELINE_TIMING}" ]]; then
    echo "--- Comparison vs Baseline ---" >> "${SUMMARY}"
    echo "" >> "${SUMMARY}"
    printf "%-6s  %-12s  %-12s  %-12s\n" "Test" "Baseline(ms)" "Reduce(ms)" "Speedup" >> "${SUMMARY}"
    printf "%-6s  %-12s  %-12s  %-12s\n" "------" "----------" "----------" "----------" >> "${SUMMARY}"

    python3 -c "
import csv, statistics

# Parse baseline
baseline = {}
with open('${BASELINE_TIMING}') as f:
    for row in csv.DictReader(f):
        test = row['test']
        if test not in baseline:
            baseline[test] = []
        baseline[test].append(float(row['elapsed_ms']))

# Parse the CUB reductions timings
opt = {}
with open('${TIMING_LOG}') as f:
    for row in csv.DictReader(f):
        test = row['test']
        if test not in opt:
            opt[test] = []
        opt[test].append(float(row['elapsed_ms']))

# Compare
for test in sorted(opt.keys()):
    if test in baseline:
        b_mean = statistics.mean(baseline[test])
        p_mean = statistics.mean(opt[test])
        speedup = b_mean / p_mean if p_mean > 0 else 0
        print(f'{test:<6}  {b_mean:<12.1f}  {p_mean:<12.1f}  {speedup:<12.2f}x')
" >> "${SUMMARY}" 2>/dev/null || true

    echo "" >> "${SUMMARY}"
fi

echo "Generated: $(date -Iseconds)" >> "${SUMMARY}"

report_info "Timing table:"
cat "${SUMMARY}"

report_info ""
report_info "CSV: ${TIMING_LOG}"
report_info "Summary: ${SUMMARY}"

report_pass "Benchmark complete"
