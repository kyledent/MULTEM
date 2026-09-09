#!/usr/bin/env bash
# 01_validate_correctness.sh -- Validate occupancy tuning output against baseline reference.
#
# Occupancy tuning changes:
# - cudaOccupancyMaxPotentialBlockSize replaces hardcoded thread block sizes
# - Thread block x-dimension stays 32 for coalesced memory access (coalescing)
# - Block y-dimension adjusted per-kernel based on register/shared memory usage
#
# For EWRS tests, output should be BIT-IDENTICAL to baseline since:
# - Only launch configuration changes, not computation
# - Same data accessed in same order within each warp
#
# PASS if all test cases match baseline reference within tolerance.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

require_binary
require_gpu

report_header "Correctness Validation (Occupancy vs Baseline)"

RESULTS=$(setup_results_dir occupancy correctness)

# Check that baseline reference exists
BASELINE_REF="${RESULTS_DIR}/baseline/reference"
if [[ ! -d "${BASELINE_REF}" ]]; then
    report_fail "Baseline reference not found at ${BASELINE_REF}"
    report_info "Run baseline suite first: ./run_suite.sh baseline"
    exit 1
fi

# Determine which tests to run based on GPU memory
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' \r' || echo 0)
report_info "GPU memory: ${GPU_MEM_MB} MB"

TESTS_TO_RUN="T1 T2 T3 T4"
if [[ ${GPU_MEM_MB} -ge 4000 ]]; then
    TESTS_TO_RUN="T1 T2 T3 T4 T5"
fi
if [[ ${GPU_MEM_MB} -ge 6000 ]]; then
    TESTS_TO_RUN="T1 T2 T3 T4 T5 T6"
fi
# T7/T8 removed: OOM on an 8 GB GPU. Restore for >=16 GB GPUs.

report_info "Running tests: ${TESTS_TO_RUN}"

ALL_PASSED=true

for TEST in ${TESTS_TO_RUN}; do
    report_info ""
    report_info "--- ${TEST} ---"

    TEST_DIR="${RESULTS}/${TEST}"
    mkdir -p "${TEST_DIR}"

    # Run test the occupancy tuning build
    "${MULTEM_TEST_BIN}" \
        --test "${TEST}" \
        --device device \
        --precision float \
        --output "${TEST_DIR}" \
        > "${TEST_DIR}/stdout.txt" 2>&1
    RC=$?

    if [[ ${RC} -ne 0 ]]; then
        report_fail "${TEST}: test harness failed (exit ${RC})"
        cat "${TEST_DIR}/stdout.txt"
        ALL_PASSED=false
        continue
    fi

    # Check output files exist
    if [[ ! -f "${TEST_DIR}/${TEST}_real.bin" ]]; then
        report_fail "${TEST}: no output files produced"
        cat "${TEST_DIR}/stdout.txt"
        ALL_PASSED=false
        continue
    fi

    # Find baseline reference for this test
    REF_DIR="${BASELINE_REF}/reference_${TEST}"
    if [[ ! -d "${REF_DIR}" ]]; then
        REF_DIR="${BASELINE_REF}/${TEST}"
    fi

    if [[ ! -d "${REF_DIR}" ]]; then
        report_fail "${TEST}: baseline reference not found"
        ALL_PASSED=false
        continue
    fi

    # Compare against baseline.
    # Occupancy tuning changes block sizes, which may change thread-to-element
    # mapping and thus atomicAdd ordering, potentially causing small FP diffs.
    # Use tolerance 1e-5 as safety margin.
    MATCH=true
    BIT_IDENTICAL=true
    for suffix in _real.bin _imag.bin _m2psi.bin; do
        ACTUAL="${TEST_DIR}/${TEST}${suffix}"
        EXPECTED="${REF_DIR}/${TEST}${suffix}"

        if [[ -f "${ACTUAL}" && -f "${EXPECTED}" ]]; then
            if ! compare_arrays "${ACTUAL}" "${EXPECTED}" "0" 2>/dev/null; then
                BIT_IDENTICAL=false
            fi
            if ! compare_arrays "${ACTUAL}" "${EXPECTED}" "1e-5"; then
                report_fail "${TEST}: ${suffix} exceeds tolerance vs baseline"
                MATCH=false
            fi
        fi
    done

    if $MATCH && $BIT_IDENTICAL; then
        report_pass "${TEST}: BIT-IDENTICAL to baseline"
    elif $MATCH; then
        report_pass "${TEST}: within tolerance (1e-5)"
    else
        ALL_PASSED=false
    fi

    # Print summary from stdout
    grep -E "^  (Time|Grid|Atoms|Slices|psi_coh|m2psi_tot):" "${TEST_DIR}/stdout.txt" || true
done

echo ""
if $ALL_PASSED; then
    report_pass "All correctness tests passed"
    exit 0
else
    report_fail "Some correctness tests failed"
    exit 1
fi
