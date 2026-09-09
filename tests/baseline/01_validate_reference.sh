#!/usr/bin/env bash
# 01_validate_reference.sh -- Run test cases and capture reference output.
#
# Runs T1-T3 (T4/T5 are optional, require more VRAM/time) and saves the
# output arrays as reference files for future correctness validation.
#
# On first run: captures reference output.
# On subsequent runs: compares against saved reference.
#
# PASS if all test cases produce valid output (and match reference if it exists).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

require_binary
require_gpu

report_header "Reference Validation"

RESULTS=$(setup_results_dir baseline reference)

# Determine which tests to run based on available GPU memory
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' \r' || echo 0)
report_info "GPU memory: ${GPU_MEM_MB} MB"

# VRAM requirements: T1 9MB, T2 54MB, T3 215MB, T4 376MB, T5 858MB,
#                    T6 1.5GB, T7 3.4GB, T8 6GB
# Time estimates (8 GB consumer GPU): T1 ~2s, T2 ~5s, T3 ~10s, T4 ~20s, T5 ~50s,
#                             T6 ~3min, T7 ~8min, T8 ~?
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

    # Run test
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

    # Check for reference
    REF_DIR="${RESULTS}/reference_${TEST}"
    if [[ -d "${REF_DIR}" ]]; then
        # Compare against reference
        report_info "Comparing against reference..."

        MATCH=true
        for suffix in _real.bin _imag.bin; do
            ACTUAL="${TEST_DIR}/${TEST}${suffix}"
            EXPECTED="${REF_DIR}/${TEST}${suffix}"

            if [[ -f "${ACTUAL}" && -f "${EXPECTED}" ]]; then
                if ! compare_arrays "${ACTUAL}" "${EXPECTED}" "1e-5"; then
                    report_fail "${TEST}: ${suffix} differs from reference"
                    MATCH=false
                fi
            fi
        done

        if $MATCH; then
            report_pass "${TEST}: output matches reference"
        else
            ALL_PASSED=false
        fi
    else
        # First run -- save as reference
        report_info "First run: saving as reference"
        mkdir -p "${REF_DIR}"
        cp "${TEST_DIR}/${TEST}"_*.bin "${TEST_DIR}/${TEST}_meta.txt" "${REF_DIR}/" 2>/dev/null || true
        report_pass "${TEST}: reference captured"
    fi

    # Print summary from stdout
    grep -E "^  (Time|Grid|Atoms|Slices|psi_coh|m2psi_tot):" "${TEST_DIR}/stdout.txt" || true
done

echo ""
if $ALL_PASSED; then
    report_pass "All validation tests passed"
    exit 0
else
    report_fail "Some validation tests failed"
    exit 1
fi
