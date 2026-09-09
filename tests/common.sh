#!/usr/bin/env bash
# common.sh -- Shared utilities for the MULTEM test suite.
#
# Source this file from individual test scripts:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/../common.sh"
#
# Environment variable overrides:
#   MULTEM_TEST_BIN  -- path to test harness binary
#   RESULTS_DIR      -- path to results root

set -euo pipefail

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTEM_ROOT="$(cd "${TESTS_DIR}/.." && pwd)"
MULTEM_SRC="${MULTEM_ROOT}/src"
MULTEM_BUILD="${TESTS_DIR}/build"
MULTEM_TEST_BIN="${MULTEM_TEST_BIN:-${MULTEM_BUILD}/test_multem}"
RESULTS_DIR="${RESULTS_DIR:-${TESTS_DIR}/results}"

# ---------------------------------------------------------------------------
# Terminal colours (disabled if not a tty)
# ---------------------------------------------------------------------------

if [[ -t 1 ]]; then
    CLR_GREEN="\033[1;32m"
    CLR_RED="\033[1;31m"
    CLR_YELLOW="\033[1;33m"
    CLR_CYAN="\033[1;36m"
    CLR_BOLD="\033[1m"
    CLR_RESET="\033[0m"
else
    CLR_GREEN="" CLR_RED="" CLR_YELLOW="" CLR_CYAN="" CLR_BOLD="" CLR_RESET=""
fi

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_TEST_RESULT=""

report_pass() {
    local msg="${1:-}"
    _TEST_RESULT="PASS"
    echo -e "${CLR_GREEN}[PASS]${CLR_RESET} ${msg}"
}

report_fail() {
    local msg="${1:-}"
    _TEST_RESULT="FAIL"
    echo -e "${CLR_RED}[FAIL]${CLR_RESET} ${msg}"
}

report_skip() {
    local msg="${1:-}"
    _TEST_RESULT="SKIP"
    echo -e "${CLR_YELLOW}[SKIP]${CLR_RESET} ${msg}"
}

report_info() {
    local msg="${1:-}"
    echo -e "${CLR_CYAN}[INFO]${CLR_RESET} ${msg}"
}

report_header() {
    local name="${1:-}"
    echo ""
    echo -e "${CLR_BOLD}===== ${name} =====${CLR_RESET}"
    echo ""
}

# ---------------------------------------------------------------------------
# Results directory management
# ---------------------------------------------------------------------------

setup_results_dir() {
    local suite="${1:?suite name required}"
    local test_name="${2:?test name required}"
    local dir="${RESULTS_DIR}/${suite}/${test_name}"
    mkdir -p "${dir}"
    echo "${dir}"
}

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

time_cmd() {
    local label="${1:?label required}"
    shift

    local start_seconds=${SECONDS}
    local rc=0
    "$@" || rc=$?
    local elapsed=$(( SECONDS - start_seconds ))

    local minutes=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    report_info "${label}: ${minutes}m ${secs}s (${elapsed}s wall-clock)"

    if [[ -n "${TIMING_LOG:-}" ]]; then
        echo "${label}|${elapsed}|${minutes}m${secs}s" >> "${TIMING_LOG}"
    fi

    return ${rc}
}

# ---------------------------------------------------------------------------
# Array comparison (binary files)
# ---------------------------------------------------------------------------

# compare_arrays ACTUAL EXPECTED [TOLERANCE]
#   Compares two binary float arrays element-wise.
#   TOLERANCE is max relative difference (default: 0, i.e. bit-identical).
#   Returns 0 if within tolerance, 1 otherwise.
compare_arrays() {
    local actual="${1:?actual file required}"
    local expected="${2:?expected file required}"
    local tolerance="${3:-0}"

    if [[ ! -f "${actual}" ]]; then
        echo "  Actual file not found: ${actual}"
        return 1
    fi
    if [[ ! -f "${expected}" ]]; then
        echo "  Expected file not found: ${expected}"
        return 1
    fi

    # Size check
    local size_a size_b
    size_a=$(stat -c%s "${actual}")
    size_b=$(stat -c%s "${expected}")
    if [[ "${size_a}" != "${size_b}" ]]; then
        echo "  Size mismatch: actual=${size_a} expected=${size_b}"
        return 1
    fi

    if [[ "${tolerance}" == "0" ]]; then
        # Bit-identical comparison
        if cmp -s "${actual}" "${expected}"; then
            return 0
        else
            echo "  Files differ (bit-identical check)"
            return 1
        fi
    else
        # Tolerance comparison via Python
        python3 -c "
import numpy as np, sys
a = np.fromfile('${actual}', dtype=np.float32)
b = np.fromfile('${expected}', dtype=np.float32)
if a.shape != b.shape:
    print(f'  Shape mismatch: {a.shape} vs {b.shape}')
    sys.exit(1)
max_ref = np.max(np.abs(b))
if max_ref == 0:
    max_ref = 1.0
max_diff = np.max(np.abs(a - b))
rel_diff = max_diff / max_ref
mean_diff = np.mean(np.abs(a - b))
print(f'  Max |diff|: {max_diff:.2e}  Rel: {rel_diff:.2e}  Mean |diff|: {mean_diff:.2e}')
if rel_diff > float('${tolerance}'):
    print(f'  EXCEEDS tolerance {float(\"${tolerance}\"):.1e}')
    sys.exit(1)
"
    fi
}

# ---------------------------------------------------------------------------
# Machine info
# ---------------------------------------------------------------------------

print_machine_info() {
    echo "Date:     $(date -Iseconds)"
    echo "Hostname: $(hostname)"
    echo "Kernel:   $(uname -r)"
    if [[ -f /proc/cpuinfo ]]; then
        echo "CPU:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
        echo "Cores:    $(nproc)"
    fi
    if [[ -f /proc/meminfo ]]; then
        echo "RAM:      $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
    fi
    echo "Compiler: $(g++ --version 2>/dev/null | head -1 || echo 'unknown')"
    echo ""
}

print_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Info:"
        nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
            --format=csv,noheader 2>/dev/null | while read -r line; do
            echo "  ${line}"
        done
        echo "CUDA:     $(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//' || echo 'unknown')"
    else
        echo "GPU:      none detected"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

require_binary() {
    if [[ ! -x "${MULTEM_TEST_BIN}" ]]; then
        report_fail "Test binary not found at ${MULTEM_TEST_BIN}"
        report_info "Run 00_build.sh first, or set MULTEM_TEST_BIN."
        exit 1
    fi
}

require_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        report_skip "No nvidia-smi found -- GPU required"
        exit 0
    fi
    if ! nvidia-smi &> /dev/null; then
        report_skip "nvidia-smi failed -- GPU not available"
        exit 0
    fi
}
