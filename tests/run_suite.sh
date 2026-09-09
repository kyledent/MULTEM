#!/usr/bin/env bash
# run_suite.sh -- Run all tests in a named suite directory.
#
# Usage:
#   ./run_suite.sh baseline           # Run all baseline tests
#   ./run_suite.sh baseline 01 02     # Run only tests 01 and 02
#   ./run_suite.sh --list baseline    # List available tests without running
#
# Test scripts are discovered by globbing for [0-9][0-9]_*.sh in the suite
# directory, and executed in lexicographic order.
#
# Exit code: 0 if all tests pass, 1 if any test fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

LIST_ONLY=0
if [[ "${1:-}" == "--list" ]]; then
    LIST_ONLY=1
    shift
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--list] <suite_name> [test_numbers...]"
    echo ""
    echo "Available suites:"
    for d in "${SCRIPT_DIR}"/*/; do
        [[ -d "${d}" ]] || continue
        name=$(basename "${d}")
        [[ "${name}" == "results" ]] && continue
        echo "  ${name}"
    done
    exit 1
fi

SUITE_NAME="$1"
shift
SUITE_DIR="${SCRIPT_DIR}/${SUITE_NAME}"

if [[ ! -d "${SUITE_DIR}" ]]; then
    echo "Error: Suite directory not found: ${SUITE_DIR}"
    exit 1
fi

# Collect test scripts
TEST_FILTER=("$@")

mapfile -t ALL_SCRIPTS < <(
    find "${SUITE_DIR}" -maxdepth 1 -name '[0-9][0-9]_*.sh' -type f | sort
)

if [[ ${#ALL_SCRIPTS[@]} -eq 0 ]]; then
    echo "Error: No test scripts found in ${SUITE_DIR}/"
    exit 1
fi

# Filter if specific test numbers were given
SCRIPTS=()
if [[ ${#TEST_FILTER[@]} -gt 0 ]]; then
    for script in "${ALL_SCRIPTS[@]}"; do
        base=$(basename "${script}")
        num="${base:0:2}"
        for filt in "${TEST_FILTER[@]}"; do
            filt_pad=$(printf '%02d' "${filt}" 2>/dev/null || echo "${filt}")
            if [[ "${num}" == "${filt_pad}" ]]; then
                SCRIPTS+=("${script}")
                break
            fi
        done
    done
else
    SCRIPTS=("${ALL_SCRIPTS[@]}")
fi

# ---------------------------------------------------------------------------
# List mode
# ---------------------------------------------------------------------------

if [[ ${LIST_ONLY} -eq 1 ]]; then
    echo "Suite: ${SUITE_NAME}"
    echo ""
    printf "  %-6s  %s\n" "ID" "Script"
    printf "  %-6s  %s\n" "----" "------"
    for script in "${ALL_SCRIPTS[@]}"; do
        base=$(basename "${script}")
        num="${base:0:2}"
        printf "  %-6s  %s\n" "${num}" "${base}"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if [[ -t 1 ]]; then
    C_GREEN="\033[1;32m" C_RED="\033[1;31m" C_YELLOW="\033[1;33m"
    C_BOLD="\033[1m" C_RESET="\033[0m"
else
    C_GREEN="" C_RED="" C_YELLOW="" C_BOLD="" C_RESET=""
fi

echo ""
echo -e "${C_BOLD}MULTEM Test Suite: ${SUITE_NAME}${C_RESET}"
echo -e "${C_BOLD}$(printf '=%.0s' {1..50})${C_RESET}"
echo ""

total=0
passed=0
failed=0
results=()

for script in "${SCRIPTS[@]}"; do
    base=$(basename "${script}")
    num="${base:0:2}"
    total=$(( total + 1 ))

    echo -e "${C_BOLD}--- [${num}] ${base} ---${C_RESET}"

    rc=0
    bash "${script}" || rc=$?

    if [[ ${rc} -eq 0 ]]; then
        passed=$(( passed + 1 ))
        results+=("${C_GREEN}PASS${C_RESET}  ${base}")
    else
        failed=$(( failed + 1 ))
        results+=("${C_RED}FAIL${C_RESET}  ${base}")
    fi

    echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo -e "${C_BOLD}$(printf '=%.0s' {1..50})${C_RESET}"
echo -e "${C_BOLD}Summary: ${SUITE_NAME}${C_RESET}"
echo ""

for r in "${results[@]}"; do
    echo -e "  ${r}"
done

echo ""
echo -e "  Total:   ${total}"
echo -e "  Passed:  ${C_GREEN}${passed}${C_RESET}"
if [[ ${failed} -gt 0 ]]; then
    echo -e "  Failed:  ${C_RED}${failed}${C_RESET}"
fi

echo ""

if [[ ${failed} -gt 0 ]]; then
    echo -e "${C_RED}SUITE FAILED${C_RESET}"
    exit 1
else
    echo -e "${C_GREEN}ALL TESTS PASSED${C_RESET}"
    exit 0
fi
