#!/usr/bin/env bash
# 00_clone_upstream.sh -- Clone original upstream python-multem source.
#
# Clones the unmodified python-multem repo (with MULTEM submodule) so we
# can build a true baseline binary for comparison against optimized phases.
#
# PASS if upstream source tree is available with MULTEM/src/ populated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

report_header "Clone Upstream python-multem (Unmodified Baseline)"

RESULTS=$(setup_results_dir upstream_baseline clone)

UPSTREAM_REPO="https://github.com/rosalindfranklininstitute/python-multem.git"
UPSTREAM_DIR="${TESTS_DIR}/upstream_python_multem"

# Record what we're doing
{
    print_machine_info
    echo "Upstream repo:  ${UPSTREAM_REPO}"
    echo "Clone target:   ${UPSTREAM_DIR}"
} > "${RESULTS}/environment.txt"

# Check if already cloned
if [[ -d "${UPSTREAM_DIR}/MULTEM/src" ]] && [[ -f "${UPSTREAM_DIR}/src/multem/multem_ext.cu" ]]; then
    report_info "Upstream source already present at ${UPSTREAM_DIR}"
    report_info "To force re-clone, remove ${UPSTREAM_DIR} and re-run."

    # Show what commit we have
    if [[ -d "${UPSTREAM_DIR}/.git" ]]; then
        COMMIT=$(git -C "${UPSTREAM_DIR}" log --oneline -1 2>/dev/null || echo "unknown")
        report_info "Current commit: ${COMMIT}"
    fi

    report_pass "Upstream source available"
    exit 0
fi

# Clone
report_info "Cloning ${UPSTREAM_REPO}..."
report_info "This downloads ~50 MB (source + MULTEM submodule)..."

if [[ -d "${UPSTREAM_DIR}" ]]; then
    report_info "Removing incomplete previous clone..."
    rm -rf "${UPSTREAM_DIR}"
fi

git clone --depth 1 "${UPSTREAM_REPO}" "${UPSTREAM_DIR}" \
    > "${RESULTS}/clone.log" 2>&1
CLONE_RC=$?

if [[ ${CLONE_RC} -ne 0 ]]; then
    report_fail "git clone failed (exit ${CLONE_RC})"
    cat "${RESULTS}/clone.log"
    exit 1
fi

report_info "Clone OK"

# Init MULTEM submodule
report_info "Initialising MULTEM submodule..."
(
    cd "${UPSTREAM_DIR}"
    git submodule update --init --depth 1 MULTEM
) >> "${RESULTS}/clone.log" 2>&1
SUB_RC=$?

if [[ ${SUB_RC} -ne 0 ]]; then
    report_fail "Submodule init failed (exit ${SUB_RC})"
    cat "${RESULTS}/clone.log"
    exit 1
fi

# Verify the source tree has what we need
if [[ ! -d "${UPSTREAM_DIR}/MULTEM/src" ]]; then
    report_fail "MULTEM/src/ not found in upstream clone"
    exit 1
fi

if [[ ! -f "${UPSTREAM_DIR}/src/multem/multem_ext.cu" ]]; then
    report_fail "src/multem/multem_ext.cu not found in upstream clone"
    exit 1
fi

# Also need cmake/FindFFTW.cmake
if [[ ! -f "${UPSTREAM_DIR}/cmake/FindFFTW.cmake" ]]; then
    report_fail "cmake/FindFFTW.cmake not found in upstream clone"
    exit 1
fi

COMMIT=$(git -C "${UPSTREAM_DIR}" log --oneline -1 2>/dev/null || echo "unknown")
report_info "Upstream commit: ${COMMIT}"

echo "Upstream commit: ${COMMIT}" >> "${RESULTS}/environment.txt"

report_pass "Upstream source cloned and verified"
