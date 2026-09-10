#!/usr/bin/env bash
# 01_build.sh -- Build test harness from upstream (unmodified) MULTEM source.
#
# Uses -DMULTEM_SOURCE_ROOT to point CMake at the upstream python-multem
# clone instead of the optimized fork. Builds into a separate directory
# (build_upstream) so both binaries can coexist.
#
# PASS if test_multem_upstream binary is produced and executable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

report_header "Build Upstream Baseline (Unmodified MULTEM)"

RESULTS=$(setup_results_dir upstream_baseline build)

UPSTREAM_DIR="${TESTS_DIR}/upstream_python_multem"

# Verify upstream source exists
if [[ ! -d "${UPSTREAM_DIR}/MULTEM/src" ]]; then
    report_fail "Upstream source not found. Run 00_clone_upstream.sh first."
    exit 1
fi

# Record environment
{
    print_machine_info
    print_gpu_info
    echo "Source: upstream (unmodified)"
    echo "MULTEM_SOURCE_ROOT: ${UPSTREAM_DIR}"
    echo "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES:-auto-detect}"
} > "${RESULTS}/environment.txt"

cat "${RESULTS}/environment.txt"

# Build into separate directory to avoid clobbering the optimized build
BUILD_DIR="${TESTS_DIR}/build_upstream"
report_info "Build directory: ${BUILD_DIR}"

if [[ -d "${BUILD_DIR}" ]]; then
    report_info "Removing previous upstream build..."
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# Configure -- point at upstream source
report_info "Running CMake configure (upstream source)..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DMULTEM_SOURCE_ROOT="${UPSTREAM_DIR}"
    -DMULTEM_UPSTREAM_SRC="${UPSTREAM_DIR}/MULTEM/src"   # pristine upstream, not this repo's src
)

# Auto-detect GPU architecture if not specified
if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
elif command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | head -1 | tr -d '. ')
    if [[ -n "${COMPUTE_CAP}" ]]; then
        report_info "Auto-detected GPU compute capability: sm_${COMPUTE_CAP}"
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${COMPUTE_CAP}")
    fi
fi

# Pass CUDA host compiler if set
if [[ -n "${CMAKE_CUDA_HOST_COMPILER:-}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER}")
fi

# Auto-detect CUDA compiler if not on PATH
if ! command -v nvcc &> /dev/null; then
    for CUDA_DIR in /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda; do
        if [[ -x "${CUDA_DIR}/bin/nvcc" ]]; then
            report_info "Found nvcc at ${CUDA_DIR}/bin/nvcc"
            CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER="${CUDA_DIR}/bin/nvcc")
            break
        fi
    done
fi

cmake -S "${TESTS_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}" \
    > "${RESULTS}/cmake.log" 2>&1
CMAKE_RC=$?

if [[ ${CMAKE_RC} -ne 0 ]]; then
    report_fail "CMake configure failed (exit ${CMAKE_RC})"
    report_info "See: ${RESULTS}/cmake.log"
    cat "${RESULTS}/cmake.log"
    exit 1
fi

report_info "CMake configure OK"

# Build
report_info "Building upstream baseline..."
NPROC=$(nproc 2>/dev/null || echo 4)

cmake --build "${BUILD_DIR}" -j "${NPROC}" \
    > "${RESULTS}/build.log" 2>&1
BUILD_RC=$?

if [[ ${BUILD_RC} -ne 0 ]]; then
    report_fail "Build failed (exit ${BUILD_RC})"
    report_info "See: ${RESULTS}/build.log"
    tail -30 "${RESULTS}/build.log"
    exit 1
fi

# Rename binary to avoid confusion
if [[ -x "${BUILD_DIR}/test_multem" ]]; then
    cp "${BUILD_DIR}/test_multem" "${BUILD_DIR}/test_multem_upstream"
    report_pass "Upstream binary: ${BUILD_DIR}/test_multem_upstream"
    ls -lh "${BUILD_DIR}/test_multem_upstream" | tee -a "${RESULTS}/environment.txt"
else
    report_fail "Binary not found at ${BUILD_DIR}/test_multem"
    exit 1
fi
