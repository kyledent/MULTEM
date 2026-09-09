#!/usr/bin/env bash
# 00_build.sh -- Build the MULTEM test harness with the async streaming changes.
#
# Double-buffered async atom streaming + batch size 2048.
# Built on top of Memory coalescing + CUB + __ldg + Occupancy.
#
# PASS if test_multem binary is produced and executable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

report_header "Build MULTEM Test Harness (Async Atom Streaming)"

RESULTS=$(setup_results_dir async_streaming build)

# Record environment
{
    print_machine_info
    print_gpu_info
    echo "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES:-auto-detect}"
    echo "Optimisation: double-buffered async atom streaming (on top of coalescing + CUB reductions + occupancy)"
} > "${RESULTS}/environment.txt"

cat "${RESULTS}/environment.txt"

# Clean build
BUILD_DIR="${TESTS_DIR}/build"
report_info "Build directory: ${BUILD_DIR}"

if [[ -d "${BUILD_DIR}" ]]; then
    report_info "Removing previous build..."
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# Configure
report_info "Running CMake configure..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
)

# Auto-detect GPU architecture if not specified
if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
elif command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [[ -n "${COMPUTE_CAP}" ]]; then
        report_info "Auto-detected GPU compute capability: sm_${COMPUTE_CAP}"
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${COMPUTE_CAP}")
    fi
fi

# Pass CUDA host compiler if set (needed on some systems)
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
report_info "Building..."
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

# Verify binary
if [[ -x "${BUILD_DIR}/test_multem" ]]; then
    report_pass "Binary produced: ${BUILD_DIR}/test_multem"
    ls -lh "${BUILD_DIR}/test_multem" | tee -a "${RESULTS}/environment.txt"
else
    report_fail "Binary not found at ${BUILD_DIR}/test_multem"
    exit 1
fi
