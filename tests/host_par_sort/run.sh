#!/usr/bin/env bash
# Build and run the host_par_sort unit test against thrust's host sort_by_key. No GPU needed:
# thrust is compiled for the host by g++, with its device system set to the host too.
#
#   bash tests/host_par_sort/run.sh          (CUDA_HOME selects the thrust headers)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${HERE}/../../src"
CUDA_INC="${CUDA_HOME:-/usr/local/cuda}/include"
[ -f "${CUDA_INC}/thrust/sort.h" ] || { echo "FATAL: no thrust under ${CUDA_INC} -- set CUDA_HOME" >&2; exit 1; }

OUT="${TMPDIR:-/tmp}/test_host_par_sort"
g++ -std=c++17 -O2 -pthread -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP \
    -I"${CUDA_INC}" -I"${SRC}" "${HERE}/test_host_par_sort.cpp" -o "${OUT}"
"${OUT}"
