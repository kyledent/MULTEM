#!/usr/bin/env bash
# Build and run the band-limited potential unit test. No GPU needed: MULTEM's headers are compiled
# for the host by g++ with the CPU-only shims (src/cuda_stubs, src/thrust_cpu).
#
#   bash tests/bandlimited_potential/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${HERE}/../../src"

OUT="${TMPDIR:-/tmp}/test_bandlimited_potential"
g++ -std=c++17 -O2 -pthread -DMULTEM_NO_CUDA \
    -I"${SRC}/cuda_stubs" -I"${SRC}/thrust_cpu" -I"${SRC}" \
    "${HERE}/test_bandlimited_potential.cpp" -o "${OUT}"
"${OUT}"
