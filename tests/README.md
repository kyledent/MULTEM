# MULTEM test harness

A build / correctness / benchmark harness for the GPU optimisations on this branch.
Each optimisation has its own suite; every suite builds the source, checks its output
against the baseline reference, and times the same six test cases, so a change can be
shown to be both correct and faster before it is believed.

## Layout

| Suite | What it builds and measures |
|---|---|
| `baseline/` | Unmodified source. Establishes the reference output and the reference timings every other suite compares against. |
| `coalesce/` | Memory coalescing — 2D thread-block dimensions matched to the warp size. |
| `reductions/` | `cub::BlockReduce` in place of hand-rolled shared-memory reductions, plus `__ldg()` read-only loads. |
| `occupancy/` | `cudaOccupancyMaxPotentialBlockSize` launch configuration. |
| `async_streaming/` | Double-buffered atom streaming on a second CUDA stream, with GPU-resident atom arrays. |
| `tiled_potential/` | Tiled potential accumulation with GPU spatial binning, to cut `atomicAdd` contention. |
| `pinned_fused/` | Pinned staging buffers, and a fused transmission-multiply kernel. |
| `upstream_baseline/` | Clones unmodified upstream and benchmarks it, as an external control on the in-tree baseline. |

Each suite has `00_build.sh`, `01_validate_correctness.sh` (or `01_validate_reference.sh`
for the baseline) and `03_benchmark.sh`, runnable individually or through `run_suite.sh`.

## Requirements

- **A sibling `python-multem` checkout.** `tests/CMakeLists.txt` takes its FFTW finder from
  `../../python-multem/cmake`, and `test_multem.cu` drives MULTEM through python-multem's
  `multem/multem_ext.h`. Clone https://github.com/rosalindfranklininstitute/python-multem
  next to this repository.
- CMake ≥ 3.17 and FFTW. CUDA is **optional**: it is auto-detected, and
  `-DMULTEM_ENABLE_CUDA=OFF` forces a host-only build using the shims in
  `src/cuda_stubs/` and `src/thrust_cpu/`.

## Running

```sh
cd tests
./run_suite.sh baseline          # first: reference output + reference timings
./run_suite.sh reductions        # then any optimisation suite
```

Results are written under `tests/results/<suite>/`, or wherever `RESULTS_DIR` points.
Correctness is a comparison against the baseline reference arrays captured on the
baseline suite's first run, to 1e-5 — relaxed to 1e-4 for the fused kernel, which
changes the order of evaluation rather than the arithmetic.

Everything the harness produces — built binaries, reference arrays, timings, FFTW
wisdom, the upstream clone — is regenerable and git-ignored. Only the scripts and
`test_multem.cu` are tracked.

## Test cases

`T1`–`T6` are multislice runs of increasing size. `T7`/`T8` exist in the source but are
commented out: they need more than 8 GB of device memory. Restore them when benchmarking
on a larger-memory GPU.
