/*
 * cuBLAS stub for CPU-only builds.
 * Provides type definitions so blas.cuh compiles. The GPU BLAS
 * specialization is never instantiated in CPU-only mode.
 */
#ifndef CUBLAS_V2_STUB_H
#define CUBLAS_V2_STUB_H

#include <cstddef>

typedef int cublasStatus_t;
typedef void* cublasHandle_t;

enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
};

#define CUBLAS_STATUS_SUCCESS 0

inline cublasStatus_t cublasCreate(cublasHandle_t*) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }

// Stubs for geam (never called in CPU-only builds)
inline cublasStatus_t cublasSgeam(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, const float*, const float*, int, const float*, const float*, int, float*, int) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDgeam(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, const double*, const double*, int, const double*, const double*, int, double*, int) {
    return CUBLAS_STATUS_SUCCESS;
}

// Complex geam stubs
struct cuComplex { float x, y; };
struct cuDoubleComplex { double x, y; };

inline cublasStatus_t cublasCgeam(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) {
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasZgeam(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) {
    return CUBLAS_STATUS_SUCCESS;
}

#endif
