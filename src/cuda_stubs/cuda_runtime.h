/*
 * CUDA runtime stub for CPU-only builds (Apple Silicon, etc.)
 *
 * Provides minimal type definitions and no-op functions so that MULTEM
 * source files that include <cuda_runtime.h> can compile without CUDA.
 * Only the host code path (e_host) is functional.
 */
#ifndef CUDA_RUNTIME_STUB_H
#define CUDA_RUNTIME_STUB_H

#include <cstddef>

// CUDA qualifiers (no-ops for CPU-only builds)
#ifndef __global__
#define __global__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __syncthreads
inline void __syncthreads() {}
#endif

// Atomic stubs
inline int atomicAdd(int* addr, int val) { int old = *addr; *addr += val; return old; }
inline float atomicAdd(float* addr, float val) { float old = *addr; *addr += val; return old; }
inline double atomicAdd(double* addr, double val) { double old = *addr; *addr += val; return old; }

// Thread indexing stubs (never used in CPU path but must compile)
struct __stub_dim3 { unsigned int x, y, z; };
#ifndef threadIdx
static __stub_dim3 threadIdx = {0, 0, 0};
static __stub_dim3 blockIdx = {0, 0, 0};
static __stub_dim3 blockDim = {1, 1, 1};
static __stub_dim3 gridDim = {1, 1, 1};
#endif

// Error types
enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorNoDevice = 100
};

inline const char* cudaGetErrorString(cudaError_t) {
    return "CUDA not available (CPU-only build)";
}

inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceCount(int* count) {
    *count = 0;
    return cudaErrorNoDevice;
}

inline cudaError_t cudaSetDevice(int) {
    return cudaErrorNoDevice;
}

inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

// Memory stubs (should never be called in CPU-only path)
inline cudaError_t cudaMalloc(void**, size_t) { return cudaErrorNoDevice; }
inline cudaError_t cudaFree(void*) { return cudaErrorNoDevice; }
inline cudaError_t cudaMallocHost(void**, size_t) { return cudaErrorNoDevice; }
inline cudaError_t cudaFreeHost(void*) { return cudaErrorNoDevice; }

// Stream stubs
typedef void* cudaStream_t;
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaErrorNoDevice; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaErrorNoDevice; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

// Memcpy stubs
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

inline cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) {
    return cudaErrorNoDevice;
}

inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t = 0) {
    return cudaErrorNoDevice;
}

// dim3 type (used in kernel launch configuration structs)
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

// Device properties stub
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int multiProcessorCount;
    int major;
    int minor;
};

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int) {
    return cudaErrorNoDevice;
}

// Memory info
inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    *free = 0;
    *total = 0;
    return cudaErrorNoDevice;
}

inline cudaError_t cudaMemset(void*, int, size_t) { return cudaErrorNoDevice; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t = 0) { return cudaErrorNoDevice; }

// Event stubs
typedef void* cudaEvent_t;
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0.0f; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }

// Host memory registration stubs
enum { cudaHostRegisterDefault = 0, cudaHostRegisterPortable = 1, cudaHostRegisterMapped = 2 };
inline cudaError_t cudaHostRegister(void*, size_t, unsigned int) { return cudaSuccess; }
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }

// Pinned memory stubs (expanded)
inline cudaError_t cudaMallocManaged(void**, size_t, unsigned int = 0) { return cudaErrorNoDevice; }

#endif
