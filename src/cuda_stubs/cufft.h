/*
 * cuFFT stub for CPU-only builds.
 *
 * Defines enough types and function signatures so that the cuFFT-based
 * FFT<T, e_device> specialization compiles (but is never instantiated —
 * CPU builds only use FFT<T, e_host> which uses FFTW).
 */
#ifndef CUFFT_STUB_H
#define CUFFT_STUB_H

typedef int cufftHandle;
typedef int cufftResult;
typedef float cufftReal;
typedef double cufftDoubleReal;

struct cufftComplex { float x, y; };
struct cufftDoubleComplex { double x, y; };

#define CUFFT_SUCCESS 0
#define CUFFT_C2C 0x29
#define CUFFT_Z2Z 0x69
#define CUFFT_R2C 0x2a
#define CUFFT_C2R 0x2c
#define CUFFT_FORWARD -1
#define CUFFT_INVERSE 1

// Stub functions (never called in CPU-only builds)
inline cufftResult cufftPlan1d(cufftHandle*, int, int, int) { return CUFFT_SUCCESS; }
inline cufftResult cufftPlan2d(cufftHandle*, int, int, int) { return CUFFT_SUCCESS; }
inline cufftResult cufftPlanMany(cufftHandle*, int, int*, int*, int, int, int*, int, int, int, int) { return CUFFT_SUCCESS; }
inline cufftResult cufftDestroy(cufftHandle) { return CUFFT_SUCCESS; }
inline cufftResult cufftSetStream(cufftHandle, void*) { return CUFFT_SUCCESS; }
inline cufftResult cufftExecC2C(cufftHandle, cufftComplex*, cufftComplex*, int) { return CUFFT_SUCCESS; }
inline cufftResult cufftExecZ2Z(cufftHandle, cufftDoubleComplex*, cufftDoubleComplex*, int) { return CUFFT_SUCCESS; }

#endif
