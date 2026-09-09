/*
 * Thrust CPU stub: thrust::complex -> std::complex
 */
#ifndef THRUST_CPU_COMPLEX_H
#define THRUST_CPU_COMPLEX_H

#include <complex>

namespace thrust {

template <typename T>
using complex = std::complex<T>;

using std::norm;
using std::polar;
using std::arg;
using std::abs;

}  // namespace thrust

#endif
