/*
 * Thrust CPU stub: thrust::fill with optional OpenMP parallelism
 */
#ifndef THRUST_CPU_FILL_H
#define THRUST_CPU_FILL_H

#include <algorithm>
#include <iterator>

// Minimum elements to justify OpenMP dispatch overhead
#ifndef MULTEM_OMP_THRESHOLD
#define MULTEM_OMP_THRESHOLD 4096
#endif

namespace thrust {

template <typename ForwardIt, typename T>
void fill(ForwardIt first, ForwardIt last, const T& value) {
#ifdef MULTEM_USE_OPENMP
    auto n = std::distance(first, last);
    if (n >= MULTEM_OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (decltype(n) i = 0; i < n; ++i) {
            *(first + i) = value;
        }
        return;
    }
#endif
    std::fill(first, last, value);
}

}  // namespace thrust

#endif
