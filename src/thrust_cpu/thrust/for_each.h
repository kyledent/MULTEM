/*
 * Thrust CPU stub: thrust::for_each with optional OpenMP parallelism
 */
#ifndef THRUST_CPU_FOR_EACH_H
#define THRUST_CPU_FOR_EACH_H

#include <algorithm>
#include <iterator>

// Minimum elements to justify OpenMP dispatch overhead
#ifndef MULTEM_OMP_THRESHOLD
#define MULTEM_OMP_THRESHOLD 4096
#endif

namespace thrust {

template <typename InputIt, typename UnaryFunction>
InputIt for_each(InputIt first, InputIt last, UnaryFunction f) {
#ifdef MULTEM_USE_OPENMP
    auto n = std::distance(first, last);
    if (n >= MULTEM_OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (decltype(n) i = 0; i < n; ++i) {
            f(*(first + i));
        }
        return last;
    }
#endif
    return std::for_each(first, last, f);
}

}  // namespace thrust

#endif
