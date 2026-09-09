/*
 * Thrust CPU stub: thrust::transform with optional OpenMP parallelism
 */
#ifndef THRUST_CPU_TRANSFORM_H
#define THRUST_CPU_TRANSFORM_H

#include <algorithm>
#include <iterator>

// Minimum elements to justify OpenMP dispatch overhead
#ifndef MULTEM_OMP_THRESHOLD
#define MULTEM_OMP_THRESHOLD 4096
#endif

namespace thrust {

// Unary transform
template <typename InputIt, typename OutputIt, typename UnaryOp>
OutputIt transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op) {
#ifdef MULTEM_USE_OPENMP
    auto n = std::distance(first, last);
    if (n >= MULTEM_OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (decltype(n) i = 0; i < n; ++i) {
            *(d_first + i) = op(*(first + i));
        }
        return d_first + n;
    }
#endif
    return std::transform(first, last, d_first, op);
}

// Binary transform
template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
OutputIt transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                   OutputIt d_first, BinaryOp op) {
#ifdef MULTEM_USE_OPENMP
    auto n = std::distance(first1, last1);
    if (n >= MULTEM_OMP_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (decltype(n) i = 0; i < n; ++i) {
            *(d_first + i) = op(*(first1 + i), *(first2 + i));
        }
        return d_first + n;
    }
#endif
    return std::transform(first1, last1, first2, d_first, op);
}

}  // namespace thrust

#endif
