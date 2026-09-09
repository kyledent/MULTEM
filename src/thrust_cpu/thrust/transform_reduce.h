/*
 * Thrust CPU stub: thrust::transform_reduce with optional OpenMP parallelism
 */
#ifndef THRUST_CPU_TRANSFORM_REDUCE_H
#define THRUST_CPU_TRANSFORM_REDUCE_H

#include <numeric>
#include <functional>
#include <iterator>

// Minimum elements to justify OpenMP dispatch overhead
#ifndef MULTEM_OMP_THRESHOLD
#define MULTEM_OMP_THRESHOLD 4096
#endif

namespace thrust {

template <typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
T transform_reduce(InputIt first, InputIt last, UnaryOp unary_op, T init, BinaryOp binary_op) {
#ifdef MULTEM_USE_OPENMP
    auto n = std::distance(first, last);
    if (n < MULTEM_OMP_THRESHOLD) {
        T result = init;
        for (; first != last; ++first) {
            result = binary_op(result, unary_op(*first));
        }
        return result;
    }
    if (n <= 0) return init;
    T result = init;
    #pragma omp parallel
    {
        T local = T{};
        #pragma omp for schedule(static) nowait
        for (decltype(n) i = 0; i < n; ++i) {
            local = binary_op(local, unary_op(*(first + i)));
        }
        #pragma omp critical
        {
            result = binary_op(result, local);
        }
    }
    return result;
#else
    T result = init;
    for (; first != last; ++first) {
        result = binary_op(result, unary_op(*first));
    }
    return result;
#endif
}

}  // namespace thrust

#endif
