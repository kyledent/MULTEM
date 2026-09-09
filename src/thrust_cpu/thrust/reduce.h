/*
 * Thrust CPU stub: thrust::reduce with optional OpenMP parallelism
 */
#ifndef THRUST_CPU_REDUCE_H
#define THRUST_CPU_REDUCE_H

#include <numeric>
#include <functional>
#include <iterator>

// Minimum elements to justify OpenMP dispatch overhead
#ifndef MULTEM_OMP_THRESHOLD
#define MULTEM_OMP_THRESHOLD 4096
#endif

namespace thrust {

namespace detail_reduce {
    template <typename InputIt, typename T, typename BinaryOp>
    T reduce_impl(InputIt first, InputIt last, T init, BinaryOp op) {
#ifdef MULTEM_USE_OPENMP
        auto n = std::distance(first, last);
        if (n < MULTEM_OMP_THRESHOLD) return std::accumulate(first, last, init, op);
        if (n <= 0) return init;
        T result = init;
        #pragma omp parallel
        {
            T local = T{};
            #pragma omp for schedule(static) nowait
            for (decltype(n) i = 0; i < n; ++i) {
                local = op(local, *(first + i));
            }
            #pragma omp critical
            {
                result = op(result, local);
            }
        }
        return result;
#else
        return std::accumulate(first, last, init, op);
#endif
    }
}

// reduce with default plus
template <typename InputIt>
typename std::iterator_traits<InputIt>::value_type
reduce(InputIt first, InputIt last) {
    using T = typename std::iterator_traits<InputIt>::value_type;
    return detail_reduce::reduce_impl(first, last, T{}, std::plus<T>{});
}

// reduce with init value
template <typename InputIt, typename T>
T reduce(InputIt first, InputIt last, T init) {
    return detail_reduce::reduce_impl(first, last, init, std::plus<T>{});
}

// reduce with init value and binary op
template <typename InputIt, typename T, typename BinaryOp>
T reduce(InputIt first, InputIt last, T init, BinaryOp op) {
    return detail_reduce::reduce_impl(first, last, init, op);
}

}  // namespace thrust

#endif
