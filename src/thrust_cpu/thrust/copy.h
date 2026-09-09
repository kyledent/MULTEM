/*
 * Thrust CPU stub: thrust::copy -> std::copy
 */
#ifndef THRUST_CPU_COPY_H
#define THRUST_CPU_COPY_H

#include <algorithm>

namespace thrust {

template <typename InputIt, typename OutputIt>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
    return std::copy(first, last, d_first);
}

}  // namespace thrust

#endif
