/*
 * Thrust CPU stub: thrust::max_element/min_element -> std:: equivalents
 */
#ifndef THRUST_CPU_EXTREMA_H
#define THRUST_CPU_EXTREMA_H

#include <algorithm>

namespace thrust {

template <typename ForwardIt>
ForwardIt max_element(ForwardIt first, ForwardIt last) {
    return std::max_element(first, last);
}

template <typename ForwardIt, typename Compare>
ForwardIt max_element(ForwardIt first, ForwardIt last, Compare comp) {
    return std::max_element(first, last, comp);
}

template <typename ForwardIt>
ForwardIt min_element(ForwardIt first, ForwardIt last) {
    return std::min_element(first, last);
}

template <typename ForwardIt, typename Compare>
ForwardIt min_element(ForwardIt first, ForwardIt last, Compare comp) {
    return std::min_element(first, last, comp);
}

template <typename ForwardIt>
std::pair<ForwardIt, ForwardIt> minmax_element(ForwardIt first, ForwardIt last) {
    return std::minmax_element(first, last);
}

template <typename ForwardIt, typename Compare>
std::pair<ForwardIt, ForwardIt> minmax_element(ForwardIt first, ForwardIt last, Compare comp) {
    return std::minmax_element(first, last, comp);
}

}  // namespace thrust

#endif
