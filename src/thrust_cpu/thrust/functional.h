/*
 * Thrust CPU stub: thrust functors -> std:: equivalents
 */
#ifndef THRUST_CPU_FUNCTIONAL_H
#define THRUST_CPU_FUNCTIONAL_H

#include <functional>

namespace thrust {

template <typename T = void>
using plus = std::plus<T>;

template <typename T = void>
using minus = std::minus<T>;

template <typename T = void>
using multiplies = std::multiplies<T>;

template <typename T = void>
using divides = std::divides<T>;

template <typename T = void>
using negate = std::negate<T>;

template <typename T = void>
using less = std::less<T>;

template <typename T = void>
using greater = std::greater<T>;

}  // namespace thrust

#endif
