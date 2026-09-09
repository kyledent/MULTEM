/*
 * Thrust CPU stub: random number generators -> <random>
 */
#ifndef THRUST_CPU_RANDOM_H
#define THRUST_CPU_RANDOM_H

#include <random>

namespace thrust {

using default_random_engine = std::mt19937;

template <typename T = double>
using uniform_real_distribution = std::uniform_real_distribution<T>;

template <typename T = int>
using uniform_int_distribution = std::uniform_int_distribution<T>;

}  // namespace thrust

#endif
