/*
 * Thrust CPU stub: thrust::tuple -> std::tuple
 */
#ifndef THRUST_CPU_TUPLE_H
#define THRUST_CPU_TUPLE_H

#include <tuple>

namespace thrust {

template <typename... Types>
using tuple = std::tuple<Types...>;

template <typename... Types>
auto make_tuple(Types&&... args) {
    return std::make_tuple(std::forward<Types>(args)...);
}

template <std::size_t I, typename... Types>
auto get(const std::tuple<Types...>& t) -> decltype(std::get<I>(t)) {
    return std::get<I>(t);
}

template <std::size_t I, typename... Types>
auto get(std::tuple<Types...>& t) -> decltype(std::get<I>(t)) {
    return std::get<I>(t);
}

}  // namespace thrust

#endif
