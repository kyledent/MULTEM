/*
 * Thrust CPU stub: execution policies
 *
 * On CPU-only builds, all execution policies resolve to sequential host execution.
 * thrust::device is an alias for thrust::host (no GPU).
 */
#ifndef THRUST_CPU_EXECUTION_POLICY_H
#define THRUST_CPU_EXECUTION_POLICY_H

namespace thrust {

struct host_execution_policy {};
struct device_execution_policy {};

// On CPU-only builds, both host and device resolve to the same thing
static constexpr host_execution_policy host{};
static constexpr device_execution_policy device{};

// Overloads that accept an execution policy and ignore it (delegate to default)
// These match the patterns: thrust::transform(thrust::device, ...)
template <typename InputIt, typename OutputIt, typename UnaryOp>
OutputIt transform(device_execution_policy, InputIt first, InputIt last,
                   OutputIt d_first, UnaryOp op) {
    return std::transform(first, last, d_first, op);
}

template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
OutputIt transform(device_execution_policy, InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, OutputIt d_first, BinaryOp op) {
    return std::transform(first1, last1, first2, d_first, op);
}

}  // namespace thrust

#endif
