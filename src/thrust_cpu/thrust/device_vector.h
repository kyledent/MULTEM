/*
 * Thrust CPU stub: device_vector -> std::vector (no GPU)
 *
 * On CPU-only builds, device_vector is an alias for host_vector (std::vector).
 * All operations run on the host.
 */
#ifndef THRUST_CPU_DEVICE_VECTOR_H
#define THRUST_CPU_DEVICE_VECTOR_H

#include <thrust/host_vector.h>

namespace thrust {

template <typename T, typename Alloc = std::allocator<T>>
using device_vector = host_vector<T, Alloc>;

// device_reference: on CPU, just a reference wrapper
template <typename T>
using device_reference = T&;

}  // namespace thrust

#endif
