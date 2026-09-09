/*
 * Thrust CPU stub: thrust::raw_pointer_cast -> identity (std::vector::data())
 *
 * For std::vector, raw_pointer_cast(&v[0]) is just &v[0].
 */
#ifndef THRUST_CPU_RAW_POINTER_CAST_H
#define THRUST_CPU_RAW_POINTER_CAST_H

namespace thrust {

template <typename T>
T* raw_pointer_cast(T* ptr) {
    return ptr;
}

}  // namespace thrust

#endif
