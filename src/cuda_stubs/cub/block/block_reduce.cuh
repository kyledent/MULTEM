/*
 * CUB stub: BlockReduce for CPU-only builds.
 */
#ifndef CUB_BLOCK_REDUCE_STUB_H
#define CUB_BLOCK_REDUCE_STUB_H

namespace cub {

template <typename T, int BLOCK_SIZE>
struct BlockReduce {
    struct TempStorage {};

    T val_;
    BlockReduce(TempStorage&) : val_(0) {}

    T Sum(T val) { return val; }
    T Reduce(T val) { return val; }
};

}  // namespace cub

#endif
