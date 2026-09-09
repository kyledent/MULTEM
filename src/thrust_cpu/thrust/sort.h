/*
 * Thrust CPU stub: thrust::sort/stable_sort -> std::sort/stable_sort
 */
#ifndef THRUST_CPU_SORT_H
#define THRUST_CPU_SORT_H

#include <algorithm>
#include <numeric>
#include <vector>

namespace thrust {

template <typename RandomIt>
void sort(RandomIt first, RandomIt last) {
    std::sort(first, last);
}

template <typename RandomIt, typename Compare>
void sort(RandomIt first, RandomIt last, Compare comp) {
    std::sort(first, last, comp);
}

template <typename RandomIt>
void stable_sort(RandomIt first, RandomIt last) {
    std::stable_sort(first, last);
}

template <typename RandomIt, typename Compare>
void stable_sort(RandomIt first, RandomIt last, Compare comp) {
    std::stable_sort(first, last, comp);
}

// sort_by_key: sort keys and rearrange values accordingly
// Uses index-based permutation that works with any random-access iterator
// (including zip_iterator which lacks a copyable value_type in std::iterator_traits)
template <typename KeyIt, typename ValueIt>
void sort_by_key(KeyIt key_first, KeyIt key_last, ValueIt value_first) {
    auto n = std::distance(key_first, key_last);
    if (n <= 1) return;

    // Build index array sorted by key
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
        [&](std::size_t a, std::size_t b) { return *(key_first + a) < *(key_first + b); });

    // Apply permutation in-place using cycle-following
    // This avoids needing value_type for temporary storage
    std::vector<bool> done(n, false);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        if (done[i] || idx[i] == i) {
            done[i] = true;
            continue;
        }
        // Save the displaced element
        auto key_tmp = *(key_first + i);
        auto val_tmp = *(value_first + i);
        std::size_t j = i;
        while (!done[j]) {
            done[j] = true;
            std::size_t k = idx[j];
            if (k == i) {
                *(key_first + j) = key_tmp;
                *(value_first + j) = val_tmp;
            } else {
                *(key_first + j) = *(key_first + k);
                *(value_first + j) = *(value_first + k);
                j = k;
            }
        }
    }
}

template <typename KeyIt, typename ValueIt, typename Compare>
void sort_by_key(KeyIt key_first, KeyIt key_last, ValueIt value_first, Compare comp) {
    auto n = std::distance(key_first, key_last);
    if (n <= 1) return;

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
        [&](std::size_t a, std::size_t b) { return comp(*(key_first + a), *(key_first + b)); });

    std::vector<bool> done(n, false);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        if (done[i] || idx[i] == i) {
            done[i] = true;
            continue;
        }
        auto key_tmp = *(key_first + i);
        auto val_tmp = *(value_first + i);
        std::size_t j = i;
        while (!done[j]) {
            done[j] = true;
            std::size_t k = idx[j];
            if (k == i) {
                *(key_first + j) = key_tmp;
                *(value_first + j) = val_tmp;
            } else {
                *(key_first + j) = *(key_first + k);
                *(value_first + j) = *(value_first + k);
                j = k;
            }
        }
    }
}

}  // namespace thrust

#endif
