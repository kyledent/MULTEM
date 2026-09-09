/*
 * Thrust CPU stub: counting_iterator
 *
 * An iterator that returns incrementing values: 0, 1, 2, ...
 */
#ifndef THRUST_CPU_COUNTING_ITERATOR_H
#define THRUST_CPU_COUNTING_ITERATOR_H

#include <iterator>

namespace thrust {

template <typename T>
class counting_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = T;

    counting_iterator() : value_(0) {}
    explicit counting_iterator(T value) : value_(value) {}

    reference operator*() const { return value_; }
    reference operator[](difference_type n) const { return value_ + n; }

    counting_iterator& operator++() { ++value_; return *this; }
    counting_iterator operator++(int) { auto tmp = *this; ++value_; return tmp; }
    counting_iterator& operator--() { --value_; return *this; }

    counting_iterator operator+(difference_type n) const { return counting_iterator(value_ + n); }
    counting_iterator operator-(difference_type n) const { return counting_iterator(value_ - n); }
    difference_type operator-(const counting_iterator& o) const { return value_ - o.value_; }

    counting_iterator& operator+=(difference_type n) { value_ += n; return *this; }
    counting_iterator& operator-=(difference_type n) { value_ -= n; return *this; }

    bool operator==(const counting_iterator& o) const { return value_ == o.value_; }
    bool operator!=(const counting_iterator& o) const { return value_ != o.value_; }
    bool operator<(const counting_iterator& o) const { return value_ < o.value_; }

private:
    T value_;
};

}  // namespace thrust

#endif
