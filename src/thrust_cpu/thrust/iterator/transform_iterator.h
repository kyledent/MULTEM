/*
 * Thrust CPU stub: transform_iterator (minimal)
 */
#ifndef THRUST_CPU_TRANSFORM_ITERATOR_H
#define THRUST_CPU_TRANSFORM_ITERATOR_H

#include <iterator>

namespace thrust {

template <typename UnaryFunc, typename Iterator>
class transform_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = decltype(std::declval<UnaryFunc>()(*std::declval<Iterator>()));
    using pointer = value_type*;
    using reference = value_type;

    transform_iterator() = default;
    transform_iterator(Iterator it, UnaryFunc fn) : it_(it), fn_(fn) {}

    reference operator*() const { return fn_(*it_); }
    reference operator[](difference_type n) const { return fn_(*(it_ + n)); }

    transform_iterator& operator++() { ++it_; return *this; }
    transform_iterator operator+(difference_type n) const { return {it_ + n, fn_}; }
    difference_type operator-(const transform_iterator& o) const { return it_ - o.it_; }
    bool operator!=(const transform_iterator& o) const { return it_ != o.it_; }
    bool operator==(const transform_iterator& o) const { return it_ == o.it_; }

private:
    Iterator it_;
    UnaryFunc fn_;
};

template <typename UnaryFunc, typename Iterator>
transform_iterator<UnaryFunc, Iterator>
make_transform_iterator(Iterator it, UnaryFunc fn) {
    return {it, fn};
}

}  // namespace thrust

#endif
