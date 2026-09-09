/*
 * Thrust CPU stub: zip_iterator
 *
 * Provides zip_iterator for CPU-only builds. Returns tuples of references
 * so that assignment through operator* works (needed for sort_by_key).
 */
#ifndef THRUST_CPU_ZIP_ITERATOR_H
#define THRUST_CPU_ZIP_ITERATOR_H

#include <thrust/tuple.h>
#include <iterator>

namespace thrust {

// Helper: build a tuple of reference types from a tuple of iterator types
namespace detail_zip {
    template <typename IterTuple, std::size_t... Is>
    auto deref_impl(const IterTuple& iters, std::index_sequence<Is...>) {
        return std::tie(*std::get<Is>(iters)...);
    }

    template <typename IterTuple, std::size_t... Is>
    auto deref_val_impl(const IterTuple& iters, std::index_sequence<Is...>) {
        return std::make_tuple(*std::get<Is>(iters)...);
    }

    template <typename IterTuple, std::size_t... Is>
    void advance_impl(IterTuple& iters, std::ptrdiff_t n, std::index_sequence<Is...>) {
        ((std::advance(std::get<Is>(iters), n)), ...);
    }
}

template <typename IteratorTuple>
class zip_iterator {
    static constexpr std::size_t N = std::tuple_size<IteratorTuple>::value;
    using Seq = std::make_index_sequence<N>;
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    // value_type: tuple of values (for temp storage in algorithms)
    using value_type = decltype(detail_zip::deref_val_impl(
        std::declval<const IteratorTuple&>(), Seq{}));
    // reference: tuple of references (for assignment)
    using reference = decltype(detail_zip::deref_impl(
        std::declval<const IteratorTuple&>(), Seq{}));
    using pointer = void;

    IteratorTuple iterators_;

    zip_iterator() = default;
    explicit zip_iterator(IteratorTuple iters) : iterators_(iters) {}

    // Dereference returns tuple of references (assignable)
    reference operator*() const {
        return detail_zip::deref_impl(iterators_, Seq{});
    }

    // Dereference at offset
    value_type operator[](difference_type n) const {
        auto tmp = *this + n;
        return detail_zip::deref_val_impl(tmp.iterators_, Seq{});
    }

    zip_iterator& operator++() {
        detail_zip::advance_impl(iterators_, 1, Seq{});
        return *this;
    }

    zip_iterator operator++(int) {
        zip_iterator tmp(*this);
        ++(*this);
        return tmp;
    }

    zip_iterator& operator--() {
        detail_zip::advance_impl(iterators_, -1, Seq{});
        return *this;
    }

    zip_iterator& operator+=(difference_type n) {
        detail_zip::advance_impl(iterators_, n, Seq{});
        return *this;
    }

    zip_iterator operator+(difference_type n) const {
        zip_iterator result(*this);
        result += n;
        return result;
    }

    zip_iterator operator-(difference_type n) const {
        zip_iterator result(*this);
        result += -n;
        return result;
    }

    difference_type operator-(const zip_iterator& other) const {
        return std::get<0>(iterators_) - std::get<0>(other.iterators_);
    }

    bool operator!=(const zip_iterator& other) const {
        return std::get<0>(iterators_) != std::get<0>(other.iterators_);
    }

    bool operator==(const zip_iterator& other) const {
        return std::get<0>(iterators_) == std::get<0>(other.iterators_);
    }

    bool operator<(const zip_iterator& other) const {
        return std::get<0>(iterators_) < std::get<0>(other.iterators_);
    }
};

template <typename... Iterators>
zip_iterator<std::tuple<Iterators...>>
make_zip_iterator(std::tuple<Iterators...> t) {
    return zip_iterator<std::tuple<Iterators...>>(t);
}

}  // namespace thrust

#endif
