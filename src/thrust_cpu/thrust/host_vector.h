/*
 * Thrust CPU stub: host_vector
 *
 * Provides a minimal thrust::host_vector that exposes T* as its iterator type,
 * matching real Thrust behavior. This is critical because MULTEM's matlab_types.cuh
 * returns raw pointers from begin()/end() typed as host_vector<T>::iterator.
 *
 * Built on top of std::vector but with iterator = T* (via data() pointer).
 */
#ifndef THRUST_CPU_HOST_VECTOR_H
#define THRUST_CPU_HOST_VECTOR_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <initializer_list>

namespace thrust {

// Make std::swap available as thrust::swap (real Thrust provides this)
using std::swap;

template <typename T, typename Alloc = std::allocator<T>>
class host_vector {
public:
    // Match Thrust's iterator = T* (not std::vector<T>::iterator)
    using value_type      = T;
    using allocator_type  = Alloc;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;
    using iterator        = T*;
    using const_iterator  = const T*;

    host_vector() = default;
    explicit host_vector(size_type n) : v_(n) {}
    host_vector(size_type n, const T& val) : v_(n, val) {}

    template <typename InputIt>
    host_vector(InputIt first, InputIt last) : v_(first, last) {}

    host_vector(std::initializer_list<T> init) : v_(init) {}
    host_vector(const host_vector&) = default;
    host_vector(host_vector&&) noexcept = default;
    host_vector& operator=(const host_vector&) = default;
    host_vector& operator=(host_vector&&) noexcept = default;

    // Cross-assign from any other host_vector (used by thrust assign patterns)
    template <typename U, typename A>
    void assign(const U* first, const U* last) {
        v_.assign(first, last);
    }

    // assign from iterators (used by thrust D2H copies: M_h.assign(M_d.begin(), M_d.end()))
    template <typename InputIt>
    void assign(InputIt first, InputIt last) {
        v_.assign(first, last);
    }

    // fill-assign: assign(count, value)
    void assign(size_type count, const T& value) {
        v_.assign(count, value);
    }

    // Iterators — return raw pointers (matching real Thrust)
    iterator       begin()       { return v_.data(); }
    const_iterator begin() const { return v_.data(); }
    iterator       end()         { return v_.data() + v_.size(); }
    const_iterator end()   const { return v_.data() + v_.size(); }

    // Element access
    reference       operator[](size_type i)       { return v_[i]; }
    const_reference operator[](size_type i) const { return v_[i]; }
    reference       front()       { return v_.front(); }
    const_reference front() const { return v_.front(); }
    reference       back()        { return v_.back(); }
    const_reference back()  const { return v_.back(); }
    pointer         data()        { return v_.data(); }
    const_pointer   data()  const { return v_.data(); }

    // Capacity
    size_type size()     const { return v_.size(); }
    bool      empty()    const { return v_.empty(); }
    size_type capacity() const { return v_.capacity(); }
    void reserve(size_type n) { v_.reserve(n); }

    // Modifiers
    void clear()                { v_.clear(); }
    void resize(size_type n)    { v_.resize(n); }
    void resize(size_type n, const T& val) { v_.resize(n, val); }
    void push_back(const T& val) { v_.push_back(val); }
    void push_back(T&& val)      { v_.push_back(std::move(val)); }
    template <typename... Args>
    void emplace_back(Args&&... args) { v_.emplace_back(std::forward<Args>(args)...); }
    void pop_back()             { v_.pop_back(); }
    void shrink_to_fit()        { v_.shrink_to_fit(); }
    void swap(host_vector& other) noexcept { v_.swap(other.v_); }
    iterator erase(const_iterator pos) {
        auto offset = pos - begin();
        auto it = v_.erase(v_.begin() + offset);
        return v_.data() + (it - v_.begin());
    }
    iterator erase(const_iterator first, const_iterator last) {
        auto off_first = first - begin();
        auto off_last = last - begin();
        auto it = v_.erase(v_.begin() + off_first, v_.begin() + off_last);
        return v_.data() + (it - v_.begin());
    }
    iterator insert(const_iterator pos, const T& val) {
        auto offset = pos - begin();
        auto it = v_.insert(v_.begin() + offset, val);
        return v_.data() + (it - v_.begin());
    }
    template <typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last) {
        auto offset = pos - begin();
        auto it = v_.insert(v_.begin() + offset, first, last);
        return v_.data() + (it - v_.begin());
    }

private:
    std::vector<T, Alloc> v_;
};

// Specialization for bool: std::vector<bool> is bit-packed so data()/pointer
// iterators don't work.  Use std::vector<char> internally and cast.
template <typename Alloc>
class host_vector<bool, Alloc> {
public:
    using value_type      = bool;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = bool&;
    using const_reference = const bool&;
    // Use char storage so data() works; cast through reinterpret isn't safe
    // so fall back to std::vector iterators for the bool case.
    using iterator        = std::vector<bool>::iterator;
    using const_iterator  = std::vector<bool>::const_iterator;

    host_vector() = default;
    explicit host_vector(size_type n) : v_(n) {}
    host_vector(size_type n, bool val) : v_(n, val) {}
    host_vector(const host_vector&) = default;
    host_vector(host_vector&&) noexcept = default;
    host_vector& operator=(const host_vector&) = default;
    host_vector& operator=(host_vector&&) noexcept = default;

    template <typename InputIt>
    void assign(InputIt first, InputIt last) { v_.assign(first, last); }
    void assign(size_type count, bool value) { v_.assign(count, value); }

    iterator       begin()       { return v_.begin(); }
    const_iterator begin() const { return v_.begin(); }
    iterator       end()         { return v_.end(); }
    const_iterator end()   const { return v_.end(); }

    std::vector<bool>::reference       operator[](size_type i)       { return v_[i]; }
    std::vector<bool>::const_reference operator[](size_type i) const { return v_[i]; }

    size_type size()     const { return v_.size(); }
    bool      empty()    const { return v_.empty(); }
    void clear()                { v_.clear(); }
    void resize(size_type n)    { v_.resize(n); }
    void resize(size_type n, bool val) { v_.resize(n, val); }
    void push_back(bool val)    { v_.push_back(val); }
    void pop_back()             { v_.pop_back(); }
    void swap(host_vector& other) noexcept { v_.swap(other.v_); }

private:
    std::vector<bool> v_;
};

}  // namespace thrust

// Transitively include commonly-used algorithm stubs (matching real Thrust behavior)
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#endif
