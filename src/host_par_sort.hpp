#ifndef HOST_PAR_SORT_H
#define HOST_PAR_SORT_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

#include "phase_timer.hpp"

namespace mt
{
	// Host-thread sort for the atom arrays, selected at runtime by MULTEM_PAR_SORT=1.
	//
	// The sequential path is thrust's stable radix sort over an 8-way zip: single-threaded, while
	// the GPU waits, and on a large specimen the largest single cost before the first slice. This
	// sorts a compact (key, original index) array on several threads -- a radix sort per thread, then
	// pairwise merges split across all the threads by merge path -- and applies the resulting
	// permutation to each field once.
	//
	// It reproduces the sequential order element for element, not merely an equivalent one. Keys
	// are compared in the radix sort's own bit encoding (so -0.0 sorts below +0.0, as it does
	// there), and equal keys are ordered by original index, which is where a stable sort leaves
	// them. The result is therefore independent of the thread count, and can be checked exactly:
	// MULTEM_SORT_CHECK=1 runs both and counts differing atoms.

	// Threads this process may use: its affinity mask where the OS exposes one -- a container or
	// batch slot given 8 of 64 cores must not start 64 threads -- else hardware_concurrency.
	// MULTEM_SORT_THREADS overrides. The cap is a guard against oversubscription, not a tuned value.
	inline int host_sort_threads()
	{
		if(const char *e = std::getenv("MULTEM_SORT_THREADS"))
		{
			const int n = std::atoi(e);
			if(n > 0) return n;
		}
		int n = 0;
#ifdef __linux__
		cpu_set_t set;
		if(sched_getaffinity(0, sizeof(set), &set) == 0) n = CPU_COUNT(&set);
#endif
		if(n < 1) n = static_cast<int>(std::thread::hardware_concurrency());
		return std::max(1, std::min(n, 16));
	}

	// fn(ithread) on nthreads threads, the calling thread included.
	template <class TFn>
	void host_parallel_for(int nthreads, TFn fn)
	{
		std::vector<std::thread> pool;
		pool.reserve(nthreads > 1 ? nthreads - 1 : 0);
		for(int t = 1; t < nthreads; t++) pool.emplace_back(fn, t);
		fn(0);
		for(auto &th : pool) th.join();
	}

	// The radix sort's key encoding: an unsigned integer whose order is the floating-point order,
	// with -0.0 below +0.0.
	inline uint32_t radix_order_key(float f)
	{
		uint32_t u;
		std::memcpy(&u, &f, sizeof u);
		return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
	}

	inline uint64_t radix_order_key(double f)
	{
		uint64_t u;
		std::memcpy(&u, &f, sizeof u);
		return (u & 0x8000000000000000ull) ? ~u : (u | 0x8000000000000000ull);
	}

	// Merge sorted A and B into out on nthreads threads. Each thread writes an equal share of the
	// output, having found by binary search how much of that share comes from A (merge path).
	template <class TItem, class TLess>
	void host_parallel_merge(const TItem *A, std::size_t la, const TItem *B, std::size_t lb,
		TItem *out, int nthreads, TLess less)
	{
		// how many of the first d outputs come from A, when A wins ties as in std::merge
		auto from_a = [&](std::size_t d)
		{
			std::size_t lo = (d > lb) ? d - lb : 0, hi = std::min(d, la);
			while(lo < hi)
			{
				const std::size_t i = lo + (hi - lo)/2;
				if(!less(B[d - i - 1], A[i])) lo = i + 1; else hi = i;
			}
			return lo;
		};

		const std::size_t n = la + lb;
		host_parallel_for(nthreads, [&](int t)
		{
			const std::size_t d0 = n*t/nthreads, d1 = n*(t + 1)/nthreads;
			const std::size_t i0 = from_a(d0), i1 = from_a(d1);
			std::merge(A + i0, A + i1, B + (d0 - i0), B + (d1 - i1), out + d0, less);
		});
	}

	// Stable LSD radix sort of n items by their unsigned key k, one byte per pass, using tmp (n items)
	// as scratch. Stable, so items entering in index order leave ordered by (key, index) -- the same
	// order the merge comparator imposes. A pass whose byte is the same for every item would only copy
	// the array and is skipped. It replaces a comparison sort per chunk, which at few threads could be
	// slower than the sequential radix sort it was meant to beat (n log n per chunk against n overall).
	template <class TItem>
	void host_radix_sort_items(TItem *a, TItem *tmp, std::size_t n)
	{
		if(n < 2) return;
		using TCode = decltype(a->k);
		TItem *src = a, *dst = tmp;
		for(int pass = 0; pass < int(sizeof(TCode)); pass++)
		{
			const int shift = 8*pass;
			std::size_t count[257] = {0};
			for(std::size_t i = 0; i < n; i++) count[((src[i].k >> shift) & 0xFF) + 1]++;
			if(count[((src[0].k >> shift) & 0xFF) + 1] == n) continue;
			for(int d = 0; d < 256; d++) count[d + 1] += count[d];
			for(std::size_t i = 0; i < n; i++) dst[count[(src[i].k >> shift) & 0xFF]++] = src[i];
			std::swap(src, dst);
		}
		if(src != a) std::copy(src, src + n, a);
	}

	// The permutation that stably sorts key[0, n): key[perm[0]] <= key[perm[1]] <= ...
	template <class TKey>
	std::unique_ptr<uint32_t[]> host_stable_sort_permutation(const TKey *key, std::size_t n, int nthreads)
	{
		using TCode = decltype(radix_order_key(TKey()));
		struct Item { TCode k; uint32_t i; };
		auto less = [](const Item &a, const Item &b){ return a.k < b.k || (a.k == b.k && a.i < b.i); };

		// new[] rather than vector(n): the buffers are overwritten in full, so zero-filling them
		// first would be a wasted single-threaded pass over the whole array
		std::unique_ptr<Item[]> a(new Item[n]), b(new Item[n]);

		std::vector<std::size_t> runs(nthreads + 1);
		for(int t = 0; t <= nthreads; t++) runs[t] = n*t/nthreads;

		{
			PhaseTimer _t("  sort: keys + chunk sorts");
			host_parallel_for(nthreads, [&](int t)
			{
				for(std::size_t j = runs[t]; j < runs[t + 1]; j++)
				{
					a[j] = Item{radix_order_key(key[j]), static_cast<uint32_t>(j)};
				}
				// b is free until the merges, so its matching range is this chunk's scratch
				host_radix_sort_items(a.get() + runs[t], b.get() + runs[t], runs[t + 1] - runs[t]);
			});
		}

		// merge neighbouring runs until one is left; every merge uses all the threads
		{
			PhaseTimer _t("  sort: merges");
			while(runs.size() > 2)
			{
				std::vector<std::size_t> next;
				for(std::size_t r = 0; r + 1 < runs.size(); r += 2)
				{
					next.push_back(runs[r]);
					if(r + 2 < runs.size())
					{
						host_parallel_merge(a.get() + runs[r], runs[r + 1] - runs[r],
							a.get() + runs[r + 1], runs[r + 2] - runs[r + 1], b.get() + runs[r], nthreads, less);
					}
					else
					{
						std::copy(a.get() + runs[r], a.get() + runs[r + 1], b.get() + runs[r]);
					}
				}
				next.push_back(n);
				runs.swap(next);
				std::swap(a, b);
			}
		}

		std::unique_ptr<uint32_t[]> perm(new uint32_t[n]);
		host_parallel_for(nthreads, [&](int t)
		{
			for(std::size_t j = n*t/nthreads; j < n*(t + 1)/nthreads; j++) perm[j] = a[j].i;
		});
		return perm;
	}
}

#endif
