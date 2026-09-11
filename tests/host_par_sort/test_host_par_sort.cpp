// Unit test for src/host_par_sort.hpp: the parallel sort must reproduce thrust's sequential
// sort_by_key order EXACTLY -- the property the atom sort relies on -- for every thread count,
// every size including the degenerate ones, and the inputs that separate a stable sort from an
// unstable one (heavy ties) and a bit-level key order from a value-level one (-0.0 vs +0.0).
//
//   bash tests/host_par_sort/run.sh
//
// The reference is the real thing, not a re-implementation: thrust's host sort_by_key on a zip of
// (original index, key copy), which is exactly the call Atom_Data::sort_by_z makes.

#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "host_par_sort.hpp"

template <class T>
std::vector<uint32_t> reference_order(const std::vector<T> &key)
{
	thrust::host_vector<T> z(key.begin(), key.end()), zc(key.begin(), key.end());
	thrust::host_vector<uint32_t> idx(key.size());
	for(std::size_t i = 0; i < key.size(); i++) idx[i] = static_cast<uint32_t>(i);
	thrust::sort_by_key(z.begin(), z.end(), thrust::make_zip_iterator(thrust::make_tuple(idx.begin(), zc.begin())));
	return std::vector<uint32_t>(idx.begin(), idx.end());
}

template <class T>
std::vector<T> make_keys(const std::string &kind, std::size_t n, std::mt19937_64 &rng)
{
	std::vector<T> k(n);
	std::uniform_real_distribution<double> u(-100.0, 100.0);
	std::uniform_int_distribution<int> small(0, 9), four(0, 3);
	for(std::size_t i = 0; i < n; i++)
	{
		if(kind == "uniform") k[i] = static_cast<T>(u(rng));
		else if(kind == "ties") k[i] = static_cast<T>(small(rng)) * T(0.5);
		else if(kind == "signed_zero")
		{
			const T v[4] = {T(-0.0), T(0.0), T(1.0), T(-1.0)};
			k[i] = v[four(rng)];
		}
		else if(kind == "sorted") k[i] = static_cast<T>(i / 7);
		else k[i] = static_cast<T>((n - i) / 7);          // "reverse", with ties
	}
	return k;
}

template <class T>
int run_cases(const char *tname, int &n_cases)
{
	int failures = 0;
	std::mt19937_64 rng(2026);
	const std::size_t sizes[] = {0, 1, 2, 3, 7, 1000, 65537, 250000};
	const int threads[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 17};
	const char *kinds[] = {"uniform", "ties", "signed_zero", "sorted", "reverse"};

	for(auto kind : kinds)
	for(auto n : sizes)
	{
		const auto key = make_keys<T>(kind, n, rng);
		const auto ref = reference_order(key);
		for(auto nt : threads)
		{
			n_cases++;
			auto perm = mt::host_stable_sort_permutation(key.data(), n, nt);
			// Applying the permutation to the atom fields is Atom_Data's job (a packed gather) and is
			// checked there, on every field of every atom, by MULTEM_SORT_CHECK=1.
			std::size_t bad = 0;
			for(std::size_t j = 0; j < n; j++) bad += (perm[j] != ref[j]);

			if(bad)
			{
				failures++;
				printf("FAIL %s %-11s n=%-7zu threads=%-2d : %zu mismatches\n", tname, kind, n, nt, bad);
			}
		}
	}
	return failures;
}

// Negative control: the comparison above must be able to fail. Swap two tied entries of a correct
// answer -- a stable-vs-unstable difference, invisible to any check of sortedness -- and require
// that it is caught.
int negative_control()
{
	std::mt19937_64 rng(7);
	const auto key = make_keys<float>("ties", 1000, rng);
	const auto ref = reference_order(key);
	auto perm = mt::host_stable_sort_permutation(key.data(), key.size(), 4);
	std::size_t j = 0;
	while(j + 1 < key.size() && key[perm[j]] != key[perm[j + 1]]) j++;
	std::swap(perm[j], perm[j + 1]);
	std::size_t bad = 0;
	for(std::size_t i = 0; i < key.size(); i++) bad += (perm[i] != ref[i]);
	printf("negative control: a swapped tie gives %zu mismatches (must be > 0)\n", bad);
	return bad > 0 ? 0 : 1;
}

int main()
{
	int n_cases = 0;
	int failures = run_cases<float>("float ", n_cases) + run_cases<double>("double", n_cases);
	failures += negative_control();
	printf("%d cases: %s\n", n_cases, failures ? "FAILED" : "all identical to thrust's sequential sort_by_key");
	return failures ? 1 : 0;
}
