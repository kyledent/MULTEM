// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 Diamond Light Source
//
// This file is part of MULTEM (fork); see LICENSE.txt.

#ifndef ATOM_TYPE_CACHE_H
#define ATOM_TYPE_CACHE_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "types.cuh"

namespace mt
{
	// Process-wide cache of the atom-type tables Spec::set_input_data builds (all c_nAtomsTypes
	// elements, every call). The tables depend only on the potential parameterisation, Vrl, nR,
	// the smallest grid spacing and the band-limit spacing, so a series of calls on one grid --
	// the tilts of a tilt series, the frames of a movie -- rebuilds identical tables each time.
	//
	// MULTEM_ATOM_TYPE_CACHE=1 keeps the tables of the first call with a given key and copies them
	// on later calls. MULTEM_ATOM_TYPE_CACHE_CHECK=1 (with the cache on) builds the tables anyway on
	// a hit and reports, on stderr, how many values differ from the cached copy; it must be 0.
	// Off by default: the build is the same binary either way, so the two can be compared directly.
	namespace atom_type_cache
	{
		struct Key
		{
			int potential_type;
			double Vrl;
			int nR;
			double dR_min;
			double dR_bl;

			bool operator==(const Key &k) const
			{
				return potential_type == k.potential_type && Vrl == k.Vrl && nR == k.nR &&
					dR_min == k.dR_min && dR_bl == k.dR_bl;
			}
		};

		inline bool enabled()
		{
			const char *e = std::getenv("MULTEM_ATOM_TYPE_CACHE");
			return e && e[0] == '1';
		}

		inline bool checking()
		{
			const char *e = std::getenv("MULTEM_ATOM_TYPE_CACHE_CHECK");
			return e && e[0] == '1';
		}

		template <class TVector>
		void copy_tables(TVector &src, TVector &dst)
		{
			dst.resize(src.size());
			for(std::size_t i = 0; i < src.size(); i++)
			{
				dst[i].assign(src[i]);
			}
		}

		// One store per table type (Spec<float> and Spec<double> keep separate caches).
		template <class TVector>
		struct Store
		{
			std::mutex mutex;
			std::vector<Key> keys;
			std::vector<TVector> tables;

			static Store &instance()
			{
				static Store store;
				return store;
			}

			// Copies the cached tables for key into out; false on a miss.
			bool get(const Key &key, TVector &out)
			{
				std::lock_guard<std::mutex> lock(mutex);
				for(std::size_t i = 0; i < keys.size(); i++)
				{
					if(keys[i] == key)
					{
						copy_tables(tables[i], out);
						return true;
					}
				}
				return false;
			}

			void put(const Key &key, TVector &in)
			{
				std::lock_guard<std::mutex> lock(mutex);
				for(std::size_t i = 0; i < keys.size(); i++)
				{
					if(keys[i] == key) return;
				}
				keys.push_back(key);
				tables.emplace_back();
				copy_tables(in, tables.back());
			}
		};

		// Value-by-value comparison of two table sets, for the check mode. Returns the number of
		// differing values (sizes count as values: a size mismatch counts once).
		template <class TVec>
		std::size_t diff_vec(const TVec &a, const TVec &b)
		{
			if(a.size() != b.size()) return 1;
			std::size_t n = 0;
			for(std::size_t i = 0; i < a.size(); i++)
			{
				if(std::memcmp(&a[i], &b[i], sizeof(a[i])) != 0) n++;
			}
			return n;
		}

		template <class TPP>
		std::size_t diff_pp(const TPP &a, const TPP &b)
		{
			return diff_vec(a.cl, b.cl) + diff_vec(a.cnl, b.cnl);
		}

		template <class TVector>
		std::size_t diff_tables(const TVector &a, const TVector &b)
		{
			if(a.size() != b.size()) return 1;
			std::size_t n = 0;
			for(std::size_t i = 0; i < a.size(); i++)
			{
				const auto &x = a[i];
				const auto &y = b[i];
				n += (x.Z != y.Z) + (x.m != y.m) + (x.A != y.A) + (x.rn_e != y.rn_e) +
					(x.rn_c != y.rn_c) + (x.ra_e != y.ra_e) + (x.ra_c != y.ra_c);
				if(x.coef.size() != y.coef.size()) { n++; continue; }
				for(std::size_t j = 0; j < x.coef.size(); j++)
				{
					const auto &c = x.coef[j];
					const auto &d = y.coef[j];
					n += (c.charge != d.charge) + (c.tag != d.tag) + (c.R_min != d.R_min) +
						(c.R_max != d.R_max) + (c.R_tap != d.R_tap) + (c.tap_cf != d.tap_cf);
					n += diff_pp(c.feg, d.feg) + diff_pp(c.fxg, d.fxg) + diff_pp(c.Pr, d.Pr) +
						diff_pp(c.Vr, d.Vr) + diff_pp(c.VR, d.VR);
					n += diff_vec(c.R, d.R) + diff_vec(c.R2, d.R2);
					n += diff_vec(c.ciVR.c0, d.ciVR.c0) + diff_vec(c.ciVR.c1, d.ciVR.c1) +
						diff_vec(c.ciVR.c2, d.ciVR.c2) + diff_vec(c.ciVR.c3, d.ciVR.c3);
				}
			}
			return n;
		}
	}
}

#endif
