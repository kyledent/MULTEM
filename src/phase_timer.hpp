#ifndef PHASE_TIMER_H
#define PHASE_TIMER_H

#include <chrono>
#include <cstdio>
#include <cstdlib>
#ifdef __linux__
#include <sys/resource.h>
#endif

namespace mt
{
	// Scoped phase timing, off unless MULTEM_PHASE_TIMING=1 in the environment.
	//
	// A CUDA profile can place a host cost only between two CUDA calls; this says which part of
	// that gap is which, which is the difference between a target and a guess. The guard is an
	// environment variable rather than a compile flag so an instrumented build and a normal one
	// are the same binary -- the comparison then cannot be confounded by a rebuild.
	//
	// MULTEM_PHASE_FAULTS=1 (Linux) also reports the page faults taken inside the region, on a
	// separate "[faults]" line so the "[phase]" lines keep their format. Time that a loop spends
	// faulting in memory is not time spent on its own work, and the two look the same on a clock.
	struct PhaseTimer
	{
		const char *name;
		bool on;
		bool faults;
		long minflt0, majflt0;
		std::chrono::high_resolution_clock::time_point t0;

		static bool fault_counts(long &minflt, long &majflt)
		{
#ifdef __linux__
			rusage u;
#ifdef RUSAGE_THREAD
			if(getrusage(RUSAGE_THREAD, &u) != 0) return false;
#else
			if(getrusage(RUSAGE_SELF, &u) != 0) return false;
#endif
			minflt = u.ru_minflt;
			majflt = u.ru_majflt;
			return true;
#else
			(void)minflt; (void)majflt;
			return false;
#endif
		}

		explicit PhaseTimer(const char *n) : name(n), faults(false), minflt0(0), majflt0(0)
		{
			const char *e = std::getenv("MULTEM_PHASE_TIMING");
			on = (e && e[0] == '1');
			if(!on) return;
			const char *f = std::getenv("MULTEM_PHASE_FAULTS");
			faults = (f && f[0] == '1') && fault_counts(minflt0, majflt0);
			t0 = std::chrono::high_resolution_clock::now();
		}

		~PhaseTimer()
		{
			if(!on) return;
			auto ms = std::chrono::duration<double, std::milli>(
				std::chrono::high_resolution_clock::now() - t0).count();
			fprintf(stderr, "  [phase] %-28s %9.1f ms\n", name, ms);
			long minflt1, majflt1;
			if(faults && fault_counts(minflt1, majflt1))
			{
				fprintf(stderr, "  [faults] %-27s %9ld minor %6ld major\n", name, minflt1 - minflt0, majflt1 - majflt0);
			}
		}
	};
}

#endif
