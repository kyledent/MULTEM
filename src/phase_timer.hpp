#ifndef PHASE_TIMER_H
#define PHASE_TIMER_H

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace mt
{
	// Scoped phase timing, off unless MULTEM_PHASE_TIMING=1 in the environment.
	//
	// A CUDA profile can place a host cost only between two CUDA calls; this says which part of
	// that gap is which, which is the difference between a target and a guess. The guard is an
	// environment variable rather than a compile flag so an instrumented build and a normal one
	// are the same binary -- the comparison then cannot be confounded by a rebuild.
	struct PhaseTimer
	{
		const char *name;
		bool on;
		std::chrono::high_resolution_clock::time_point t0;

		explicit PhaseTimer(const char *n) : name(n)
		{
			const char *e = std::getenv("MULTEM_PHASE_TIMING");
			on = (e && e[0] == '1');
			if(on) t0 = std::chrono::high_resolution_clock::now();
		}

		~PhaseTimer()
		{
			if(!on) return;
			auto ms = std::chrono::duration<double, std::milli>(
				std::chrono::high_resolution_clock::now() - t0).count();
			fprintf(stderr, "  [phase] %-28s %9.1f ms\n", name, ms);
		}
	};
}

#endif
