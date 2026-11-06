// Unit test for the band-limited projected-potential tables (src/bandlimited_potential.hpp,
// Atomic_Data::get_atom_coef with dR_bl > 0).
//
//   bash tests/bandlimited_potential/run.sh
//
// The property that matters: the potential an atom deposits on the grid -- the sum over pixels of
// MULTEM's own evaluation (host_device_detail::eval_cubic_poly, the function the CPU and GPU paths
// both call), times the pixel area -- equals the atom's projected-potential integral c_Potf f_e(0),
// whatever the sub-pixel position of the atom and whatever the pixel size. MULTEM's own tables fail
// this at cryo-EM sampling, and the test checks that it can tell: it measures them too, and requires
// oxygen at 1.5 A/px to come out far short.
//
// Checks:
//   1. Gauss-Legendre rule: weights sum to the interval, odd powers up to 2n-1 integrate exactly.
//   2. The integral reference: c_Potf f_e(0) for H, C, N, O equals an independent evaluation of the
//      same Lobato coefficients (numpy, to the 3 decimals quoted below).
//   3. A band-limited table's own integral equals c_Potf f_e(0).
//   4. Grid sum over random sub-pixel offsets: mean within 2e-4 of 1, every offset within 1e-3,
//      for H C N O P S Fe Au at 0.05 to 4 A/px, and every element 1-103 at four pixel sizes.
//   5. The table reproduces the band-limited function it tabulates, inside the taper radius.
//   6. MULTEM's own tables (dR_bl = 0) are unchanged and still lose most of the integral at
//      1.5 A/px for oxygen -- the fault this fixes, so check 4 is not vacuous.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// the CUDA runtime first, as python-multem's multem_ext.cu does: with MULTEM_NO_CUDA it is the stub
// in src/cuda_stubs, which supplies dim3 and the runtime calls types.cuh uses
#include <cuda_runtime.h>
#include "types.cuh"
#include "atomic_data.hpp"

namespace
{
	using AtomType = mt::Atom_Type<double, mt::e_host>;

	int n_fail = 0;

	void check(bool ok, const char *what)
	{
		if(!ok)
		{
			n_fail++;
			std::printf("FAIL: %s\n", what);
		}
	}

	AtomType build(int Z, double dR, double dR_bl)
	{
		mt::Atomic_Data atomic_data(mt::ePT_Lobato_0_12);
		AtomType at;
		atomic_data.To_atom_type_CPU(Z, mt::c_Vrl, mt::c_nR, dR, at, dR_bl);
		return at;
	}

	double exact_integral(AtomType &at)
	{
		mt::Atom_Cal<double> cal;
		cal.Set_Atom_Type(mt::ePT_Lobato_0_12, 0, &at);
		double fe0;
		cal.feg(0.0, fe0);
		return mt::c_Potf*fe0;
	}

	// Sum of MULTEM's evaluation over the pixels of a grid of spacing dR, the atom at (x0, y0) dR
	// from a pixel centre; pixel centres at i dR (Grid_2d::Rx), times the pixel area.
	double grid_sum(AtomType &at, double dR, double x0, double y0)
	{
		auto &coef = at.coef[0];
		mt::Atom_Vp<double> atom;
		atom.R2 = coef.R2.data();
		atom.c0 = coef.ciVR.c0.data();
		atom.c1 = coef.ciVR.c1.data();
		atom.c2 = coef.ciVR.c2.data();
		atom.c3 = coef.ciVR.c3.data();
		const double R2_max = coef.R2_max();
		const double xa = x0*dR, ya = y0*dR;
		const int n = static_cast<int>(std::ceil(coef.R_max/dR)) + 2;
		double sum = 0;
		for(int ix = -n; ix <= n; ix++)
		{
			for(int iy = -n; iy <= n; iy++)
			{
				const double dx = ix*dR - xa, dy = iy*dR - ya;
				const double R2 = dx*dx + dy*dy;
				if(R2 < R2_max)
				{
					sum += mt::host_device_detail::eval_cubic_poly(R2, atom);
				}
			}
		}
		return sum*dR*dR;
	}

	struct Ratio { double mean, sd, worst; };

	Ratio grid_ratio(AtomType &at, double dR, int n_off, std::mt19937_64 &rng)
	{
		std::uniform_real_distribution<double> u(0.0, 1.0);
		const double ex = exact_integral(at);
		double s = 0, s2 = 0, worst = 0;
		for(int k = 0; k < n_off; k++)
		{
			const double r = grid_sum(at, dR, u(rng), u(rng))/ex;
			s += r;
			s2 += r*r;
			worst = std::fmax(worst, std::fabs(r - 1));
		}
		const double mean = s/n_off;
		const double var = std::fmax(0.0, (s2 - n_off*mean*mean)/(n_off - 1));
		return {mean, std::sqrt(var), worst};
	}

	void test_gauss_legendre()
	{
		std::vector<double> x, w;
		for(int n : {1, 2, 5, 32, 77})
		{
			mt::bandlimit::gauss_legendre(n, 0.3, 2.1, x, w);
			double sw = 0;
			for(auto wi : w) sw += wi;
			check(std::fabs(sw - 1.8) < 1e-13, "Gauss-Legendre weights sum to the interval");
			for(int p = 1; p <= 2*n - 1; p += 2)
			{
				double s = 0;
				for(int i = 0; i < n; i++) s += w[i]*std::pow(x[i], p);
				const double ref = (std::pow(2.1, p + 1) - std::pow(0.3, p + 1))/(p + 1);
				check(std::fabs(s/ref - 1) < 1e-12, "Gauss-Legendre integrates x^(2n-1) exactly");
			}
		}
		std::printf("1. Gauss-Legendre rule: checked n = 1, 2, 5, 32, 77\n");
	}

	void test_reference_integrals()
	{
		// an independent numpy evaluation of the same Lobato coefficients
		const struct { int Z; double ref; } refs[] = {{1, 25.336}, {6, 120.111}, {7, 106.145}, {8, 95.200}};
		std::printf("2. c_Potf f_e(0), V A^3:");
		for(auto r : refs)
		{
			auto at = build(r.Z, 1.0, 1.0);
			const double ex = exact_integral(at);
			std::printf("  Z=%d %.3f", r.Z, ex);
			check(std::fabs(ex - r.ref) < 6e-4, "c_Potf f_e(0) equals the independent reference");
		}
		std::printf("\n");
	}

	void test_table_integral()
	{
		double worst = 0;
		for(int Z : {1, 6, 8, 79})
		{
			for(double dR : {0.05, 0.5, 1.5, 4.0})
			{
				auto at = build(Z, dR, dR);
				auto &c = at.coef[0];
				const double I = mt::bandlimit::table_integral(mt::c_nR, c.R2, c.ciVR.c0, c.ciVR.c1, c.ciVR.c2, c.ciVR.c3);
				worst = std::fmax(worst, std::fabs(I/exact_integral(at) - 1));
			}
		}
		std::printf("3. table integral / c_Potf f_e(0): worst |ratio - 1| = %.1e\n", worst);
		check(worst < 1e-12, "a band-limited table integrates to c_Potf f_e(0)");
	}

	void test_grid_sum()
	{
		std::mt19937_64 rng(20261106);
		std::printf("4. grid sum / c_Potf f_e(0), band-limited, 64 offsets: mean (sd) [worst |r-1|]\n");
		std::printf("   %-6s", "A/px");
		const double pxs[] = {0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0};
		for(double p : pxs) std::printf(" %21.2f", p);
		std::printf("\n");
		for(int Z : {1, 6, 7, 8, 15, 16, 26, 79})
		{
			std::printf("   Z=%-4d", Z);
			for(double dR : pxs)
			{
				auto at = build(Z, dR, dR);
				const Ratio r = grid_ratio(at, dR, 64, rng);
				std::printf(" %.5f(%.0e)[%.0e]", r.mean, r.sd, r.worst);
				check(std::fabs(r.mean - 1) < 2e-4, "mean grid-sum ratio within 2e-4 of 1");
				check(r.worst < 1e-3, "every offset's grid-sum ratio within 1e-3 of 1");
			}
			std::printf("\n");
		}

		double worst_mean = 0, worst_any = 0;
		int worst_Z = 0;
		double worst_dR = 0;
		for(int Z = 1; Z <= 103; Z++)
		{
			for(double dR : {0.1, 0.5, 1.5, 3.0})
			{
				auto at = build(Z, dR, dR);
				const Ratio r = grid_ratio(at, dR, 16, rng);
				if(std::fabs(r.mean - 1) > worst_mean)
				{
					worst_mean = std::fabs(r.mean - 1);
					worst_Z = Z;
					worst_dR = dR;
				}
				worst_any = std::fmax(worst_any, r.worst);
			}
		}
		std::printf("   Z = 1-103 at 0.1, 0.5, 1.5, 3 A/px, 16 offsets: worst |mean - 1| = %.1e (Z=%d, %.1f A/px), worst offset %.1e\n",
			worst_mean, worst_Z, worst_dR, worst_any);
		check(worst_mean < 2e-4, "every element's mean grid-sum ratio within 2e-4 of 1");
		check(worst_any < 1e-3, "every element's grid-sum ratio within 1e-3 of 1 at every offset");
	}

	void test_shape()
	{
		// the table against the band-limited function itself, inside the taper radius (the taper is
		// applied on top), each normalised by its value at R = 0
		double worst = 0;
		for(int Z : {1, 8, 79})
		{
			for(double dR : {0.05, 0.25, 1.5, 4.0})
			{
				auto at = build(Z, dR, dR);
				auto &c = at.coef[0];
				mt::Atom_Cal<double> cal;
				cal.Set_Atom_Type(mt::ePT_Lobato_0_12, 0, &at);
				auto F = [&cal](double q)->double { double fe; cal.feg(q, fe); return mt::c_Potf*fe; };

				const int n = 2001;
				std::vector<double> R(n), V(n), dV(n);
				for(int i = 0; i < n; i++) R[i] = (c.R_tap*0.999)*i/(n - 1);
				mt::bandlimit::projected_potential(F, dR, n, R.data(), V.data(), dV.data());

				mt::Atom_Vp<double> atom;
				atom.R2 = c.R2.data();
				atom.c0 = c.ciVR.c0.data();
				atom.c1 = c.ciVR.c1.data();
				atom.c2 = c.ciVR.c2.data();
				atom.c3 = c.ciVR.c3.data();
				const double t0 = mt::host_device_detail::eval_cubic_poly(0.0, atom);
				for(int i = 0; i < n; i++)
				{
					const double t = mt::host_device_detail::eval_cubic_poly(R[i]*R[i], atom);
					worst = std::fmax(worst, std::fabs(t/t0 - V[i]/V[0]));
				}
			}
		}
		std::printf("5. table vs band-limited function inside R_tap, relative to the peak: worst %.1e\n", worst);
		check(worst < 2e-5, "the table reproduces the band-limited function to 2e-5 of its peak");
	}

	void test_original_tables()
	{
		std::mt19937_64 rng(7);
		auto at = build(8, 1.5, 0.0);
		const Ratio r = grid_ratio(at, 1.5, 64, rng);
		std::printf("6. MULTEM's own table, O at 1.5 A/px: grid sum / c_Potf f_e(0) = %.4f (sd %.1e)\n", r.mean, r.sd);
		check(r.mean < 0.5, "MULTEM's own table still shows the fault (so check 4 can fail)");
		check(std::fabs(at.coef[0].R_min - 1.5) < 1e-12, "MULTEM's own table still starts at the grid spacing");
	}
}

int main()
{
	test_gauss_legendre();
	test_reference_integrals();
	test_table_integral();
	test_grid_sum();
	test_shape();
	test_original_tables();
	if(n_fail == 0)
	{
		std::printf("ALL PASS\n");
		return 0;
	}
	std::printf("%d CHECK(S) FAILED\n", n_fail);
	return 1;
}
