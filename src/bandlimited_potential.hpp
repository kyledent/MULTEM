#ifndef BANDLIMITED_POTENTIAL_H
#define BANDLIMITED_POTENTIAL_H

#include <cmath>
#include <cstdlib>
#include <vector>

// window() is called from device code too (python-multem's random-field amplitude)
#if defined(__CUDACC__)
#define BANDLIMIT_HOST_DEVICE __host__ __device__
#else
#define BANDLIMIT_HOST_DEVICE
#endif

namespace mt
{
	// Band-limited projected-potential tables, selected at runtime by MULTEM_BANDLIMITED_POTENTIAL=1.
	//
	// MULTEM evaluates an atom's projected potential at pixel centres from a radial table: a cubic in
	// R^2 between log-spaced nodes. The table starts at the grid spacing, and pixels closer to the
	// atom than that take its first interval extrapolated inwards. At the sampling used for cryo-EM,
	// almost all of an atom's projected potential lies within one pixel of it, so most of it is
	// missed -- about 85% for oxygen at 1.5 A/px and 99.7% at 3 A/px -- and the loss differs between
	// elements, so a material keeps neither its mean inner potential nor its contrast with another.
	//
	// Selected, the table holds the projected potential low-pass filtered to the grid instead:
	//
	//     V_bl(R) = 2 pi int_0^q2 c_Potf f_e(q) W(q) J0(2 pi q R) q dq,
	//
	// with W = 1 up to half the Nyquist frequency and a raised cosine down to 0 at Nyquist; the
	// Nyquist frequency is the coarser axis's, 1/(2 dR). A function with no content at or beyond 1/dR
	// sums on a grid of spacing dR to its integral, whatever its sub-pixel position (Poisson
	// summation), so each atom's pixel sum is its full c_Potf f_e(0), and nothing above Nyquist
	// aliases. V_bl is finite and smooth at R = 0, so the nodes start well inside a pixel. The table
	// reaches at least c_r_max_px pixels so that the filter's ringing is kept, and it is scaled so that
	// its own integral is exactly c_Potf f_e(0), which absorbs the small loss to the cutoff and the
	// taper. The evaluator is unchanged.
	//
	// Only neutral atoms are filtered; the tables of ions are built as before.
	namespace bandlimit
	{
		constexpr double c_pi = 3.141592653589793238463;

		// Bessel functions of the first kind, orders 0 and 1 (POSIX; MSVC spells them _j0, _j1).
		inline double bessel_j0(double x)
		{
#ifdef _MSC_VER
			return ::_j0(x);
#else
			return ::j0(x);
#endif
		}

		inline double bessel_j1(double x)
		{
#ifdef _MSC_VER
			return ::_j1(x);
#else
			return ::j1(x);
#endif
		}

		constexpr double c_q1_frac = 0.5;	// W = 1 below this fraction of Nyquist...
		constexpr double c_q2_frac = 1.0;	// ...and 0 from this fraction on
		constexpr double c_r_max_px = 8.0;	// the table reaches at least this many pixels
		constexpr double c_r_min_px = 0.2;	// its first node, in pixels

		// Whether MULTEM_BANDLIMITED_POTENTIAL selects the band-limited tables (any non-zero integer).
		inline bool enabled()
		{
			const char *e = std::getenv("MULTEM_BANDLIMITED_POTENTIAL");
			return (e != nullptr) && (std::atoi(e) != 0);
		}

		// The grid spacing whose Nyquist frequency sets the band: the coarser axis.
		inline double band_spacing(double dRx, double dRy)
		{
			return (dRx > dRy) ? dRx : dRy;
		}

		// W(q) for a grid of spacing dR; q in 1/A (cycles per Angstrom).
		BANDLIMIT_HOST_DEVICE inline double window(double q, double dR)
		{
			const double q_nyq = 0.5/dR;
			const double q1 = c_q1_frac*q_nyq;
			const double q2 = c_q2_frac*q_nyq;
			if(q <= q1) return 1.0;
			if(q >= q2) return 0.0;
			return 0.5*(1.0 + ::cos(c_pi*(q - q1)/(q2 - q1)));
		}

		// Gauss-Legendre nodes and weights for n points on [a, b] (Newton iteration on the Legendre
		// polynomial from the usual cosine estimate; converges to machine precision in a few steps).
		inline void gauss_legendre(int n, double a, double b, std::vector<double> &x, std::vector<double> &w)
		{
			x.resize(n);
			w.resize(n);
			const double xm = 0.5*(b + a), xl = 0.5*(b - a);
			for(int i = 0; i < (n + 1)/2; i++)
			{
				double z = std::cos(c_pi*(i + 0.75)/(n + 0.5));
				double dp = 1.0;
				for(int it = 0; it < 100; it++)
				{
					double p0 = 1.0, p1 = 0.0;
					for(int j = 0; j < n; j++)
					{
						const double p2 = p1;
						p1 = p0;
						p0 = ((2.0*j + 1.0)*z*p1 - j*p2)/(j + 1.0);
					}
					// p0 = P_n(z), p1 = P_{n-1}(z)
					dp = n*(z*p0 - p1)/(z*z - 1.0);
					const double dz = p0/dp;
					z -= dz;
					if(std::fabs(dz) < 1e-15) break;
				}
				x[i] = xm - xl*z;
				x[n - 1 - i] = xm + xl*z;
				w[i] = w[n - 1 - i] = 2.0*xl/((1.0 - z*z)*dp*dp);
			}
		}

		// V_bl and dV_bl/dR at the nR radii R, for an atom whose projected potential has the 2D Fourier
		// transform F(q) (c_Potf f_e(q) for MULTEM's tables), on a grid of spacing dR.
		//
		// The integrand is smooth on each side of q1, where W has a kink, so each side gets its own
		// Gauss-Legendre rule. J0(2 pi q R) oscillates about q R times over the band at the outermost
		// node, and the point count grows with that; 32 + 2 q2 R_max points per side reproduce a
		// 2000-point rule to round-off for O, H and S from 0.02 to 3 A/px.
		template <class TF>
		void projected_potential(TF &&F, double dR, int nR, const double *R, double *V, double *dV)
		{
			const double q_nyq = 0.5/dR;
			const double q1 = c_q1_frac*q_nyq;
			const double q2 = c_q2_frac*q_nyq;
			const double R_max = R[nR - 1];
			int n = 32 + 2*static_cast<int>(std::ceil(q2*R_max));
			if(n > 4096) n = 4096;

			for(int iR = 0; iR < nR; iR++)
			{
				V[iR] = 0;
				dV[iR] = 0;
			}

			std::vector<double> xq, wq;
			const double edges[3] = {0.0, q1, q2};
			for(int s = 0; s < 2; s++)
			{
				gauss_legendre(n, edges[s], edges[s + 1], xq, wq);
				for(int k = 0; k < n; k++)
				{
					const double q = xq[k];
					// 2 pi F(q) W(q) q dq
					const double c = 2.0*c_pi*F(q)*window(q, dR)*q*wq[k];
					const double kq = 2.0*c_pi*q;
					for(int iR = 0; iR < nR; iR++)
					{
						const double x = kq*R[iR];
						V[iR] += c*bessel_j0(x);
						// d/dR J0(2 pi q R) = -2 pi q J1(2 pi q R)
						dV[iR] -= c*kq*bessel_j1(x);
					}
				}
			}
		}

		// The integral over the plane of a table as MULTEM evaluates it: per interval a cubic in
		// x = R^2 - R2[i], so 2 pi int V R dR = pi int V d(R^2). Below the first node the evaluator
		// extrapolates interval 0 to x = -R2[0]; beyond the last node it returns 0.
		template <class TR2, class TC>
		double table_integral(int nR, const TR2 &R2, const TC &c0, const TC &c1, const TC &c2, const TC &c3)
		{
			auto cubic = [&](int i, double a, double b)->double
			{
				auto F = [&](double x)->double
				{
					return x*(c0[i] + x*(c1[i]/2.0 + x*(c2[i]/3.0 + x*c3[i]/4.0)));
				};
				return F(b) - F(a);
			};

			double sum = cubic(0, -static_cast<double>(R2[0]), 0.0);
			for(int i = 0; i < nR - 1; i++)
			{
				sum += cubic(i, 0.0, static_cast<double>(R2[i + 1] - R2[i]));
			}
			return c_pi*sum;
		}
	}
}

#endif
