/*
 * This file is part of MULTEM.
 * Copyright 2020 Ivan Lobato <Ivanlh20@gmail.com>
 *
 * MULTEM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MULTEM is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MULTEM. If not, see <http:// www.gnu.org/licenses/>.
 */

#ifndef SPECIMEN_H
#define SPECIMEN_H

#include "phase_timer.hpp"
#include <climits>
#include <memory>
#include "math.cuh"
#include "types.cuh"
#include "lin_alg_def.cuh"
#include "cgpu_rand.cuh"
#include "atomic_data.hpp"
#include "atomic_data_mt.hpp"
#include "input_multislice.cuh"
#include "cgpu_fcns.cuh"
#include "cpu_fcns.hpp"
#include "slicing.hpp"

namespace mt
{
	template <class T>
	class Spec
	{
		public:
			using T_r = T;
			using size_type = std::size_t;

			Spec(): input_multislice(nullptr), atoms_conf(c_conf_none){}

			void set_input_data(Input_Multislice<T> *input_multislice_i)
			{
				mt::PhaseTimer _t_all("set_input_data TOTAL");
				input_multislice = input_multislice_i;

				/***************************************************************************/
				{
				mt::PhaseTimer _t("atom_type tables (all 103)");
				Atomic_Data atomic_data(input_multislice->potential_type);

				atom_type.resize(c_nAtomsTypes); 
				for(auto i = 0; i<atom_type.size(); i++)
				{
					atomic_data.To_atom_type_CPU(i+1, input_multislice->Vrl, input_multislice->nR, input_multislice->grid_2d.dR_min(), atom_type[i]);
				}
				}

				/***************************************************************************/
				{ mt::PhaseTimer _t("atoms_u.set_atoms"); 
				atoms_u.set_atoms(input_multislice->atoms, input_multislice->grid_2d.pbc_xy, &atom_type); }
				{ mt::PhaseTimer _t("atoms_u.sort_by_z");
				sort_atoms_by_z(atoms_u); }

				/***************************************************************************/
				{ mt::PhaseTimer _t("slicing.set_input_data");
				slicing.set_input_data(input_multislice, &atoms_u, &atoms); }

				/***************************************************************************/
				if((input_multislice->is_phase_object()) || (atoms_u.s_z_int < 2.0*input_multislice->grid_2d.dz) || ((slicing.z_plane.size() == 1) && input_multislice->is_slicing_by_planes()))
				{
					input_multislice->grid_2d.dz = atoms_u.s_z_int;
					input_multislice->interaction_model = eESIM_Phase_Object;
					input_multislice->islice = 0;
					input_multislice->pn_dim.z = false;
					if(input_multislice->is_through_slices())
					{
						input_multislice->thick_type = eTT_Through_Thick;
					}
					input_multislice->slice_storage = input_multislice->slice_storage || !input_multislice->is_whole_spec();
					atoms_u.dz = input_multislice->grid_2d.dz;
				}

				{ mt::PhaseTimer _t("atoms.set_atoms (2nd copy)");
				// This copy takes every atom of atoms_u with pbc filtering off, so its order-independent
				// statistics are the ones just computed and can be copied; mean/std depend on summation
				// order, which the sort changed, and are recomputed. MULTEM_SKIP_REDUNDANT_STATS=1
				// selects it, and both paths emit dump_statistic() so the result is checked, not assumed.
				const char *_skip_stats = std::getenv("MULTEM_SKIP_REDUNDANT_STATS");
				if(_skip_stats && _skip_stats[0] == '1')
				{
					atoms.set_atoms(atoms_u, false, &atom_type, false);
					atoms.copy_statistic_from(atoms_u);
				}
				else
				{
					atoms.set_atoms(atoms_u, false, &atom_type);
				} }
				// This is needed for memory preallocation in Transmission function
				{ mt::PhaseTimer _t("slicing.calculate");
				slicing.calculate(); }

				// atoms now holds the still (undisplaced) configuration: a copy of atoms_u
				atoms_conf = c_conf_still;
			}

			// Sort by z. MULTEM_PAR_SORT=1 sorts on host threads, in the same order as the sequential
			// sort (host_par_sort.hpp); MULTEM_SORT_CHECK=1 runs both and reports any difference.
			void sort_atoms_by_z(Atom_Data<T> &a)
			{
				const char *par = std::getenv("MULTEM_PAR_SORT");
				const char *check = std::getenv("MULTEM_SORT_CHECK");
				if(check && check[0] == '1')
				{
					const int nt = mt::host_sort_threads();
					const auto n_diff = a.sort_by_z_check(nt);
					fprintf(stderr, "  [check] sort_by_z: %zu atoms, %d threads, %zu differ from the sequential sort\n",
						static_cast<size_t>(a.size()), nt, static_cast<size_t>(n_diff));
				}
				else if(par && par[0] == '1')
				{
					a.sort_by_z_parallel(mt::host_sort_threads());
				}
				else
				{
					a.sort_by_z();
				}
			}

			/* Move atoms (cgpu_rand distribution will be included in the future) */
			void move_atoms(const int &fp_iconf)
			{
				// Which configuration this call produces. Without frozen phonons every call yields the
				// still configuration -- a copy of atoms_u -- whatever fp_iconf is; with them, the
				// displacement is seeded by pn_seed and fp_iconf alone, so a repeated fp_iconf yields the
				// same atoms. Only Spec writes these arrays, so when atoms already holds the requested
				// configuration, rebuilding it (copy, sort, statistics) reproduces what is there.
				// MULTEM_REUSE_CONF=1 skips the rebuild then; MULTEM_REUSE_CHECK=1 rebuilds anyway and
				// counts every difference from what would have been kept.
				const int conf = input_multislice->is_frozen_phonon() ? fp_iconf : int(c_conf_still);
				const bool held = (conf == atoms_conf);
				const char *reuse = std::getenv("MULTEM_REUSE_CONF");
				const char *check = std::getenv("MULTEM_REUSE_CHECK");
				const bool checking = held && check && check[0] == '1';
				if(held && !checking && reuse && reuse[0] == '1')
				{
					slicing.calculate();
					return;
				}

				// copy-CONSTRUCTED on purpose: Atom_Data's templated operator= re-derives the atoms
				// through set_atoms, so an assignment would not be a copy
				std::unique_ptr<Atom_Data<T>> kept(checking ? new Atom_Data<T>(atoms) : nullptr);
				{ mt::PhaseTimer _t("move_atoms (rebuild)");

				// set phonon configuration
				if(input_multislice->is_frozen_phonon())
				{
					rand.seed(input_multislice->pn_seed, fp_iconf);
					rand.set_activation(input_multislice->pn_dim.x, input_multislice->pn_dim.y, input_multislice->pn_dim.z);
				}

				// move atoms
				for(int iatoms = 0; iatoms<atoms_u.size(); iatoms++)
				{
					atoms.Z[iatoms] = atoms_u.Z[iatoms];
					auto r = atoms_u.to_r3d(iatoms);

					if(input_multislice->is_frozen_phonon())
					{
						auto sigma_x = atoms_u.sigma[iatoms];
						auto sigma_y = atoms_u.sigma[iatoms];
						auto sigma_z = atoms_u.sigma[iatoms];
						r += rand(sigma_x, sigma_y, sigma_z);
					}

					atoms.x[iatoms] = r.x;
					atoms.y[iatoms] = r.y;
					atoms.z[iatoms] = r.z;
					atoms.sigma[iatoms] = atoms_u.sigma[iatoms];
					atoms.occ[iatoms] = atoms_u.occ[iatoms];
					atoms.region[iatoms] = atoms_u.region[iatoms];
					atoms.charge[iatoms] = atoms_u.charge[iatoms];
				}

				if(input_multislice->pn_dim.z)
				{
					sort_atoms_by_z(atoms);
				}

				// get atom information
				atoms.get_statistic(&atom_type);

				}

				// slicing procedure
				slicing.calculate();
				atoms_conf = conf;

				if(kept)
				{
					const auto n_diff = atoms.count_differences(*kept);
					fprintf(stderr, "  [check] move_atoms: %zu atoms, %zu differ from the configuration already held\n",
						static_cast<size_t>(atoms.size()), static_cast<size_t>(n_diff));
				}
			}

			T dz(const int &islice)
			{
				return slicing.dz(islice)/cos(input_multislice->theta);
			}

			T dz_m(const int &islice_0, const int &islice_e)
			{
				return slicing.dz_m(islice_0, islice_e)/cos(input_multislice->theta);
			}

			Input_Multislice<T> *input_multislice; 			

			Atom_Data<T> atoms; 								// displaced atoms
			Slicing<T> slicing; 								// slicing procedure
			Vector<Atom_Type<T, e_host>, e_host> atom_type;		// Atom types
		private:
			Randn_3d<T, e_host> rand;
			Atom_Data<T> atoms_u;

			// which configuration atoms holds (see move_atoms); enumerators, so no definition is needed
			enum : int { c_conf_none = INT_MIN, c_conf_still = INT_MIN + 1 };
			int atoms_conf;
	};

} // namespace mt

#endif
