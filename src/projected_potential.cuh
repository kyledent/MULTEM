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

#ifndef PROJECTED_POTENTIAL_H
#define PROJECTED_POTENTIAL_H

#include "math.cuh"
#include "types.cuh"
#include "traits.cuh"
#include "stream.cuh"
#include "quadrature.hpp"
#include "input_multislice.cuh"
#include "output_multislice.hpp"
#include "cpu_fcns.hpp"
#include "gpu_fcns.cuh"
#include "cgpu_fcns.cuh"
#include "spec.hpp"

namespace mt
{
	template <class T, eDevice dev>
	class Projected_Potential: public Spec<T>{
		public:
			using size_type = std::size_t;

			static const eDevice device = dev;

			Projected_Potential(): stream(nullptr), n_atoms_s(2048),
				atom_Vp_pinned{nullptr, nullptr},
				atom_Vp_d{nullptr, nullptr},
				atom_Vp_capacity(0),
				atom_Vp_all_d(nullptr),
				atom_Vp_all_count(0),
				gpu_atoms_ready(false),
				d_tile_counts(nullptr),
				d_tile_offsets(nullptr),
				d_tile_write_pos(nullptr),
				d_atom_tile_indices(nullptr),
				d_cub_temp(nullptr),
				d_cub_temp_size(0),
				tile_indices_capacity(0),
				n_tiles(0), n_tiles_x(0), n_tiles_y(0)
			{
				async_stream[0] = nullptr;
				async_stream[1] = nullptr;
			}

			~Projected_Potential()
			{
#ifndef MULTEM_NO_CUDA
				free_tile_buffers();
				free_async_buffers();
#endif
			}

			void set_input_data(Input_Multislice<T> *input_multislice_i, Stream<dev> *stream_i)
			{
				Spec<T>::set_input_data(input_multislice_i);
				stream = stream_i;

				Quadrature quadrature;
				quadrature(0, c_nqz, qz); // 0: int_-1^1 y(x) dx - TanhSinh quadrature

				atom_type.resize(c_nAtomsTypes);
				for(auto iatom_type = 0; iatom_type<atom_type.size(); iatom_type++)
				{
					atom_type[iatom_type].assign(Spec<T>::atom_type[iatom_type]);
				}

				n_atoms_s = (device==e_host)?(stream->size()):2048;

				int nv = max(this->input_multislice->grid_2d.nx_dRx(this->atoms.l_x_int), this->input_multislice->grid_2d.ny_dRy(this->atoms.l_y_int));

				stream_data.resize(n_atoms_s);

				for(auto i = 0; i<stream_data.size(); i++)
				{
					if(device==e_host)
					{
						stream_data.iv[i].resize(nv);
						stream_data.v[i].resize(nv);
					}

					if(this->input_multislice->is_subslicing())
					{
						stream_data.c0[i].resize(c_nR);
						stream_data.c1[i].resize(c_nR);
						stream_data.c2[i].resize(c_nR);
						stream_data.c3[i].resize(c_nR);
					}
				}

				atom_Vp_h.resize(n_atoms_s);
				atom_Vp.resize(n_atoms_s);

	#ifndef MULTEM_NO_CUDA
				// Allocate pinned host + device double buffers and async streams
				// for the GPU path (double-buffered async atom streaming)
				if(device == e_device)
				{
					alloc_async_buffers(n_atoms_s);
					prepare_gpu_atoms();
					alloc_tile_buffers();  // tile binning data structures
				}
#endif

				V_0.resize(this->input_multislice->grid_2d.nxy());
				V_1.resize(this->input_multislice->grid_2d.nxy());
			}

			/************************Host************************/
			template<eDevice devn = dev>
			enable_if_dev_host<devn, void>
			operator()(const T &z_0, const T &z_e, const int &iatom_0, const int &iatom_e, Vector<T, dev> &V)
			{
				auto eval_cubic_poly = [](Stream<e_host> &stream, Grid_2d<T> &grid_2d, 
				Vector<Atom_Vp<T>, e_host> &atom, Vector<T, dev> &M_o)
				{
					if(stream.n_act_stream<= 0)
					{
						return;
					}

					for(auto istream = 0; istream < stream.n_act_stream-1; istream++)
					{
						stream[istream] = std::thread(std::bind(host_detail::eval_cubic_poly<T>, std::ref(stream), std::ref(grid_2d), std::ref(atom[istream]), std::ref(M_o)));
					}

					host_detail::eval_cubic_poly<T>(stream, grid_2d, atom[stream.n_act_stream-1], M_o);

					stream.synchronize();
				};

				mt::fill(*stream, V, T(0));

				int iatoms = iatom_0;
				while (iatoms <= iatom_e)
				{
					stream->set_n_act_stream(iatom_e-iatoms+1);
					set_atom_Vp(z_0, z_e, iatoms, stream->n_act_stream, atom_Vp);
					//get_cubic_poly_coef_Vz(*stream, atom_Vp);
					eval_cubic_poly(*stream, this->input_multislice->grid_2d, atom_Vp, V);
					iatoms += stream->n_act_stream;
				}

				stream->synchronize();
			}

			/***********************Device***********************/
		#ifdef __CUDACC__
			template<eDevice devn = dev>
			enable_if_dev_device<devn, void>
			operator()(const T &z_0, const T &z_e, const int &iatom_0, const int &iatom_e, Vector<T, dev> &V)
			{
				device_compute(z_0, z_e, iatom_0, iatom_e, V, true);
			}

			// Start computing projected potential asynchronously into V.
			// Runs entirely on async_stream[0/1], does NOT block on class stream.
			// Caller must call finish_compute_async() before reading V.
			void start_compute_async(const int &islice, Vector<T, dev> &V)
			{
				if((islice<0) || (islice>=this->slicing.slice.size()))
				{
					cudaMemsetAsync(raw_pointer_cast(V.data()), 0,
						V.size() * sizeof(T), async_stream[0]);
					return;
				}
				auto &s = this->slicing.slice[islice];
				device_compute(s.z_0, s.z_e, s.iatom_0, s.iatom_e, V, false);
			}

			// Wait for async potential computation to complete.
			void finish_compute_async()
			{
				cudaStreamSynchronize(async_stream[0]);
				cudaStreamSynchronize(async_stream[1]);
			}

			// Pre-compute all Atom_Vp structs and upload to device memory.
			// Eliminates per-slice CPU packing and H->D transfers — each slice
			// becomes batched kernel launches with pointer offsets.
			// Uses double-buffered chunked upload via the existing pinned
			// buffers: CPU pre-computes chunk N+1 while DMA transfers chunk N.
			// Call after set_input_data() or after move_atoms() for frozen phonon.
			void prepare_gpu_atoms()
			{
				gpu_atoms_ready = false;

				if (this->input_multislice->is_subslicing())
					return;  // Subslicing requires per-slice z-dependent fields

				int total_atoms = static_cast<int>(this->atoms.Z.size());
				if (total_atoms <= 0 || !atom_Vp_pinned[0] || !atom_Vp_pinned[1])
					return;

				// Allocate device memory for all atoms
				free_gpu_atoms();
				cudaError_t err = cudaMalloc(&atom_Vp_all_d, total_atoms * sizeof(Atom_Vp<T>));
				if (err != cudaSuccess)
				{
					cudaGetLastError();  // Clear error state
					return;  // Falls back to the double-buffered path
				}

				// Double-buffered chunked upload: pre-compute into pinned buffer[cur],
				// async DMA to device, swap to buffer[nxt] for next chunk.
				// Overlaps CPU pre-computation with PCIe DMA transfer.
				int chunk_size = atom_Vp_capacity;  // 2048
				int cur = 0;

				for (int chunk_start = 0; chunk_start < total_atoms; chunk_start += chunk_size)
				{
					int n = min(chunk_size, total_atoms - chunk_start);
					Atom_Vp<T> *dst = atom_Vp_pinned[cur];

					// Wait for any previous transfer on this buffer
					cudaStreamSynchronize(async_stream[cur]);

					// Pre-compute Atom_Vp for this chunk
					for (int i = 0; i < n; i++)
					{
						int ai = chunk_start + i;
						auto iZ = (this->atoms.Z[ai] % 1000) - 1;
						auto charge = this->atoms.charge[ai];
						int icharge = atom_type[iZ].charge_to_idx(charge);
						auto &coef = atom_type[iZ].coef[icharge];

						dst[i].charge = charge;
						dst[i].x = this->atoms.x[ai];
						dst[i].y = this->atoms.y[ai];
						dst[i].occ = this->atoms.occ[ai];
						dst[i].R2_min = coef.R2_min();
						dst[i].R2_max = coef.R2_max();
						dst[i].R2 = raw_pointer_cast(coef.R2.data());
						dst[i].set_ix0_ixn(this->input_multislice->grid_2d, coef.R_max);
						dst[i].set_iy0_iyn(this->input_multislice->grid_2d, coef.R_max);
						dst[i].R2_tap = coef.R2_tap();
						dst[i].tap_cf = coef.tap_cf;
						dst[i].c0 = raw_pointer_cast(coef.ciVR.c0.data());
						dst[i].c1 = raw_pointer_cast(coef.ciVR.c1.data());
						dst[i].c2 = raw_pointer_cast(coef.ciVR.c2.data());
						dst[i].c3 = raw_pointer_cast(coef.ciVR.c3.data());
					}

					// Async DMA transfer to device array
					cudaMemcpyAsync(atom_Vp_all_d + chunk_start, dst,
						n * sizeof(Atom_Vp<T>), cudaMemcpyHostToDevice, async_stream[cur]);

					cur = 1 - cur;  // Swap to other buffer
				}

				// Drain: wait for both streams to complete
				cudaStreamSynchronize(async_stream[0]);
				cudaStreamSynchronize(async_stream[1]);

				atom_Vp_all_count = total_atoms;
				gpu_atoms_ready = true;
			}

			// Invalidate GPU-resident atom data.
			// Must be called after move_atoms() for frozen phonon simulations.
			void invalidate_gpu_atoms()
			{
				gpu_atoms_ready = false;
			}

			// Tiled potential accumulation path.
			// Bins atoms to output-space tiles via count/prefix-sum/scatter,
			// then launches one block per tile for shared-memory accumulation.
			// Eliminates atomicAdd contention on M_o for dense slices.
			void device_compute_tiled(int iatom_0, int n_atoms,
				Vector<T, dev> &V, bool sync_final)
			{
				using namespace device_detail;
				auto &grid_2d = this->input_multislice->grid_2d;
				const Atom_Vp<T> *atoms_ptr = atom_Vp_all_d + iatom_0;

				// Ensure atom_tile_indices is large enough (9 tiles per atom worst case)
				int max_overlaps = 9 * n_atoms;
				if (max_overlaps > tile_indices_capacity)
				{
					if (d_atom_tile_indices) cudaFree(d_atom_tile_indices);
					cudaMalloc(&d_atom_tile_indices, max_overlaps * sizeof(int));
					tile_indices_capacity = max_overlaps;
				}

				// Step 1: Zero tile counts and count atom-tile overlaps
				// All ops on async_stream[0] — CUDA stream ordering guarantees
				// sequential execution without explicit syncs between steps.
				cudaMemsetAsync(d_tile_counts, 0, n_tiles * sizeof(int), async_stream[0]);

				int threads_1d = 256;
				int blocks_1d = (n_atoms + threads_1d - 1) / threads_1d;
				tile_count_atoms<T><<<blocks_1d, threads_1d, 0, async_stream[0]>>>(
					grid_2d, atoms_ptr, n_atoms,
					n_tiles_x, n_tiles_y, d_tile_counts);

				// Step 2: Inclusive prefix sum → tile_offsets[1..n_tiles]
				// tile_offsets[0] = 0
				cudaMemsetAsync(d_tile_offsets, 0, sizeof(int), async_stream[0]);

				cub::DeviceScan::InclusiveSum(
					d_cub_temp, d_cub_temp_size,
					d_tile_counts, d_tile_offsets + 1, n_tiles,
					async_stream[0]);

				// Step 3: Zero write positions, scatter atom indices
				cudaMemsetAsync(d_tile_write_pos, 0, n_tiles * sizeof(int), async_stream[0]);

				tile_scatter_atoms<T><<<blocks_1d, threads_1d, 0, async_stream[0]>>>(
					grid_2d, atoms_ptr, n_atoms,
					n_tiles_x, n_tiles_y,
					d_tile_offsets, d_tile_write_pos, d_atom_tile_indices);

				// Step 4: Tiled accumulation — one block per tile
				rVector<T> V_rv;
				V_rv.m_size = V.size();
				V_rv.V = raw_pointer_cast(V.data());

				dim3 tile_thr(c_tile_h, c_tile_w, 1);  // x→iy (coalesced), y→ix
				eval_cubic_poly_tiled<T><<<n_tiles, tile_thr, 0, async_stream[0]>>>(
					grid_2d, atoms_ptr,
					n_tiles_y, d_tile_offsets,
					d_atom_tile_indices, V_rv);

				if (sync_final)
				{
					cudaStreamSynchronize(async_stream[0]);
				}
			}
		#endif

			void operator()(const int &islice_0, const int &islice_e, Vector<T, dev> &V)
			{
				if((islice_0<0) || (islice_e>=this->slicing.slice.size()))
				{
					mt::fill(*stream, V, 0.0);
					return;
				}
				this->operator()(this->slicing.slice[islice_0].z_0, this->slicing.slice[islice_e].z_e, 
				this->slicing.slice[islice_0].iatom_0, this->slicing.slice[islice_e].iatom_e, V);
			}

			void operator()(const int &islice_0, const int &islice_e)
			{
				this->operator()(islice_0, islice_e, V_0);
			}

			void operator()(const int &islice, Vector<T, dev> &V)
			{
				this->operator()(islice, islice, V);
			}

			void operator()(const int &islice)
			{
				this->operator()(islice, islice, V_0);
			}

			template <class TOutput_multislice>
			void operator()(const int &islice, TOutput_multislice &output_multislice)
			{
				this->operator()(islice, islice, V_0);
				mt::copy_to_host(output_multislice.stream, V_0, output_multislice.V[0]);
			}

			Vector<T, dev> V_0;
			Vector<T, dev> V_1;	// second potential buffer for pipelined slice overlap
			Stream<dev> *stream;
		private:
			int n_atoms_s;

			struct Stream_Data
			{
				using value_type = T;
				using size_type = std::size_t;

				static const eDevice device = e_host;

				size_type size() const
				{
					return iv.size();
				}

				void resize(const size_type &new_size)
				{
					iv.resize(new_size);
					v.resize(new_size);
					c0.resize(new_size);
					c1.resize(new_size);
					c2.resize(new_size);
					c3.resize(new_size);
				}

				Vector<Vector<int, dev>, e_host> iv;
				Vector<Vector<T, dev>, e_host> v;
				Vector<Vector<T, dev>, e_host> c0; 		// zero coefficient
				Vector<Vector<T, dev>, e_host> c1; 		// first coefficient
				Vector<Vector<T, dev>, e_host> c2; 		// second coefficient
				Vector<Vector<T, dev>, e_host> c3; 		// third coefficient
			};

			void set_atom_Vp(const T &z_0, const T &z_e, int iatoms, int n_atoms, Vector<Atom_Vp<T>, dev> &atom_Vp)
			{
				for(auto istream = 0; istream < n_atoms; istream++)
				{
					auto iZ = (this->atoms.Z[iatoms] % 1000)-1;
					auto charge = this->atoms.charge[iatoms];
					int icharge = atom_type[iZ].charge_to_idx(charge);
					auto &coef = atom_type[iZ].coef[icharge];

					atom_Vp_h[istream].charge = charge;
					atom_Vp_h[istream].x = this->atoms.x[iatoms];
					atom_Vp_h[istream].y = this->atoms.y[iatoms];
					atom_Vp_h[istream].occ = this->atoms.occ[iatoms];
					atom_Vp_h[istream].R2_min = coef.R2_min();
					atom_Vp_h[istream].R2_max = coef.R2_max();
					atom_Vp_h[istream].R2 = raw_pointer_cast(coef.R2.data());
					atom_Vp_h[istream].set_ix0_ixn(this->input_multislice->grid_2d, coef.R_max);
					atom_Vp_h[istream].set_iy0_iyn(this->input_multislice->grid_2d, coef.R_max);
					atom_Vp_h[istream].R2_tap = coef.R2_tap();
					atom_Vp_h[istream].tap_cf = coef.tap_cf;

					if(device==e_host)
					{
						atom_Vp_h[istream].iv = raw_pointer_cast(stream_data.iv[istream].data());
						atom_Vp_h[istream].v = raw_pointer_cast(stream_data.v[istream].data());
					}

					if(this->input_multislice->is_subslicing())
					{
						atom_Vp_h[istream].z0h = 0.5*(z_0 - this->atoms.z[iatoms]); 
						atom_Vp_h[istream].zeh = 0.5*(z_e - this->atoms.z[iatoms]);
						atom_Vp_h[istream].split = (atom_Vp_h[istream].z0h<0) && (0<atom_Vp_h[istream].zeh);
						atom_Vp_h[istream].cl = raw_pointer_cast(coef.Vr.cl.data());
						atom_Vp_h[istream].cnl = raw_pointer_cast(coef.Vr.cnl.data());
						atom_Vp_h[istream].c0 = raw_pointer_cast(stream_data.c0[istream].data());
						atom_Vp_h[istream].c1 = raw_pointer_cast(stream_data.c1[istream].data());
						atom_Vp_h[istream].c2 = raw_pointer_cast(stream_data.c2[istream].data());
						atom_Vp_h[istream].c3 = raw_pointer_cast(stream_data.c3[istream].data());
					}
					else
					{
						atom_Vp_h[istream].c0 = raw_pointer_cast(coef.ciVR.c0.data());
						atom_Vp_h[istream].c1 = raw_pointer_cast(coef.ciVR.c1.data());
						atom_Vp_h[istream].c2 = raw_pointer_cast(coef.ciVR.c2.data());
						atom_Vp_h[istream].c3 = raw_pointer_cast(coef.ciVR.c3.data());
					}
					iatoms++;
				}
				thrust::copy(atom_Vp_h.begin(), atom_Vp_h.begin()+n_atoms, atom_Vp.begin());
			}
			
			void get_cubic_poly_coef_Vz(Stream<dev> &stream, Vector<Atom_Vp<T>, e_host> &atom_Vp)
			{
				if(this->input_multislice->is_subslicing())
				{
					mt::linear_Vz(stream, this->input_multislice->potential_type, qz, atom_Vp);
					mt::cubic_poly_coef(stream, atom_Vp);
				}
			}

		#ifdef __CUDACC__
			// Core device compute with double-buffered async atom streaming.
			// Zeros V on async_stream[0] to avoid race with class stream.
			// If sync_final=true (normal path), blocks until all work completes.
			// If sync_final=false (async path), returns without final sync.
			void device_compute(const T &z_0, const T &z_e, const int &iatom_0,
				const int &iatom_e, Vector<T, dev> &V, bool sync_final)
			{
				// Zero V on async_stream[0] for correct stream ordering
				cudaMemsetAsync(raw_pointer_cast(V.data()), 0,
					V.size() * sizeof(T), async_stream[0]);

				if (iatom_0 > iatom_e)
				{
					if (sync_final)
					{
						cudaStreamSynchronize(async_stream[0]);
					}
					return;
				}

				// GPU-resident path — batched kernel launches using pointer
				// offsets into pre-uploaded atom array. Eliminates per-batch CPU
				// packing and H->D transfer while maintaining batch-size scheduling.
				if (gpu_atoms_ready)
				{
					int total_slice_atoms = iatom_e - iatom_0 + 1;

					// Use tiled path for slices with enough atoms to benefit
					// from shared-memory accumulation (reduces atomicAdd contention).
					// Threshold: tiles overhead is ~3 lightweight kernels + prefix sum;
					// worthwhile when atom density creates significant contention.
					if (total_slice_atoms >= 256 && n_tiles > 0 && d_tile_counts)
					{
						device_compute_tiled(iatom_0, total_slice_atoms, V, sync_final);
						return;
					}

					// Fallback: batched atom-centric kernel
					int iatoms = iatom_0;
					while (iatoms <= iatom_e)
					{
						int n_atoms = min(n_atoms_s, iatom_e - iatoms + 1);

						Grid_BT grid_bt;
						grid_bt.Blk = dim3(n_atoms, 1, 1);
						grid_bt.Thr = dim3(c_thrn_x, c_thrn_y, 1);

						rVector<Atom_Vp<T>> atoms_rv;
						atoms_rv.m_size = n_atoms;
						atoms_rv.V = atom_Vp_all_d + iatoms;

						device_detail::eval_cubic_poly<T><<<grid_bt.Blk, grid_bt.Thr, 0, async_stream[0]>>>(
							this->input_multislice->grid_2d, atoms_rv, V);

						iatoms += n_atoms;
					}

					if (sync_final)
					{
						cudaStreamSynchronize(async_stream[0]);
					}
					return;
				}

				// Double-buffered async atom streaming (fallback when GPU-resident not available)
				int iatoms = iatom_0;
				int cur = 0;

				// Prime first batch: pack, async copy, wait
				int n_cur = min(n_atoms_s, iatom_e - iatoms + 1);
				set_atom_Vp_pinned(z_0, z_e, iatoms, n_cur, cur);
				cudaMemcpyAsync(atom_Vp_d[cur], atom_Vp_pinned[cur],
					n_cur * sizeof(Atom_Vp<T>),
					cudaMemcpyHostToDevice, async_stream[cur]);
				cudaStreamSynchronize(async_stream[cur]);

				while (iatoms <= iatom_e)
				{
					int n_atoms = min(n_atoms_s, iatom_e - iatoms + 1);
					int nxt = 1 - cur;

					Grid_BT grid_bt;
					grid_bt.Blk = dim3(n_atoms, 1, 1);
					grid_bt.Thr = dim3(c_thrn_x, c_thrn_y, 1);

					rVector<Atom_Vp<T>> atoms_rv;
					atoms_rv.m_size = n_atoms;
					atoms_rv.V = atom_Vp_d[cur];

					device_detail::eval_cubic_poly<T><<<grid_bt.Blk, grid_bt.Thr, 0, async_stream[cur]>>>(
						this->input_multislice->grid_2d, atoms_rv, V);

					iatoms += n_atoms;

					int n_next = min(n_atoms_s, iatom_e - iatoms + 1);
					if (n_next > 0)
					{
						set_atom_Vp_pinned(z_0, z_e, iatoms, n_next, nxt);
						cudaMemcpyAsync(atom_Vp_d[nxt], atom_Vp_pinned[nxt],
							n_next * sizeof(Atom_Vp<T>),
							cudaMemcpyHostToDevice, async_stream[nxt]);
					}

					cudaStreamSynchronize(async_stream[cur]);
					cur = nxt;
				}

				if (sync_final)
				{
					cudaStreamSynchronize(async_stream[0]);
					cudaStreamSynchronize(async_stream[1]);
				}
			}

			// Pack atoms into pinned host buffer[buf_idx] for async H->D transfer.
			// Same logic as set_atom_Vp() but writes to atom_Vp_pinned[buf_idx]
			// instead of atom_Vp_h + thrust::copy.
			void set_atom_Vp_pinned(const T &z_0, const T &z_e, int iatoms, int n_atoms, int buf_idx)
			{
				Atom_Vp<T> *dst = atom_Vp_pinned[buf_idx];
				for(int i = 0; i < n_atoms; i++)
				{
					auto iZ = (this->atoms.Z[iatoms] % 1000)-1;
					auto charge = this->atoms.charge[iatoms];
					int icharge = atom_type[iZ].charge_to_idx(charge);
					auto &coef = atom_type[iZ].coef[icharge];

					dst[i].charge = charge;
					dst[i].x = this->atoms.x[iatoms];
					dst[i].y = this->atoms.y[iatoms];
					dst[i].occ = this->atoms.occ[iatoms];
					dst[i].R2_min = coef.R2_min();
					dst[i].R2_max = coef.R2_max();
					dst[i].R2 = raw_pointer_cast(coef.R2.data());
					dst[i].set_ix0_ixn(this->input_multislice->grid_2d, coef.R_max);
					dst[i].set_iy0_iyn(this->input_multislice->grid_2d, coef.R_max);
					dst[i].R2_tap = coef.R2_tap();
					dst[i].tap_cf = coef.tap_cf;

					if(this->input_multislice->is_subslicing())
					{
						dst[i].z0h = 0.5*(z_0 - this->atoms.z[iatoms]);
						dst[i].zeh = 0.5*(z_e - this->atoms.z[iatoms]);
						dst[i].split = (dst[i].z0h<0) && (0<dst[i].zeh);
						dst[i].cl = raw_pointer_cast(coef.Vr.cl.data());
						dst[i].cnl = raw_pointer_cast(coef.Vr.cnl.data());
						dst[i].c0 = raw_pointer_cast(stream_data.c0[i].data());
						dst[i].c1 = raw_pointer_cast(stream_data.c1[i].data());
						dst[i].c2 = raw_pointer_cast(stream_data.c2[i].data());
						dst[i].c3 = raw_pointer_cast(stream_data.c3[i].data());
					}
					else
					{
						dst[i].c0 = raw_pointer_cast(coef.ciVR.c0.data());
						dst[i].c1 = raw_pointer_cast(coef.ciVR.c1.data());
						dst[i].c2 = raw_pointer_cast(coef.ciVR.c2.data());
						dst[i].c3 = raw_pointer_cast(coef.ciVR.c3.data());
					}
					iatoms++;
				}
			}

			// Allocate double-buffered pinned host + device memory and async streams.
			void alloc_async_buffers(int capacity)
			{
				free_async_buffers();
				atom_Vp_capacity = capacity;
				for(int i = 0; i < 2; i++)
				{
					cudaMallocHost(&atom_Vp_pinned[i], capacity * sizeof(Atom_Vp<T>));
					cudaMalloc(&atom_Vp_d[i], capacity * sizeof(Atom_Vp<T>));
					cudaStreamCreate(&async_stream[i]);
				}
			}

			// Free double-buffered resources.
			void free_async_buffers()
			{
				free_gpu_atoms();
				for(int i = 0; i < 2; i++)
				{
					if(atom_Vp_pinned[i])
					{
						cudaFreeHost(atom_Vp_pinned[i]);
						atom_Vp_pinned[i] = nullptr;
					}
					if(atom_Vp_d[i])
					{
						cudaFree(atom_Vp_d[i]);
						atom_Vp_d[i] = nullptr;
					}
					if(async_stream[i])
					{
						cudaStreamDestroy(async_stream[i]);
						async_stream[i] = nullptr;
					}
				}
				atom_Vp_capacity = 0;
			}

			// Free GPU-resident atom data.
			void free_gpu_atoms()
			{
				if (atom_Vp_all_d)
				{
					cudaFree(atom_Vp_all_d);
					atom_Vp_all_d = nullptr;
				}
				atom_Vp_all_count = 0;
				gpu_atoms_ready = false;
			}

			// 2D Morton code for spatial locality sorting.
			// Interleaves bits of (x, y) grid coordinates to produce a Z-curve index.
			static inline uint32_t morton2d(int x, int y)
			{
				uint32_t xx = static_cast<uint32_t>(x);
				uint32_t yy = static_cast<uint32_t>(y);
				xx = (xx | (xx << 8)) & 0x00FF00FFu;
				xx = (xx | (xx << 4)) & 0x0F0F0F0Fu;
				xx = (xx | (xx << 2)) & 0x33333333u;
				xx = (xx | (xx << 1)) & 0x55555555u;
				yy = (yy | (yy << 8)) & 0x00FF00FFu;
				yy = (yy | (yy << 4)) & 0x0F0F0F0Fu;
				yy = (yy | (yy << 2)) & 0x33333333u;
				yy = (yy | (yy << 1)) & 0x55555555u;
				return xx | (yy << 1);
			}

			// Allocate tile binning data structures for tiled potential.
			void alloc_tile_buffers()
			{
				free_tile_buffers();

				auto &g = this->input_multislice->grid_2d;
				n_tiles_x = (g.nx + device_detail::c_tile_w - 1) / device_detail::c_tile_w;
				n_tiles_y = (g.ny + device_detail::c_tile_h - 1) / device_detail::c_tile_h;
				n_tiles = n_tiles_x * n_tiles_y;

				if (n_tiles <= 0) return;

				cudaMalloc(&d_tile_counts, n_tiles * sizeof(int));
				cudaMalloc(&d_tile_offsets, (n_tiles + 1) * sizeof(int));
				cudaMalloc(&d_tile_write_pos, n_tiles * sizeof(int));

				// Initial allocation for atom_tile_indices (grows in device_compute_tiled)
				int initial_capacity = 9 * n_atoms_s;  // 9 tiles/atom × batch size
				cudaMalloc(&d_atom_tile_indices, initial_capacity * sizeof(int));
				tile_indices_capacity = initial_capacity;

				// Query CUB temp storage requirement
				d_cub_temp_size = 0;
				cub::DeviceScan::InclusiveSum(
					nullptr, d_cub_temp_size,
					d_tile_counts, d_tile_offsets + 1, n_tiles);
				if (d_cub_temp_size > 0)
				{
					cudaMalloc(&d_cub_temp, d_cub_temp_size);
				}
			}

			// Free tile binning resources.
			void free_tile_buffers()
			{
				if (d_tile_counts)    { cudaFree(d_tile_counts);    d_tile_counts = nullptr; }
				if (d_tile_offsets)   { cudaFree(d_tile_offsets);   d_tile_offsets = nullptr; }
				if (d_tile_write_pos) { cudaFree(d_tile_write_pos); d_tile_write_pos = nullptr; }
				if (d_atom_tile_indices) { cudaFree(d_atom_tile_indices); d_atom_tile_indices = nullptr; }
				if (d_cub_temp)       { cudaFree(d_cub_temp);       d_cub_temp = nullptr; }
				d_cub_temp_size = 0;
				tile_indices_capacity = 0;
				n_tiles = n_tiles_x = n_tiles_y = 0;
			}
		#endif

			Q1<T, dev> qz;
			Vector<Atom_Type<T, dev>, e_host> atom_type; // Atom types

			Stream_Data stream_data;
			Vector<Atom_Vp<T>, e_host> atom_Vp_h;
			Vector<Atom_Vp<T>, dev> atom_Vp;

			// Double-buffered async atom streaming resources (GPU path only)
			Atom_Vp<T> *atom_Vp_pinned[2];   // Pinned host buffers
			Atom_Vp<T> *atom_Vp_d[2];        // Device buffers
			cudaStream_t async_stream[2];      // Async copy/kernel streams
			int atom_Vp_capacity;              // Current buffer capacity

			// GPU-resident atom data with spatial locality sorting
			Atom_Vp<T> *atom_Vp_all_d;        // All atom Vp data pre-uploaded to device
			int atom_Vp_all_count;             // Number of atoms in GPU-resident array
			bool gpu_atoms_ready;              // Flag: atom_Vp_all_d is valid and current

			// Tile binning data structures for tiled potential accumulation
			int *d_tile_counts;               // Per-tile atom overlap counts
			int *d_tile_offsets;              // CSR offsets (n_tiles + 1)
			int *d_tile_write_pos;            // Scatter write positions (reusable)
			int *d_atom_tile_indices;         // CSR values: atom indices per tile
			void *d_cub_temp;                 // CUB prefix sum temp storage
			size_t d_cub_temp_size;           // CUB temp storage size
			int tile_indices_capacity;        // Current capacity of atom_tile_indices
			int n_tiles;                      // Total number of output tiles
			int n_tiles_x;                    // Tiles in x direction
			int n_tiles_y;                    // Tiles in y direction
	};

} // namespace mt
#endif
