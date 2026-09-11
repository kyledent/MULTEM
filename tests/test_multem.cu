/*
 * test_multem.cu -- Standalone MULTEM test harness
 *
 * Runs multislice simulations at configurable grid sizes and slice counts,
 * saves output arrays for correctness validation, and reports timing.
 *
 * Uses the python-multem wrapper API (multem::simulate) which provides a
 * clean C++ interface over the raw MULTEM template machinery.
 *
 * Usage:
 *   test_multem [OPTIONS]
 *
 * Options:
 *   --test T1|T2|...|T8|all    Test case to run (default: T1)
 *   --device host|device        CPU or GPU (default: device)
 *   --precision float|double    Floating point precision (default: float)
 *   --output DIR                Directory for output files (default: .)
 *   --repeats N                 Number of timed repeats (default: 1)
 *   --gpu-id N                  GPU device index (default: 0)
 *
 * Build:
 *   See CMakeLists.txt in multem/tests/
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef MULTEM_USE_OPENMP
#include <omp.h>
#endif

#include <multem/multem_ext.h>

// -------------------------------------------------------------------------
// Silicon [110] specimen generation
// -------------------------------------------------------------------------

// Si diamond cubic: a = 5.4307 A
// [110] zone axis: viewed along x=[1,1,0], y=[0,0,1]
// Unit cell in [110] projection:
//   lx = a * sqrt(2) = 7.6785 A (along [110])
//   ly = a = 5.4307 A (along [001])
//   lz = a / sqrt(2) = 3.8393 A (along [-110], beam direction)
//
// 4 atoms per projected unit cell at:
//   (0.00,  0.00,  0.00)
//   (0.50,  0.50,  0.25)
//   (0.50,  0.00,  0.50)
//   (0.00,  0.50,  0.75)

struct TestCase {
    const char* name;
    int nx, ny;
    int n_cells_x, n_cells_y, n_cells_z;
    double spec_dz;  // slice thickness (A)
    const char* sim_type;
};

// Test cases at realistic pixel sizes (~1.0 A/px).
// Si [110] unit cell: ux=7.68 A, uy=5.43 A, uz=3.84 A.
// Grid is filled with specimen at ~1 A/px, so n_cells_x ~ nx/7.68, n_cells_y ~ ny/5.43.
// Thickness controlled by n_cells_z: 14 cells ~ 54 A, 53 ~ 204 A, 131 ~ 503 A.
//
// VRAM estimates (grid + atoms):
//   T1:   9 MB   T2:  54 MB   T3: 215 MB   T4: 376 MB
//   T5: 858 MB   T6: 1.5 GB   T7: 3.4 GB   T8: 6.0 GB
//
static const TestCase TEST_CASES[] = {
    // T1: Quick smoke test (~1-2s)
    {"T1", 256, 256, 34, 48, 14, 1.0, "EWRS"},
    // T2: Light benchmark (~5s)
    {"T2", 512, 512, 67, 95, 27, 1.0, "EWRS"},
    // T3: Medium grid, thin specimen (~10s)
    {"T3", 1024, 1024, 134, 189, 27, 1.0, "EWRS"},
    // T4: Medium grid, 200 A thick (~20s)
    {"T4", 1024, 1024, 134, 189, 53, 1.0, "EWRS"},
    // T5: Medium grid, 500 A thick (~50s)
    {"T5", 1024, 1024, 134, 189, 131, 1.0, "EWRS"},
    // T6: Large grid, 200 A thick (~2-4 min, 1.5 GB VRAM)
    {"T6", 2048, 2048, 267, 378, 53, 1.0, "EWRS"},
    // T7, T8 removed: require >8 GB VRAM
    // Restore when benchmarking on a larger-memory GPU
};
static const int N_TEST_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);

// Build a Si [110] specimen with the given number of unit cells
void build_si110_specimen(
        int ncx, int ncy, int ncz,
        std::vector<multem::Atom>& atoms,
        double& lx, double& ly, double& lz) {

    const double a = 5.4307;
    const int Z_Si = 14;
    const double sigma = 0.085;  // Debye-Waller (A)
    const double occ = 1.0;

    // Unit cell dimensions in [110] projection
    const double ux = a * std::sqrt(2.0);  // 7.6785 A
    const double uy = a;                    // 5.4307 A
    const double uz = a / std::sqrt(2.0);  // 3.8393 A

    lx = ux * ncx;
    ly = uy * ncy;
    lz = uz * ncz;

    // Fractional positions of 4 atoms in projected [110] unit cell
    const double frac[4][3] = {
        {0.00, 0.00, 0.00},
        {0.50, 0.50, 0.25},
        {0.50, 0.00, 0.50},
        {0.00, 0.50, 0.75},
    };

    atoms.clear();
    atoms.reserve(4 * ncx * ncy * ncz);

    for (int iz = 0; iz < ncz; iz++) {
        for (int iy = 0; iy < ncy; iy++) {
            for (int ix = 0; ix < ncx; ix++) {
                for (int ia = 0; ia < 4; ia++) {
                    double x = (ix + frac[ia][0]) * ux;
                    double y = (iy + frac[ia][1]) * uy;
                    double z = (iz + frac[ia][2]) * uz;
                    atoms.emplace_back(Z_Si, x, y, z, sigma, occ, 0, 0);
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Output I/O
// -------------------------------------------------------------------------

// Save a complex array as two binary files (real + imag parts as float32)
bool save_complex_array(
        const std::string& prefix,
        const multem::Image<std::complex<double>>& img) {

    std::size_t n = img.data.size();
    if (n == 0) return false;

    std::vector<float> real_part(n), imag_part(n);
    for (std::size_t i = 0; i < n; i++) {
        real_part[i] = static_cast<float>(img.data[i].real());
        imag_part[i] = static_cast<float>(img.data[i].imag());
    }

    std::string fname_r = prefix + "_real.bin";
    std::string fname_i = prefix + "_imag.bin";

    std::ofstream fr(fname_r, std::ios::binary);
    std::ofstream fi(fname_i, std::ios::binary);
    if (!fr || !fi) return false;

    fr.write(reinterpret_cast<const char*>(real_part.data()), n * sizeof(float));
    fi.write(reinterpret_cast<const char*>(imag_part.data()), n * sizeof(float));
    return true;
}

// Save intensity (m2psi_tot) as float32 binary
bool save_intensity(
        const std::string& prefix,
        const multem::Image<double>& img) {

    std::size_t n = img.data.size();
    if (n == 0) return false;

    std::vector<float> data(n);
    for (std::size_t i = 0; i < n; i++) {
        data[i] = static_cast<float>(img.data[i]);
    }

    std::string fname = prefix + "_intensity.bin";
    std::ofstream f(fname, std::ios::binary);
    if (!f) return false;

    f.write(reinterpret_cast<const char*>(data.data()), n * sizeof(float));
    return true;
}

// -------------------------------------------------------------------------
// Run a single test case
// -------------------------------------------------------------------------

struct RunResult {
    bool success;
    double elapsed_ms;
    int nx, ny;
    std::size_t n_atoms;
    int n_slices_approx;
};

RunResult run_test(
        const TestCase& tc,
        const std::string& device,
        const std::string& precision,
        int gpu_id,
        const std::string& output_dir) {

    RunResult result = {};

    // Build specimen
    std::vector<multem::Atom> atoms;
    double lx, ly, lz;
    build_si110_specimen(tc.n_cells_x, tc.n_cells_y, tc.n_cells_z,
                         atoms, lx, ly, lz);

    // MULTEM_TEST_SHUFFLE=1: present the atoms in a fixed pseudo-random order instead of the
    // generated z-major one. Nearly-sorted input flatters any sort that adapts to its input, and a
    // specimen read from a model file is not nearly sorted. Same atoms and seed on every run.
    const char *shuffle = std::getenv("MULTEM_TEST_SHUFFLE");
    const bool shuffled = shuffle && shuffle[0] == '1';
    if (shuffled) {
        std::shuffle(atoms.begin(), atoms.end(), std::mt19937_64(12345));
    }

    // MULTEM_TEST_PHONONS=N: N frozen-phonon configurations instead of still atoms, to exercise the
    // per-configuration path (move_atoms displaces the atoms and re-derives their statistics each time).
    const char *phonons_env = std::getenv("MULTEM_TEST_PHONONS");
    const int phonons = phonons_env ? std::atoi(phonons_env) : 0;

    result.n_atoms = atoms.size();
    result.nx = tc.nx;
    result.ny = tc.ny;
    result.n_slices_approx = static_cast<int>(std::ceil(lz / tc.spec_dz));

    printf("  Test: %s\n", tc.name);
    printf("  Grid: %d x %d\n", tc.nx, tc.ny);
    printf("  Atoms: %zu (%s order, %s)\n", atoms.size(), shuffled ? "shuffled" : "generated z-major",
           phonons > 0 ? (std::to_string(phonons) + " frozen-phonon configurations").c_str() : "still atoms");
    printf("  Specimen: %.1f x %.1f x %.1f A\n", lx, ly, lz);
    printf("  Slices: ~%d (dz=%.1f A)\n", result.n_slices_approx, tc.spec_dz);
    printf("  Device: %s, Precision: %s\n", device.c_str(), precision.c_str());

    // Configure simulation
    multem::SystemConfiguration config;
    config.device = device;
    config.precision = precision;
    config.gpu_device = gpu_id;

    multem::Input input;
    input.simulation_type = tc.sim_type;
    input.interaction_model = "Multislice";
    input.potential_type = "Lobato_0_12";
    input.potential_slicing = "dz_Proj";

    input.pn_model = "Still_Atom";
    input.pn_nconf = 1;
    if (phonons > 0) {
        input.pn_model = "Frozen_Phonon";
        input.pn_nconf = phonons;
    }

    input.spec_atoms = atoms;
    input.spec_lx = lx;
    input.spec_ly = ly;
    input.spec_lz = lz;
    input.spec_dz = tc.spec_dz;

    input.nx = tc.nx;
    input.ny = tc.ny;
    input.bwl = false;

    input.E_0 = 300.0;  // 300 kV
    input.theta = 0.0;
    input.phi = 0.0;

    input.thick_type = "Whole_Spec";

    // Objective lens: Scherzer defocus for Cs = 1e-3 mm at 300 kV
    input.obj_lens_c_10 = 14.0312;  // defocus (A)
    input.obj_lens_c_30 = 1e-3;     // Cs (mm)

    // Run simulation with timing
    auto t0 = std::chrono::high_resolution_clock::now();

    multem::Output output;
    try {
        output = multem::simulate(config, input);
    } catch (const std::exception& e) {
        fprintf(stderr, "  ERROR: %s\n", e.what());
        result.success = false;
        return result;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  Time: %.1f ms (%.3f s)\n", result.elapsed_ms, result.elapsed_ms / 1000.0);
    printf("  Output thicknesses: %zu\n", output.thick.size());

    // Validate output
    if (output.data.empty()) {
        fprintf(stderr, "  ERROR: No output data\n");
        result.success = false;
        return result;
    }

    auto& data = output.data[0];
    bool has_psi = !data.psi_coh.data.empty();
    bool has_m2psi = !data.m2psi_tot.data.empty();

    printf("  psi_coh: %s (%zu x %zu)\n",
           has_psi ? "yes" : "no",
           has_psi ? data.psi_coh.shape[0] : 0,
           has_psi ? data.psi_coh.shape[1] : 0);
    printf("  m2psi_tot: %s (%zu x %zu)\n",
           has_m2psi ? "yes" : "no",
           has_m2psi ? data.m2psi_tot.shape[0] : 0,
           has_m2psi ? data.m2psi_tot.shape[1] : 0);

    if (!has_psi && !has_m2psi) {
        fprintf(stderr, "  ERROR: No output arrays produced\n");
        result.success = false;
        return result;
    }

    // Save output
    if (!output_dir.empty()) {
        std::string prefix = output_dir + "/" + tc.name;

        if (has_psi) {
            if (save_complex_array(prefix, data.psi_coh)) {
                printf("  Saved: %s_real.bin, %s_imag.bin\n", prefix.c_str(), prefix.c_str());
            }
        }
        if (has_m2psi) {
            if (save_intensity(prefix, data.m2psi_tot)) {
                printf("  Saved: %s_intensity.bin\n", prefix.c_str());
            }
        }

        // Write metadata
        std::string meta_file = prefix + "_meta.txt";
        FILE* mf = fopen(meta_file.c_str(), "w");
        if (mf) {
            fprintf(mf, "test=%s\n", tc.name);
            fprintf(mf, "nx=%d\n", tc.nx);
            fprintf(mf, "ny=%d\n", tc.ny);
            fprintf(mf, "n_atoms=%zu\n", atoms.size());
            fprintf(mf, "spec_lx=%.4f\n", lx);
            fprintf(mf, "spec_ly=%.4f\n", ly);
            fprintf(mf, "spec_lz=%.4f\n", lz);
            fprintf(mf, "spec_dz=%.4f\n", tc.spec_dz);
            fprintf(mf, "n_slices=%d\n", result.n_slices_approx);
            fprintf(mf, "E_0=%.1f\n", input.E_0);
            fprintf(mf, "device=%s\n", device.c_str());
            fprintf(mf, "precision=%s\n", precision.c_str());
            fprintf(mf, "elapsed_ms=%.1f\n", result.elapsed_ms);
            fclose(mf);
        }
    }

    result.success = true;
    return result;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

void print_usage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --test T1|T2|...|T8|all    Test case (default: T1)\n");
    printf("  --device host|device        CPU or GPU (default: device)\n");
    printf("  --precision float|double    Precision (default: float)\n");
    printf("  --output DIR                Output directory (default: .)\n");
    printf("  --repeats N                 Timed repeats (default: 1)\n");
    printf("  --gpu-id N                  GPU device index (default: 0)\n");
}

int main(int argc, char** argv) {
    std::string test_name = "T1";
#ifdef MULTEM_NO_CUDA
    std::string device = "host";
#else
    std::string device = "device";
#endif
    std::string precision = "float";
    std::string output_dir = ".";
    int repeats = 1;
    int gpu_id = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_name = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = argv[++i];
        } else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            precision = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-id") == 0 && i + 1 < argc) {
            gpu_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("MULTEM Test Harness\n");
    printf("===================\n\n");

    // Determine which tests to run
    std::vector<const TestCase*> tests;
    if (test_name == "all") {
        for (int i = 0; i < N_TEST_CASES; i++) {
            tests.push_back(&TEST_CASES[i]);
        }
    } else {
        for (int i = 0; i < N_TEST_CASES; i++) {
            if (test_name == TEST_CASES[i].name) {
                tests.push_back(&TEST_CASES[i]);
                break;
            }
        }
        if (tests.empty()) {
            fprintf(stderr, "Unknown test: %s\n", test_name.c_str());
            fprintf(stderr, "Available: T1 T2 T3 T4 T5 T6 T7 T8 all\n");
            return 1;
        }
    }

    // Run tests
    int n_passed = 0, n_failed = 0;

    for (const auto* tc : tests) {
        printf("\n--- %s ---\n", tc->name);

        for (int r = 0; r < repeats; r++) {
            if (repeats > 1) {
                printf("\n  [Repeat %d/%d]\n", r + 1, repeats);
            }

            // Only save output on first repeat
            std::string out = (r == 0) ? output_dir : "";
            RunResult result = run_test(*tc, device, precision, gpu_id, out);

            if (result.success) {
                n_passed++;
            } else {
                n_failed++;
            }
        }
    }

    printf("\n===================\n");
    printf("Results: %d passed, %d failed\n", n_passed, n_failed);

    return (n_failed > 0) ? 1 : 0;
}
