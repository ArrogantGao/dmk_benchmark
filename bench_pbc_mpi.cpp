#include <dmk.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>

namespace {

// ── helpers ──────────────────────────────────────────────────────────────────

double safe_relative_l2(double err2, double ref2) {
    return ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
}

int get_actual_threads() {
    int n = 1;
#pragma omp parallel
    {
#pragma omp single
        n = omp_get_num_threads();
    }
    return n;
}

// Generate n_total particles on rank 0, then scatter n_total/size to each rank.
// Every rank gets r_local (3*n_local), charges_local (n_local), etc.
// Rank 0 also fills r_all, charges_all with the full problem (for reference).
void make_and_scatter(MPI_Comm comm, int n_total, bool with_grad,
                      // outputs — local per rank
                      std::vector<double> &r_local, std::vector<double> &charges_local,
                      std::vector<double> &rnormal_local, std::vector<double> &dipstr_local,
                      std::vector<double> &r_trg_local,
                      int &n_local_src, int &n_local_trg,
                      // outputs — full problem on rank 0 only
                      std::vector<double> &r_all, std::vector<double> &charges_all,
                      std::vector<double> &rnormal_all, std::vector<double> &dipstr_all,
                      std::vector<double> &r_trg_all) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Round n_total down to a multiple of size
    const int n_src_total = (n_total / size) * size;
    const int n_trg_total = n_src_total; // same count for targets
    n_local_src = n_src_total / size;
    n_local_trg = n_trg_total / size;

    if (rank == 0) {
        r_all.resize(3 * n_src_total);
        charges_all.resize(n_src_total);
        rnormal_all.assign(3 * n_src_total, 0.0);
        dipstr_all.assign(n_src_total, 0.0);
        r_trg_all.resize(3 * n_trg_total);

        std::default_random_engine eng(42 + n_total);
        std::uniform_real_distribution<double> rng(0.01, 0.99);

        for (double &v : r_all)
            v = rng(eng);
        for (double &v : r_trg_all)
            v = rng(eng);
        for (int i = 0; i < n_src_total; ++i)
            charges_all[i] = rng(eng) - 0.5;

        // Enforce charge neutrality
        double sum = 0.0;
        for (double q : charges_all)
            sum += q;
        for (double &q : charges_all)
            q -= sum / n_src_total;
    }

    // Scatter sources
    r_local.resize(3 * n_local_src);
    charges_local.resize(n_local_src);
    rnormal_local.assign(3 * n_local_src, 0.0);
    dipstr_local.assign(n_local_src, 0.0);
    r_trg_local.resize(3 * n_local_trg);

    MPI_Scatter(rank == 0 ? r_all.data() : nullptr, 3 * n_local_src, MPI_DOUBLE,
                r_local.data(), 3 * n_local_src, MPI_DOUBLE, 0, comm);
    MPI_Scatter(rank == 0 ? charges_all.data() : nullptr, n_local_src, MPI_DOUBLE,
                charges_local.data(), n_local_src, MPI_DOUBLE, 0, comm);
    MPI_Scatter(rank == 0 ? rnormal_all.data() : nullptr, 3 * n_local_src, MPI_DOUBLE,
                rnormal_local.data(), 3 * n_local_src, MPI_DOUBLE, 0, comm);
    MPI_Scatter(rank == 0 ? dipstr_all.data() : nullptr, n_local_src, MPI_DOUBLE,
                dipstr_local.data(), n_local_src, MPI_DOUBLE, 0, comm);
    // Scatter targets
    MPI_Scatter(rank == 0 ? r_trg_all.data() : nullptr, 3 * n_local_trg, MPI_DOUBLE,
                r_trg_local.data(), 3 * n_local_trg, MPI_DOUBLE, 0, comm);
}

// ── accuracy mode ────────────────────────────────────────────────────────────

void run_accuracy(int n_total, double eps, int digits) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Scatter problem
    std::vector<double> r_local, charges_local, rnormal_local, dipstr_local, r_trg_local;
    std::vector<double> r_all, charges_all, rnormal_all, dipstr_all, r_trg_all;
    int n_local_src, n_local_trg;
    make_and_scatter(MPI_COMM_WORLD, n_total, true,
                     r_local, charges_local, rnormal_local, dipstr_local, r_trg_local,
                     n_local_src, n_local_trg,
                     r_all, charges_all, rnormal_all, dipstr_all, r_trg_all);

    const int n_src_total = n_local_src * size;
    const int n_trg_total = n_local_trg * size;
    const int odim = 4; // pot + grad

    // ── MPI run ──
    pdmk_params params{};
    params.n_dim = 3;
    params.eps = eps;
    params.n_per_leaf = 100;
    params.pgh_src = DMK_POTENTIAL_GRAD;
    params.pgh_trg = DMK_POTENTIAL_GRAD;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = 1;
    params.log_level = 6;

    std::vector<double> pot_src_local(n_local_src * odim);
    std::vector<double> pot_trg_local(n_local_trg * odim);

    pdmk_tree tree = pdmk_tree_create(MPI_COMM_WORLD, params, n_local_src, r_local.data(),
                                      charges_local.data(), rnormal_local.data(),
                                      dipstr_local.data(), n_local_trg, r_trg_local.data());
    pdmk_tree_eval(tree, pot_src_local.data(), pot_trg_local.data());
    pdmk_tree_destroy(tree);

    // Gather MPI results to rank 0
    std::vector<double> pot_src_mpi, pot_trg_mpi;
    if (rank == 0) {
        pot_src_mpi.resize(n_src_total * odim);
        pot_trg_mpi.resize(n_trg_total * odim);
    }
    MPI_Gather(pot_src_local.data(), n_local_src * odim, MPI_DOUBLE,
               rank == 0 ? pot_src_mpi.data() : nullptr, n_local_src * odim, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    MPI_Gather(pot_trg_local.data(), n_local_trg * odim, MPI_DOUBLE,
               rank == 0 ? pot_trg_mpi.data() : nullptr, n_local_trg * odim, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // ── Reference: single-rank run on rank 0 ──
    if (rank == 0) {
        std::vector<double> pot_src_ref(n_src_total * odim);
        std::vector<double> pot_trg_ref(n_trg_total * odim);

        pdmk_tree ref_tree = pdmk_tree_create(MPI_COMM_SELF, params, n_src_total,
                                              r_all.data(), charges_all.data(),
                                              rnormal_all.data(), dipstr_all.data(),
                                              n_trg_total, r_trg_all.data());
        pdmk_tree_eval(ref_tree, pot_src_ref.data(), pot_trg_ref.data());
        pdmk_tree_destroy(ref_tree);

        // Compute relative L2 errors: MPI vs single-rank
        double err2_pot_src = 0, ref2_pot_src = 0;
        double err2_pot_trg = 0, ref2_pot_trg = 0;
        double err2_grad_src = 0, ref2_grad_src = 0;
        double err2_grad_trg = 0, ref2_grad_trg = 0;

        for (int i = 0; i < n_src_total; ++i) {
            double d = pot_src_mpi[i * odim] - pot_src_ref[i * odim];
            err2_pot_src += d * d;
            ref2_pot_src += pot_src_ref[i * odim] * pot_src_ref[i * odim];
            for (int g = 1; g < odim; ++g) {
                double dg = pot_src_mpi[i * odim + g] - pot_src_ref[i * odim + g];
                err2_grad_src += dg * dg;
                ref2_grad_src += pot_src_ref[i * odim + g] * pot_src_ref[i * odim + g];
            }
        }
        for (int i = 0; i < n_trg_total; ++i) {
            double d = pot_trg_mpi[i * odim] - pot_trg_ref[i * odim];
            err2_pot_trg += d * d;
            ref2_pot_trg += pot_trg_ref[i * odim] * pot_trg_ref[i * odim];
            for (int g = 1; g < odim; ++g) {
                double dg = pot_trg_mpi[i * odim + g] - pot_trg_ref[i * odim + g];
                err2_grad_trg += dg * dg;
                ref2_grad_trg += pot_trg_ref[i * odim + g] * pot_trg_ref[i * odim + g];
            }
        }

        std::printf("%d,%d,%d,%.0e,%.6e,%.6e,%.6e,%.6e\n",
                    n_src_total, size, digits, eps,
                    safe_relative_l2(err2_pot_src, ref2_pot_src),
                    safe_relative_l2(err2_pot_trg, ref2_pot_trg),
                    safe_relative_l2(err2_grad_src, ref2_grad_src),
                    safe_relative_l2(err2_grad_trg, ref2_grad_trg));
    }
}

// ── speed mode ───────────────────────────────────────────────────────────────

void run_speed(int n_total, double eps, int n_runs) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> r_local, charges_local, rnormal_local, dipstr_local, r_trg_local;
    std::vector<double> r_all, charges_all, rnormal_all, dipstr_all, r_trg_all;
    int n_local_src, n_local_trg;
    make_and_scatter(MPI_COMM_WORLD, n_total, true,
                     r_local, charges_local, rnormal_local, dipstr_local, r_trg_local,
                     n_local_src, n_local_trg,
                     r_all, charges_all, rnormal_all, dipstr_all, r_trg_all);

    const int n_src_total = n_local_src * size;
    const int odim = 4;

    pdmk_params params{};
    params.n_dim = 3;
    params.eps = eps;
    params.n_per_leaf = 500;
    params.pgh_src = DMK_POTENTIAL_GRAD;
    params.pgh_trg = DMK_POTENTIAL_GRAD;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = 1;
    params.log_level = 6;

    std::vector<double> pot_src(n_local_src * odim);
    std::vector<double> pot_trg(n_local_trg * odim);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_create = -MPI_Wtime();
    pdmk_tree tree = pdmk_tree_create(MPI_COMM_WORLD, params, n_local_src, r_local.data(),
                                      charges_local.data(), rnormal_local.data(),
                                      dipstr_local.data(), n_local_trg, r_trg_local.data());
    MPI_Barrier(MPI_COMM_WORLD);
    t_create += MPI_Wtime();

    // Warmup
    MPI_Barrier(MPI_COMM_WORLD);
    double t_warmup = -MPI_Wtime();
    pdmk_tree_eval(tree, pot_src.data(), pot_trg.data());
    MPI_Barrier(MPI_COMM_WORLD);
    t_warmup += MPI_Wtime();

    // Timed runs
    double t_min = 1e30, t_max = 0, t_sum = 0;
    for (int run = 0; run < n_runs; ++run) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        pdmk_tree_eval(tree, pot_src.data(), pot_trg.data());
        MPI_Barrier(MPI_COMM_WORLD);
        double dt = MPI_Wtime() - t0;
        t_min = std::min(t_min, dt);
        t_max = std::max(t_max, dt);
        t_sum += dt;
    }

    pdmk_tree_destroy(tree);

    if (rank == 0) {
        const double t_avg = t_sum / n_runs;
        const int omp_threads = get_actual_threads();
        std::printf("pbc,%d,%d,%d,%.0e,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    n_src_total, size, omp_threads, eps, 500,
                    t_create, t_warmup, t_min, t_avg, t_max,
                    t_create + t_min,
                    n_src_total / t_min / 1e6);
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

void print_usage(const char *prog) {
    std::fprintf(stderr,
                 "Usage: %s --mode accuracy|speed [--n N] [--eps EPS] [--digits D] [--n-runs K]\n",
                 prog);
}

} // namespace

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Defaults
    std::string mode = "accuracy";
    int n_total = 10000;
    double eps = 1e-3;
    int digits = 3;
    int n_runs = 3;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto val = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            return argv[++i];
        };
        if (arg == "--mode")        mode = val("--mode");
        else if (arg == "--n")      n_total = std::atoi(val("--n"));
        else if (arg == "--eps")    eps = std::atof(val("--eps"));
        else if (arg == "--digits") digits = std::atoi(val("--digits"));
        else if (arg == "--n-runs") n_runs = std::atoi(val("--n-runs"));
        else { if (rank == 0) print_usage(argv[0]); MPI_Finalize(); return 1; }
    }

    if (mode == "accuracy") {
        if (rank == 0)
            std::printf("N,mpi_ranks,digits,eps,pot_src_l2,pot_trg_l2,grad_src_l2,grad_trg_l2\n");

        const int sizes[] = {2000, 5000, 10000};
        const int digs[]  = {3, 6, 9};
        const double epss[] = {1e-3, 1e-6, 1e-9};

        for (int n : sizes)
            for (int d = 0; d < 3; ++d)
                run_accuracy(n, epss[d], digs[d]);

    } else if (mode == "speed") {
        if (rank == 0)
            std::printf("mode,n_particles,mpi_ranks,omp_threads,eps,n_per_leaf,"
                        "create_seconds,eval_warmup_seconds,eval_min_seconds,"
                        "eval_avg_seconds,eval_max_seconds,total_min_seconds,mpts_per_sec\n");

        run_speed(n_total, eps, n_runs);

    } else {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
