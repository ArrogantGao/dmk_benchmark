#include <dmk.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>

namespace {

struct Config {
    int n_particles = 500000;
    int n_runs = 5;
    double eps = 1e-3;
    int n_per_leaf = 500;
    std::string pbc_mode = "both";
};

struct TimingRow {
    std::string mode;
    int n_particles;
    int omp_threads_requested;
    int omp_threads_actual;
    double eps;
    int n_per_leaf;
    double create_seconds;
    double eval_warmup_seconds;
    double eval_min_seconds;
    double eval_avg_seconds;
    double eval_max_seconds;
    double total_min_seconds;
    double mpts_per_sec;
};

Config parse_args(int argc, char **argv) {
    Config cfg{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--n")
            cfg.n_particles = std::atoi(require_value("--n"));
        else if (arg == "--n-runs")
            cfg.n_runs = std::atoi(require_value("--n-runs"));
        else if (arg == "--eps")
            cfg.eps = std::atof(require_value("--eps"));
        else if (arg == "--n-per-leaf")
            cfg.n_per_leaf = std::atoi(require_value("--n-per-leaf"));
        else if (arg == "--pbc")
            cfg.pbc_mode = require_value("--pbc");
        else {
            std::fprintf(stderr,
                         "Usage: %s [--n N] [--n-runs K] [--eps EPS] [--n-per-leaf NLEAF] [--pbc both|true|false]\n",
                         argv[0]);
            std::exit(1);
        }
    }
    if (cfg.pbc_mode != "both" && cfg.pbc_mode != "true" && cfg.pbc_mode != "false") {
        std::fprintf(stderr, "--pbc must be one of both|true|false\n");
        std::exit(1);
    }
    return cfg;
}

int get_actual_threads() {
    int n_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        n_threads = omp_get_num_threads();
    }
    return n_threads;
}

void make_problem(int n_particles, std::vector<double> &r_src, std::vector<double> &charges, std::vector<double> &rnormal,
                  std::vector<double> &dipstr) {
    std::default_random_engine eng(42 + n_particles);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    r_src.resize(3 * n_particles);
    charges.resize(n_particles);
    rnormal.assign(3 * n_particles, 0.0);
    dipstr.assign(n_particles, 0.0);

    for (double &v : r_src)
        v = rng(eng);
    for (int i = 0; i < n_particles; ++i)
        charges[i] = rng(eng) - 0.5;

    double sum = 0.0;
    for (double q : charges)
        sum += q;
    for (double &q : charges)
        q -= sum / n_particles;
}

TimingRow run_case(const Config &cfg, const std::vector<double> &r_src, const std::vector<double> &charges,
                   const std::vector<double> &rnormal, const std::vector<double> &dipstr, bool use_periodic) {
    const int n_particles = cfg.n_particles;
    const int output_dim = 4;
    std::vector<double> pot_src(n_particles * output_dim);

    pdmk_params params;
    params.n_dim = 3;
    params.eps = cfg.eps;
    params.n_per_leaf = cfg.n_per_leaf;
    params.pgh_src = DMK_POTENTIAL_GRAD;
    params.pgh_trg = DMK_POTENTIAL_GRAD;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = use_periodic ? 1 : 0;
    params.log_level = 6;

    const int omp_threads_requested = omp_get_max_threads();
    const int omp_threads_actual = get_actual_threads();

    double t_create = -omp_get_wtime();
    pdmk_tree tree = pdmk_tree_create(MPI_COMM_WORLD, params, n_particles, r_src.data(), charges.data(), rnormal.data(),
                                      dipstr.data(), 0, nullptr);
    t_create += omp_get_wtime();

    double t_warmup = -omp_get_wtime();
    pdmk_tree_eval(tree, pot_src.data(), nullptr);
    t_warmup += omp_get_wtime();

    double t_min = 1e300;
    double t_max = 0.0;
    double t_sum = 0.0;
    for (int run = 0; run < cfg.n_runs; ++run) {
        double t0 = omp_get_wtime();
        pdmk_tree_eval(tree, pot_src.data(), nullptr);
        const double dt = omp_get_wtime() - t0;
        t_min = std::min(t_min, dt);
        t_max = std::max(t_max, dt);
        t_sum += dt;
    }

    pdmk_tree_destroy(tree);

    TimingRow row{};
    row.mode = use_periodic ? "pbc" : "free";
    row.n_particles = n_particles;
    row.omp_threads_requested = omp_threads_requested;
    row.omp_threads_actual = omp_threads_actual;
    row.eps = cfg.eps;
    row.n_per_leaf = cfg.n_per_leaf;
    row.create_seconds = t_create;
    row.eval_warmup_seconds = t_warmup;
    row.eval_min_seconds = t_min;
    row.eval_avg_seconds = t_sum / cfg.n_runs;
    row.eval_max_seconds = t_max;
    row.total_min_seconds = t_create + t_min;
    row.mpts_per_sec = n_particles / t_min / 1e6;
    return row;
}

void print_csv_header() {
    std::printf("mode,n_particles,omp_threads_requested,omp_threads_actual,eps,n_per_leaf,create_seconds,"
                "eval_warmup_seconds,eval_min_seconds,eval_avg_seconds,eval_max_seconds,total_min_seconds,mpts_per_sec\n");
}

void print_row(const TimingRow &row) {
    std::printf("%s,%d,%d,%d,%.0e,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", row.mode.c_str(), row.n_particles,
                row.omp_threads_requested, row.omp_threads_actual, row.eps, row.n_per_leaf, row.create_seconds,
                row.eval_warmup_seconds, row.eval_min_seconds, row.eval_avg_seconds, row.eval_max_seconds,
                row.total_min_seconds, row.mpts_per_sec);
}

} // namespace

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const Config cfg = parse_args(argc, argv);

    std::vector<double> r_src, charges, rnormal, dipstr;
    make_problem(cfg.n_particles, r_src, charges, rnormal, dipstr);

    std::vector<TimingRow> rows;
    if (cfg.pbc_mode == "both" || cfg.pbc_mode == "false")
        rows.push_back(run_case(cfg, r_src, charges, rnormal, dipstr, false));
    if (cfg.pbc_mode == "both" || cfg.pbc_mode == "true")
        rows.push_back(run_case(cfg, r_src, charges, rnormal, dipstr, true));

    if (rank == 0) {
        print_csv_header();
        for (const auto &row : rows)
            print_row(row);
    }

    MPI_Finalize();
    return 0;
}
