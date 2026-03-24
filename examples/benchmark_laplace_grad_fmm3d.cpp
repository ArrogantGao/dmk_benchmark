#include <dmk.h>
#include <cstdint>

extern "C" {
void lfmm3d_st_c_g_(double *eps, int64_t *nsource, double *source, double *charge, double *pot, double *grad,
                    int64_t *nt, double *targ, double *pottarg, double *gradtarg, int64_t *ier);
}

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <random>

namespace fs = std::filesystem;

namespace {

struct Config {
    std::string mode = "benchmark";
    std::string out_dir = "results/laplace_grad_fmm3d";
    std::string csv_path;
    int n_src = 1'000'000;
    int n_trg = 10'000;
    int n_dim = 3;
    int n_mfm = 1;
    int n_per_leaf = 280;
    int n_runs = 5;
    int warmup_runs = 1;
    long seed = 0;
    bool uniform = false;
    bool set_fixed_charges = true;
    double eps = 1e-6;
    double reference_eps = 1e-14;
    int log_level = DMK_LOG_OFF;
};

struct DatasetMeta {
    int n_dim = 3;
    int n_src = 0;
    int n_trg = 0;
    int n_mfm = 1;
    long seed = 0;
    int uniform = 0;
    int set_fixed_charges = 1;
    double reference_eps = 1e-14;
    std::string kernel = "laplace";
    std::string reference_kernel_normalization = "1/(4*pi*r)";
    std::string dmk_output_normalization = "1/r";
};

struct DatasetArtifacts {
    DatasetMeta meta;
    std::vector<double> sources;
    std::vector<double> targets;
    std::vector<double> charges;
    std::vector<double> pot_src_ref;
    std::vector<double> pot_trg_ref;
    std::vector<double> grad_src_ref;
    std::vector<double> grad_trg_ref;
};

struct TimingResult {
    double build_time = 0.0;
    double eval_time = 0.0;
    double total_time = 0.0;
};

struct ErrorMetrics {
    double rel_l2 = 0.0;
    double max_rel = 0.0;
};

int local_count(int n, int np, int rank) { return n / np + (rank < (n % np) ? 1 : 0); }

int get_omp_threads() {
    int n_threads = 1;
#pragma omp parallel
    n_threads = omp_get_num_threads();
    return n_threads;
}

double reduce_max(double local_value, MPI_Comm comm) {
    double global_value = 0.0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_value;
}

template <typename T>
void write_binary_file(const fs::path &path, const std::vector<T> &data) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Failed to open " + path.string() + " for writing");
    ofs.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!ofs)
        throw std::runtime_error("Failed while writing " + path.string());
}

template <typename T>
std::vector<T> read_binary_file(const fs::path &path, std::size_t expected_count) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Failed to open " + path.string() + " for reading");
    std::vector<T> data(expected_count);
    ifs.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!ifs)
        throw std::runtime_error("Failed while reading " + path.string());
    return data;
}

void write_metadata(const fs::path &path, const DatasetMeta &meta) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("Failed to open " + path.string() + " for writing metadata");
    ofs << "n_dim: " << meta.n_dim << '\n';
    ofs << "n_src: " << meta.n_src << '\n';
    ofs << "n_trg: " << meta.n_trg << '\n';
    ofs << "n_mfm: " << meta.n_mfm << '\n';
    ofs << "seed: " << meta.seed << '\n';
    ofs << "uniform: " << meta.uniform << '\n';
    ofs << "set_fixed_charges: " << meta.set_fixed_charges << '\n';
    ofs << "reference_eps: " << std::setprecision(17) << meta.reference_eps << '\n';
    ofs << "kernel: " << meta.kernel << '\n';
    ofs << "reference_kernel_normalization: " << meta.reference_kernel_normalization << '\n';
    ofs << "dmk_output_normalization: " << meta.dmk_output_normalization << '\n';
}

DatasetMeta read_metadata(const fs::path &path) {
    std::ifstream ifs(path);
    if (!ifs)
        throw std::runtime_error("Failed to open " + path.string() + " for reading metadata");

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(ifs, line)) {
        const auto pos = line.find(':');
        if (pos == std::string::npos)
            continue;
        auto key = line.substr(0, pos);
        auto value = line.substr(pos + 1);
        while (!value.empty() && value.front() == ' ')
            value.erase(value.begin());
        kv[key] = value;
    }

    DatasetMeta meta;
    meta.n_dim = std::stoi(kv.at("n_dim"));
    meta.n_src = std::stoi(kv.at("n_src"));
    meta.n_trg = std::stoi(kv.at("n_trg"));
    meta.n_mfm = std::stoi(kv.at("n_mfm"));
    meta.seed = std::stol(kv.at("seed"));
    meta.uniform = std::stoi(kv.at("uniform"));
    meta.set_fixed_charges = std::stoi(kv.at("set_fixed_charges"));
    meta.reference_eps = std::stod(kv.at("reference_eps"));
    meta.kernel = kv.at("kernel");
    meta.reference_kernel_normalization = kv.at("reference_kernel_normalization");
    meta.dmk_output_normalization = kv.at("dmk_output_normalization");
    return meta;
}

DatasetArtifacts generate_reference_artifacts(const Config &cfg) {
    DatasetArtifacts artifacts;
    artifacts.meta.n_dim = cfg.n_dim;
    artifacts.meta.n_src = cfg.n_src;
    artifacts.meta.n_trg = cfg.n_trg;
    artifacts.meta.n_mfm = cfg.n_mfm;
    artifacts.meta.seed = cfg.seed;
    artifacts.meta.uniform = cfg.uniform ? 1 : 0;
    artifacts.meta.set_fixed_charges = cfg.set_fixed_charges ? 1 : 0;
    artifacts.meta.reference_eps = cfg.reference_eps;

    artifacts.sources.assign(static_cast<std::size_t>(cfg.n_dim) * cfg.n_src, 0.0);
    artifacts.targets.assign(static_cast<std::size_t>(cfg.n_dim) * cfg.n_trg, 0.0);
    artifacts.charges.assign(static_cast<std::size_t>(cfg.n_mfm) * cfg.n_src, 0.0);

    constexpr double rin = 0.45;
    constexpr double rwig = 0.0;
    constexpr int nwig = 6;
    std::default_random_engine eng(cfg.seed);
    std::uniform_real_distribution<double> rng(0.0, 1.0);

    auto fill_points = [&](std::vector<double> &pts, int n_pts) {
        for (int i = 0; i < n_pts; ++i) {
            if (!cfg.uniform) {
                const double theta = rng(eng) * M_PI;
                const double rr = rin + rwig * std::cos(nwig * theta);
                const double ct = std::cos(theta);
                const double st = std::sin(theta);
                const double phi = rng(eng) * 2.0 * M_PI;
                const double cp = std::cos(phi);
                const double sp = std::sin(phi);
                pts[3 * i + 0] = rr * st * cp + 0.5;
                pts[3 * i + 1] = rr * st * sp + 0.5;
                pts[3 * i + 2] = rr * ct + 0.5;
            } else {
                for (int d = 0; d < cfg.n_dim; ++d)
                    pts[cfg.n_dim * i + d] = rng(eng);
            }
        }
    };

    fill_points(artifacts.sources, cfg.n_src);
    fill_points(artifacts.targets, cfg.n_trg);

    for (int i = 0; i < cfg.n_src; ++i)
        artifacts.charges[i] = rng(eng) - 0.5;

    if (cfg.set_fixed_charges && cfg.n_src > 0)
        for (int d = 0; d < cfg.n_dim; ++d)
            artifacts.sources[d] = 0.0;
    if (cfg.set_fixed_charges && cfg.n_src > 1)
        for (int d = 0; d < cfg.n_dim; ++d)
            artifacts.sources[cfg.n_dim + d] = 1.0 - std::numeric_limits<double>::epsilon();
    if (cfg.set_fixed_charges && cfg.n_src > 2)
        for (int d = 0; d < cfg.n_dim; ++d)
            artifacts.sources[2 * cfg.n_dim + d] = 0.05;

    int64_t n_src = cfg.n_src;
    int64_t n_trg = cfg.n_trg;
    int64_t ier = 0;
    artifacts.pot_src_ref.assign(cfg.n_src, 0.0);
    artifacts.pot_trg_ref.assign(cfg.n_trg, 0.0);
    artifacts.grad_src_ref.assign(static_cast<std::size_t>(3) * cfg.n_src, 0.0);
    artifacts.grad_trg_ref.assign(static_cast<std::size_t>(3) * cfg.n_trg, 0.0);

    lfmm3d_st_c_g_(&artifacts.meta.reference_eps, &n_src, artifacts.sources.data(), artifacts.charges.data(),
                   artifacts.pot_src_ref.data(), artifacts.grad_src_ref.data(), &n_trg, artifacts.targets.data(),
                   artifacts.pot_trg_ref.data(), artifacts.grad_trg_ref.data(), &ier);
    if (ier != 0)
        throw std::runtime_error("FMM3D reference generation failed with ier=" + std::to_string(ier));

    return artifacts;
}

void write_reference_artifacts(const Config &cfg, const DatasetArtifacts &artifacts) {
    const fs::path out_dir(cfg.out_dir);
    fs::create_directories(out_dir);
    write_metadata(out_dir / "metadata.yaml", artifacts.meta);
    write_binary_file(out_dir / "sources.bin", artifacts.sources);
    write_binary_file(out_dir / "targets.bin", artifacts.targets);
    write_binary_file(out_dir / "charges.bin", artifacts.charges);
    write_binary_file(out_dir / "pot_src_ref.bin", artifacts.pot_src_ref);
    write_binary_file(out_dir / "pot_trg_ref.bin", artifacts.pot_trg_ref);
    write_binary_file(out_dir / "grad_src_ref.bin", artifacts.grad_src_ref);
    write_binary_file(out_dir / "grad_trg_ref.bin", artifacts.grad_trg_ref);
}

DatasetArtifacts load_reference_artifacts_root(const Config &cfg) {
    DatasetArtifacts artifacts;
    const fs::path out_dir(cfg.out_dir);
    artifacts.meta = read_metadata(out_dir / "metadata.yaml");
    if (artifacts.meta.n_dim != cfg.n_dim)
        throw std::runtime_error("Metadata n_dim does not match requested configuration");
    if (artifacts.meta.n_src != cfg.n_src)
        throw std::runtime_error("Metadata n_src does not match requested configuration");
    if (artifacts.meta.n_trg != cfg.n_trg)
        throw std::runtime_error("Metadata n_trg does not match requested configuration");
    if (artifacts.meta.n_mfm != cfg.n_mfm)
        throw std::runtime_error("Metadata n_mfm does not match requested configuration");

    artifacts.sources = read_binary_file<double>(out_dir / "sources.bin", static_cast<std::size_t>(cfg.n_dim) * cfg.n_src);
    artifacts.targets = read_binary_file<double>(out_dir / "targets.bin", static_cast<std::size_t>(cfg.n_dim) * cfg.n_trg);
    artifacts.charges = read_binary_file<double>(out_dir / "charges.bin", static_cast<std::size_t>(cfg.n_mfm) * cfg.n_src);
    artifacts.pot_src_ref = read_binary_file<double>(out_dir / "pot_src_ref.bin", cfg.n_src);
    artifacts.pot_trg_ref = read_binary_file<double>(out_dir / "pot_trg_ref.bin", cfg.n_trg);
    artifacts.grad_src_ref =
        read_binary_file<double>(out_dir / "grad_src_ref.bin", static_cast<std::size_t>(cfg.n_dim) * cfg.n_src);
    artifacts.grad_trg_ref =
        read_binary_file<double>(out_dir / "grad_trg_ref.bin", static_cast<std::size_t>(cfg.n_dim) * cfg.n_trg);
    return artifacts;
}

std::vector<int> make_counts(int n_items, int item_width, int np) {
    std::vector<int> counts(np);
    for (int rank = 0; rank < np; ++rank)
        counts[rank] = local_count(n_items, np, rank) * item_width;
    return counts;
}

std::vector<int> make_displacements(const std::vector<int> &counts) {
    std::vector<int> displs(counts.size(), 0);
    for (std::size_t i = 1; i < counts.size(); ++i)
        displs[i] = displs[i - 1] + counts[i - 1];
    return displs;
}

std::vector<double> scatter_vector(const std::vector<double> &global, int n_items, int item_width, MPI_Comm comm) {
    int rank = 0, np = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    const auto counts = make_counts(n_items, item_width, np);
    const auto displs = make_displacements(counts);
    std::vector<double> local(counts[rank]);
    MPI_Scatterv(rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, local.data(),
                 counts[rank], MPI_DOUBLE, 0, comm);
    return local;
}

TimingResult run_dmk_benchmark(const Config &cfg, bool with_grad, const std::vector<double> &local_sources,
                               const std::vector<double> &local_charges, const std::vector<double> &local_targets,
                               std::vector<double> &pot_src, std::vector<double> &grad_src, std::vector<double> &pot_trg,
                               std::vector<double> &grad_trg, MPI_Comm comm) {
    pdmk_params params{};
    params.eps = cfg.eps;
    params.n_dim = cfg.n_dim;
    params.n_per_leaf = cfg.n_per_leaf;
    params.n_mfm = cfg.n_mfm;
    params.pgh_src = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
    params.pgh_trg = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.log_level = cfg.log_level;

    const int n_src_local = static_cast<int>(local_sources.size()) / cfg.n_dim;
    const int n_trg_local = static_cast<int>(local_targets.size()) / cfg.n_dim;

    pot_src.assign(n_src_local, 0.0);
    pot_trg.assign(n_trg_local, 0.0);
    if (with_grad) {
        grad_src.assign(static_cast<std::size_t>(cfg.n_dim) * n_src_local, 0.0);
        grad_trg.assign(static_cast<std::size_t>(cfg.n_dim) * n_trg_local, 0.0);
    } else {
        grad_src.clear();
        grad_trg.clear();
    }

    MPI_Barrier(comm);
    double st = omp_get_wtime();
    pdmk_tree tree = pdmk_tree_create(comm, params, n_src_local, local_sources.data(), local_charges.data(), nullptr,
                                      nullptr, n_trg_local, local_targets.data());
    double ft = omp_get_wtime();
    const double build_time = reduce_max(ft - st, comm);

    MPI_Barrier(comm);
    st = omp_get_wtime();
    pdmk_tree_eval(tree, pot_src.data(), with_grad ? grad_src.data() : nullptr, nullptr, pot_trg.data(),
                   with_grad ? grad_trg.data() : nullptr, nullptr);
    ft = omp_get_wtime();
    const double eval_time = reduce_max(ft - st, comm);

    pdmk_tree_destroy(tree);

    constexpr double inv_4pi = 1.0 / (4.0 * M_PI);
    for (auto &v : pot_src)
        v *= inv_4pi;
    for (auto &v : pot_trg)
        v *= inv_4pi;
    if (with_grad) {
        for (auto &v : grad_src)
            v *= inv_4pi;
        for (auto &v : grad_trg)
            v *= inv_4pi;
    }

    return {build_time, eval_time, build_time + eval_time};
}

ErrorMetrics compute_scalar_error(const std::vector<double> &local_values, const std::vector<double> &local_reference,
                                  int n_points, MPI_Comm comm) {
    double local_err2 = 0.0;
    double local_ref2 = 0.0;
    double local_max_rel = 0.0;

    for (int i = 0; i < n_points; ++i) {
        const double diff = local_values[i] - local_reference[i];
        const double ref = local_reference[i];
        local_err2 += diff * diff;
        local_ref2 += ref * ref;
        if (std::abs(ref) > 1e-30)
            local_max_rel = std::max(local_max_rel, std::abs(diff) / std::abs(ref));
    }

    double global_err2 = 0.0;
    double global_ref2 = 0.0;
    double global_max_rel = 0.0;
    MPI_Allreduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_ref2, &global_ref2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_max_rel, &global_max_rel, 1, MPI_DOUBLE, MPI_MAX, comm);

    ErrorMetrics metrics;
    metrics.rel_l2 = global_ref2 > 0.0 ? std::sqrt(global_err2 / global_ref2) : 0.0;
    metrics.max_rel = global_max_rel;
    return metrics;
}

ErrorMetrics compute_vector3_error(const std::vector<double> &local_values, const std::vector<double> &local_reference,
                                   int n_points, MPI_Comm comm) {
    double local_err2 = 0.0;
    double local_ref2 = 0.0;
    double local_max_rel = 0.0;

    for (int i = 0; i < n_points; ++i) {
        double diff2 = 0.0;
        double ref_norm2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double diff = local_values[3 * i + d] - local_reference[3 * i + d];
            const double ref = local_reference[3 * i + d];
            diff2 += diff * diff;
            ref_norm2 += ref * ref;
        }
        local_err2 += diff2;
        local_ref2 += ref_norm2;
        const double ref_norm = std::sqrt(ref_norm2);
        if (ref_norm > 1e-30)
            local_max_rel = std::max(local_max_rel, std::sqrt(diff2) / ref_norm);
    }

    double global_err2 = 0.0;
    double global_ref2 = 0.0;
    double global_max_rel = 0.0;
    MPI_Allreduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_ref2, &global_ref2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_max_rel, &global_max_rel, 1, MPI_DOUBLE, MPI_MAX, comm);

    ErrorMetrics metrics;
    metrics.rel_l2 = global_ref2 > 0.0 ? std::sqrt(global_err2 / global_ref2) : 0.0;
    metrics.max_rel = global_max_rel;
    return metrics;
}

int digits_from_eps(double eps) { return static_cast<int>(std::llround(-std::log10(eps))); }

fs::path default_csv_path(const Config &cfg, int mpi_ranks, int omp_threads) {
    std::ostringstream oss;
    oss << "bench_digits" << digits_from_eps(cfg.eps) << "_mpi" << mpi_ranks << "_omp" << omp_threads << ".csv";
    return fs::path(cfg.out_dir) / oss.str();
}

void write_csv_header(std::ofstream &ofs, const Config &cfg, const DatasetMeta &meta, int mpi_ranks, int omp_threads) {
    ofs << "# mode: benchmark\n";
    ofs << "# mpi_ranks: " << mpi_ranks << '\n';
    ofs << "# omp_threads_per_rank: " << omp_threads << '\n';
    ofs << "# n_src: " << cfg.n_src << '\n';
    ofs << "# n_trg: " << cfg.n_trg << '\n';
    ofs << "# eps: " << std::setprecision(17) << cfg.eps << '\n';
    ofs << "# reference_eps: " << std::setprecision(17) << meta.reference_eps << '\n';
    ofs << "# seed: " << cfg.seed << '\n';
    ofs << "# uniform: " << cfg.uniform << '\n';
    ofs << "# set_fixed_charges: " << cfg.set_fixed_charges << '\n';
    ofs << "# n_per_leaf: " << cfg.n_per_leaf << '\n';
    ofs << "# warmup_runs: " << cfg.warmup_runs << '\n';
    ofs << "# measured_runs: " << cfg.n_runs << '\n';
    ofs << "# reference_kernel_normalization: " << meta.reference_kernel_normalization << '\n';
    ofs << "# dmk_output_scaled_by: 1/(4*pi)\n";
    ofs << "run_idx,pot_build_time,pot_eval_time,pot_total_time,pot_eval_pts_per_sec,pot_total_pts_per_sec,"
           "potgrad_build_time,potgrad_eval_time,potgrad_total_time,potgrad_eval_pts_per_sec,potgrad_total_pts_per_sec,"
           "eval_overhead,total_overhead,pot_src_rel_l2,pot_trg_rel_l2,pot_src_max_rel,pot_trg_max_rel,"
           "grad_src_rel_l2,grad_trg_rel_l2,grad_src_max_rel,grad_trg_max_rel\n";
}

void print_usage(const char *argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --mode reference|benchmark   Execution mode\n"
        << "  --out-dir PATH               Output artifact directory\n"
        << "  --csv PATH                   Raw benchmark CSV output path\n"
        << "  --n-src N                    Number of sources\n"
        << "  --n-trg N                    Number of targets\n"
        << "  --eps VALUE                  DMK requested tolerance in benchmark mode\n"
        << "  --reference-eps VALUE        FMM3D reference tolerance in reference mode\n"
        << "  --n-runs N                   Number of measured runs in benchmark mode\n"
        << "  --warmup-runs N              Number of warmup runs in benchmark mode\n"
        << "  --n-per-leaf N               DMK leaf size\n"
        << "  --seed N                     Dataset seed\n"
        << "  --uniform                    Use uniform random point clouds\n"
        << "  --log-level N                DMK log level\n"
        << "  -h, --help                   Show this help\n";
}

Config parse_args(int argc, char **argv) {
    Config cfg;

    static option long_options[] = {
        {"mode", required_argument, nullptr, 1001},
        {"out-dir", required_argument, nullptr, 1002},
        {"csv", required_argument, nullptr, 1003},
        {"n-src", required_argument, nullptr, 1004},
        {"n-trg", required_argument, nullptr, 1005},
        {"eps", required_argument, nullptr, 1006},
        {"reference-eps", required_argument, nullptr, 1007},
        {"n-runs", required_argument, nullptr, 1008},
        {"warmup-runs", required_argument, nullptr, 1009},
        {"seed", required_argument, nullptr, 1010},
        {"uniform", no_argument, nullptr, 1011},
        {"n-per-leaf", required_argument, nullptr, 1012},
        {"log-level", required_argument, nullptr, 1013},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    while (true) {
        const int opt = getopt_long(argc, argv, "h", long_options, nullptr);
        if (opt == -1)
            break;

        switch (opt) {
        case 1001:
            cfg.mode = optarg;
            break;
        case 1002:
            cfg.out_dir = optarg;
            break;
        case 1003:
            cfg.csv_path = optarg;
            break;
        case 1004:
            cfg.n_src = static_cast<int>(std::atof(optarg));
            break;
        case 1005:
            cfg.n_trg = static_cast<int>(std::atof(optarg));
            break;
        case 1006:
            cfg.eps = std::atof(optarg);
            break;
        case 1007:
            cfg.reference_eps = std::atof(optarg);
            break;
        case 1008:
            cfg.n_runs = std::atoi(optarg);
            break;
        case 1009:
            cfg.warmup_runs = std::atoi(optarg);
            break;
        case 1010:
            cfg.seed = std::atol(optarg);
            break;
        case 1011:
            cfg.uniform = true;
            break;
        case 1012:
            cfg.n_per_leaf = std::atoi(optarg);
            break;
        case 1013:
            cfg.log_level = std::atoi(optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            std::exit(0);
        default:
            throw std::runtime_error("Unknown command line option");
        }
    }

    if (cfg.mode != "reference" && cfg.mode != "benchmark")
        throw std::runtime_error("Unsupported mode: " + cfg.mode);
    if (cfg.n_dim != 3)
        throw std::runtime_error("This benchmark only supports 3D Laplace");
    if (cfg.n_mfm != 1)
        throw std::runtime_error("This benchmark only supports n_mfm=1");
    if (cfg.n_src <= 0 || cfg.n_trg < 0)
        throw std::runtime_error("Invalid source or target count");
    if (cfg.n_runs <= 0 || cfg.warmup_runs < 0)
        throw std::runtime_error("Invalid run counts");
    return cfg;
}

void run_reference_mode(const Config &cfg, MPI_Comm comm) {
    int rank = 0, np = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    if (np != 1)
        throw std::runtime_error("Reference mode must be run with exactly one MPI rank");

    if (rank == 0) {
        std::cout << "Generating reference artifacts in " << cfg.out_dir << '\n';
        std::cout << "n_src=" << cfg.n_src << " n_trg=" << cfg.n_trg << " reference_eps=" << cfg.reference_eps << '\n';
        const auto artifacts = generate_reference_artifacts(cfg);
        write_reference_artifacts(cfg, artifacts);
        std::cout << "Wrote metadata.yaml, sources.bin, targets.bin, charges.bin, pot_src_ref.bin, pot_trg_ref.bin, "
                     "grad_src_ref.bin, grad_trg_ref.bin\n";
    }
}

void run_benchmark_mode(const Config &cfg, MPI_Comm comm) {
    int rank = 0, np = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    DatasetArtifacts global_artifacts;
    if (rank == 0)
        global_artifacts = load_reference_artifacts_root(cfg);

    const auto local_sources = scatter_vector(rank == 0 ? global_artifacts.sources : std::vector<double>{}, cfg.n_src, cfg.n_dim, comm);
    const auto local_charges = scatter_vector(rank == 0 ? global_artifacts.charges : std::vector<double>{}, cfg.n_src, cfg.n_mfm, comm);
    const auto local_targets = scatter_vector(rank == 0 ? global_artifacts.targets : std::vector<double>{}, cfg.n_trg, cfg.n_dim, comm);
    const auto local_pot_src_ref =
        scatter_vector(rank == 0 ? global_artifacts.pot_src_ref : std::vector<double>{}, cfg.n_src, 1, comm);
    const auto local_pot_trg_ref =
        scatter_vector(rank == 0 ? global_artifacts.pot_trg_ref : std::vector<double>{}, cfg.n_trg, 1, comm);
    const auto local_grad_src_ref =
        scatter_vector(rank == 0 ? global_artifacts.grad_src_ref : std::vector<double>{}, cfg.n_src, cfg.n_dim, comm);
    const auto local_grad_trg_ref =
        scatter_vector(rank == 0 ? global_artifacts.grad_trg_ref : std::vector<double>{}, cfg.n_trg, cfg.n_dim, comm);

    std::vector<double> pot_src_only, grad_src_unused, pot_trg_only, grad_trg_unused;
    std::vector<double> pot_src_pg, grad_src_pg, pot_trg_pg, grad_trg_pg;
    const int n_local_src = static_cast<int>(local_sources.size()) / cfg.n_dim;
    const int n_local_trg = static_cast<int>(local_targets.size()) / cfg.n_dim;
    const int omp_threads = get_omp_threads();
    const fs::path csv_path = cfg.csv_path.empty() ? default_csv_path(cfg, np, omp_threads) : fs::path(cfg.csv_path);

    if (rank == 0) {
        fs::create_directories(csv_path.parent_path());
        std::cout << "Benchmark mode with mpi_ranks=" << np << " omp_threads=" << omp_threads << " eps=" << cfg.eps
                  << " n_src=" << cfg.n_src << " n_trg=" << cfg.n_trg << " (pot-only baseline + pot+grad)\n";
    }

    for (int i = 0; i < cfg.warmup_runs; ++i)
        run_dmk_benchmark(cfg, false, local_sources, local_charges, local_targets, pot_src_only, grad_src_unused,
                          pot_trg_only, grad_trg_unused, comm);
    for (int i = 0; i < cfg.warmup_runs; ++i)
        run_dmk_benchmark(cfg, true, local_sources, local_charges, local_targets, pot_src_pg, grad_src_pg, pot_trg_pg,
                          grad_trg_pg, comm);

    std::ofstream ofs;
    if (rank == 0) {
        ofs.open(csv_path);
        if (!ofs)
            throw std::runtime_error("Failed to open " + csv_path.string() + " for writing benchmark CSV");
        write_csv_header(ofs, cfg, global_artifacts.meta, np, omp_threads);
    }

    for (int run = 0; run < cfg.n_runs; ++run) {
        const auto pot_timing = run_dmk_benchmark(cfg, false, local_sources, local_charges, local_targets, pot_src_only,
                                                  grad_src_unused, pot_trg_only, grad_trg_unused, comm);
        const auto potgrad_timing =
            run_dmk_benchmark(cfg, true, local_sources, local_charges, local_targets, pot_src_pg, grad_src_pg,
                              pot_trg_pg, grad_trg_pg, comm);
        const auto pot_src_err = compute_scalar_error(pot_src_pg, local_pot_src_ref, n_local_src, comm);
        const auto pot_trg_err = compute_scalar_error(pot_trg_pg, local_pot_trg_ref, n_local_trg, comm);
        const auto grad_src_err = compute_vector3_error(grad_src_pg, local_grad_src_ref, n_local_src, comm);
        const auto grad_trg_err = compute_vector3_error(grad_trg_pg, local_grad_trg_ref, n_local_trg, comm);
        const double n_total = static_cast<double>(cfg.n_src + cfg.n_trg);
        const double pot_eval_pts_per_sec = n_total / pot_timing.eval_time;
        const double pot_total_pts_per_sec = n_total / pot_timing.total_time;
        const double potgrad_eval_pts_per_sec = n_total / potgrad_timing.eval_time;
        const double potgrad_total_pts_per_sec = n_total / potgrad_timing.total_time;
        const double eval_overhead = potgrad_timing.eval_time / pot_timing.eval_time;
        const double total_overhead = potgrad_timing.total_time / pot_timing.total_time;

        if (rank == 0) {
            ofs << run << ',' << std::setprecision(17) << pot_timing.build_time << ',' << pot_timing.eval_time << ','
                << pot_timing.total_time << ',' << pot_eval_pts_per_sec << ',' << pot_total_pts_per_sec << ','
                << potgrad_timing.build_time << ',' << potgrad_timing.eval_time << ',' << potgrad_timing.total_time
                << ',' << potgrad_eval_pts_per_sec << ',' << potgrad_total_pts_per_sec << ',' << eval_overhead << ','
                << total_overhead << ',' << pot_src_err.rel_l2 << ',' << pot_trg_err.rel_l2 << ','
                << pot_src_err.max_rel << ',' << pot_trg_err.max_rel << ',' << grad_src_err.rel_l2 << ','
                << grad_trg_err.rel_l2 << ',' << grad_src_err.max_rel << ',' << grad_trg_err.max_rel << '\n';
            ofs.flush();
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        const auto cfg = parse_args(argc, argv);
        if (cfg.mode == "reference")
            run_reference_mode(cfg, MPI_COMM_WORLD);
        else
            run_benchmark_mode(cfg, MPI_COMM_WORLD);
    } catch (const std::exception &e) {
        if (rank == 0)
            std::cerr << "Error: " << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
