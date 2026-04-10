#include <dmk.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>

namespace {

struct AccuracyRow {
    int n_particles;
    int digits;
    double eps;
    std::string mode;
    double pot_src_l2;
    double pot_trg_l2;
    double grad_src_l2;
    double grad_trg_l2;
};

struct ProblemData {
    int n_particles;
    bool with_grad;
    int odim;
    std::vector<double> r_src;
    std::vector<double> r_trg;
    std::vector<double> charges;
    std::vector<double> rnormal;
    std::vector<double> dipstr;
};

struct ReferenceData {
    int odim;
    std::vector<double> src;
    std::vector<double> trg;
};

double safe_relative_l2(double err2, double ref2) {
    return ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
}

std::vector<std::complex<double>> build_rho(const std::vector<double> &r_src, const std::vector<double> &charges,
                                            double dk, int n_ewald) {
    const int n_src = static_cast<int>(charges.size());
    const int d = 2 * n_ewald + 1;
    std::vector<std::complex<double>> rho(d * d * d, {0.0, 0.0});

    for (int is = 0; is < n_src; ++is) {
        const std::complex<double> ex0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 0]));
        const std::complex<double> ey0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 1]));
        const std::complex<double> ez0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 2]));
        std::vector<std::complex<double>> ex(d), ey(d), ez(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            ex[a + n_ewald] = std::pow(ex0, a);
            ey[a + n_ewald] = std::pow(ey0, a);
            ez[a + n_ewald] = std::pow(ez0, a);
        }
        for (int ix = 0; ix < d; ++ix) {
            for (int iy = 0; iy < d; ++iy) {
                const auto t2 = charges[is] * ex[ix] * ey[iy];
                for (int iz = 0; iz < d; ++iz)
                    rho[ix * d * d + iy * d + iz] += t2 * ez[iz];
            }
        }
    }

    return rho;
}

void ewald_pot_grad(const double *r_eval, const std::vector<double> &r_src, const std::vector<double> &charges,
                    const std::vector<std::complex<double>> &rho, double alpha, int n_ewald, double &pot_out,
                    double *grad_out) {
    constexpr int n_dim = 3;
    const int n_src = static_cast<int>(charges.size());
    const double L = 1.0;
    const double V_box = L * L * L;
    const double dk = 2.0 * M_PI / L;
    const double r_c = 0.5 * L;
    const int d = 2 * n_ewald + 1;

    pot_out = 0.0;
    if (grad_out)
        grad_out[0] = grad_out[1] = grad_out[2] = 0.0;

    for (int is = 0; is < n_src; ++is) {
        for (int mx = -1; mx <= 1; ++mx) {
            for (int my = -1; my <= 1; ++my) {
                for (int mz = -1; mz <= 1; ++mz) {
                    const double dx = r_eval[0] - r_src[is * n_dim + 0] - mx * L;
                    const double dy = r_eval[1] - r_src[is * n_dim + 1] - my * L;
                    const double dz = r_eval[2] - r_src[is * n_dim + 2] - mz * L;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    const double r = std::sqrt(r2);
                    if (r > 1e-15 && r <= r_c) {
                        pot_out += charges[is] * std::erfc(alpha * r) / r;
                        if (grad_out) {
                            const double scale =
                                -charges[is] * (std::erfc(alpha * r) / (r * r2) +
                                                2.0 * alpha * std::exp(-alpha * alpha * r2) /
                                                    (std::sqrt(M_PI) * r2));
                            grad_out[0] += scale * dx;
                            grad_out[1] += scale * dy;
                            grad_out[2] += scale * dz;
                        }
                    }
                }
            }
        }
    }

    const std::complex<double> etx0 = std::exp(std::complex<double>(0.0, dk * r_eval[0]));
    const std::complex<double> ety0 = std::exp(std::complex<double>(0.0, dk * r_eval[1]));
    const std::complex<double> etz0 = std::exp(std::complex<double>(0.0, dk * r_eval[2]));
    std::vector<std::complex<double>> etx(d), ety(d), etz(d);
    for (int a = -n_ewald; a <= n_ewald; ++a) {
        etx[a + n_ewald] = std::pow(etx0, a);
        ety[a + n_ewald] = std::pow(ety0, a);
        etz[a + n_ewald] = std::pow(etz0, a);
    }

    double pot_long = 0.0;
    double grad_long[3] = {0.0, 0.0, 0.0};
    for (int nx = -n_ewald; nx <= n_ewald; ++nx) {
        for (int ny = -n_ewald; ny <= n_ewald; ++ny) {
            for (int nz = -n_ewald; nz <= n_ewald; ++nz) {
                if (nx == 0 && ny == 0 && nz == 0)
                    continue;
                const double kx = dk * nx;
                const double ky = dk * ny;
                const double kz = dk * nz;
                const double k2 = kx * kx + ky * ky + kz * kz;
                const double G = std::exp(-k2 / (4.0 * alpha * alpha)) / k2;
                const int ix = nx + n_ewald;
                const int iy = ny + n_ewald;
                const int iz = nz + n_ewald;
                const auto &rho_k = rho[ix * d * d + iy * d + iz];
                const auto eikr = etx[nx + n_ewald] * ety[ny + n_ewald] * etz[nz + n_ewald];
                const auto rho_eikr = rho_k * eikr;
                pot_long += G * std::real(rho_eikr);
                if (grad_out) {
                    const double im = -std::imag(rho_eikr);
                    grad_long[0] += G * kx * im;
                    grad_long[1] += G * ky * im;
                    grad_long[2] += G * kz * im;
                }
            }
        }
    }

    pot_out += (4.0 * M_PI) * pot_long / V_box;
    if (grad_out) {
        grad_out[0] += (4.0 * M_PI) * grad_long[0] / V_box;
        grad_out[1] += (4.0 * M_PI) * grad_long[1] / V_box;
        grad_out[2] += (4.0 * M_PI) * grad_long[2] / V_box;
    }
}

ProblemData make_problem(int n_particles, bool with_grad) {
    constexpr int n_dim = 3;

    ProblemData problem{};
    problem.n_particles = n_particles;
    problem.with_grad = with_grad;
    problem.odim = with_grad ? 1 + n_dim : 1;
    problem.r_src.resize(n_dim * n_particles);
    problem.r_trg.resize(n_dim * n_particles);
    problem.charges.resize(n_particles);
    problem.rnormal.assign(n_dim * n_particles, 0.0);
    problem.dipstr.assign(n_particles, 0.0);

    std::default_random_engine eng(42 + n_particles);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    for (double &v : problem.r_src)
        v = rng(eng);
    for (double &v : problem.r_trg)
        v = rng(eng);
    for (int i = 0; i < n_particles; ++i)
        problem.charges[i] = rng(eng) - 0.5;

    double sum = 0.0;
    for (double q : problem.charges)
        sum += q;
    for (double &q : problem.charges)
        q -= sum / n_particles;

    return problem;
}

ReferenceData build_reference(const ProblemData &problem) {
    constexpr int n_dim = 3;
    const double alpha = 10.0;
    const int n_ewald = 15;
    const double ewald_self_factor = 2.0 * alpha / std::sqrt(M_PI);

    ReferenceData ref{};
    ref.odim = problem.odim;
    ref.src.assign(problem.n_particles * problem.odim, 0.0);
    ref.trg.assign(problem.n_particles * problem.odim, 0.0);

    const auto rho = build_rho(problem.r_src, problem.charges, 2.0 * M_PI, n_ewald);

    for (int i = 0; i < problem.n_particles; ++i) {
        double pot_ref = 0.0;
        double grad_ref[3];
        ewald_pot_grad(&problem.r_src[i * n_dim], problem.r_src, problem.charges, rho, alpha, n_ewald, pot_ref,
                       problem.with_grad ? grad_ref : nullptr);
        pot_ref -= problem.charges[i] * ewald_self_factor;

        ref.src[i * problem.odim] = pot_ref;
        if (problem.with_grad) {
            for (int d = 0; d < n_dim; ++d)
                ref.src[i * problem.odim + 1 + d] = grad_ref[d];
        }
    }

    for (int i = 0; i < problem.n_particles; ++i) {
        double pot_ref = 0.0;
        double grad_ref[3];
        ewald_pot_grad(&problem.r_trg[i * n_dim], problem.r_src, problem.charges, rho, alpha, n_ewald, pot_ref,
                       problem.with_grad ? grad_ref : nullptr);

        ref.trg[i * problem.odim] = pot_ref;
        if (problem.with_grad) {
            for (int d = 0; d < n_dim; ++d)
                ref.trg[i * problem.odim + 1 + d] = grad_ref[d];
        }
    }

    return ref;
}

AccuracyRow run_case(const ProblemData &problem, const ReferenceData &reference, int digits, double eps) {
    constexpr int n_dim = 3;

    std::vector<double> pot_src(problem.n_particles * problem.odim);
    std::vector<double> pot_trg(problem.n_particles * problem.odim);

    pdmk_params params;
    params.eps = eps;
    params.n_dim = n_dim;
    params.n_per_leaf = 100;
    params.pgh_src = problem.with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
    params.pgh_trg = problem.with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = 1;
    params.log_level = 6;

    pdmk_tree tree =
        pdmk_tree_create(MPI_COMM_WORLD, params, problem.n_particles, problem.r_src.data(), problem.charges.data(),
                         problem.rnormal.data(), problem.dipstr.data(), problem.n_particles, problem.r_trg.data());
    pdmk_tree_eval(tree, pot_src.data(), pot_trg.data());
    pdmk_tree_destroy(tree);

    double err2_pot_src = 0.0, ref2_pot_src = 0.0;
    double err2_pot_trg = 0.0, ref2_pot_trg = 0.0;
    double err2_grad_src = 0.0, ref2_grad_src = 0.0;
    double err2_grad_trg = 0.0, ref2_grad_trg = 0.0;

    for (int i = 0; i < problem.n_particles; ++i) {
        const double pot_src_ref = reference.src[i * problem.odim];
        const double pot_trg_ref = reference.trg[i * problem.odim];
        err2_pot_src += std::pow(pot_src[i * problem.odim] - pot_src_ref, 2);
        ref2_pot_src += std::pow(pot_src_ref, 2);
        err2_pot_trg += std::pow(pot_trg[i * problem.odim] - pot_trg_ref, 2);
        ref2_pot_trg += std::pow(pot_trg_ref, 2);

        if (problem.with_grad) {
            for (int d = 0; d < n_dim; ++d) {
                const double grad_src_ref = reference.src[i * problem.odim + 1 + d];
                const double grad_trg_ref = reference.trg[i * problem.odim + 1 + d];
                err2_grad_src += std::pow(pot_src[i * problem.odim + 1 + d] - grad_src_ref, 2);
                ref2_grad_src += std::pow(grad_src_ref, 2);
                err2_grad_trg += std::pow(pot_trg[i * problem.odim + 1 + d] - grad_trg_ref, 2);
                ref2_grad_trg += std::pow(grad_trg_ref, 2);
            }
        }
    }

    AccuracyRow row{};
    row.n_particles = problem.n_particles;
    row.digits = digits;
    row.eps = eps;
    row.mode = problem.with_grad ? "pot+grad" : "pot";
    row.pot_src_l2 = safe_relative_l2(err2_pot_src, ref2_pot_src);
    row.pot_trg_l2 = safe_relative_l2(err2_pot_trg, ref2_pot_trg);
    row.grad_src_l2 = problem.with_grad ? safe_relative_l2(err2_grad_src, ref2_grad_src)
                                        : std::numeric_limits<double>::quiet_NaN();
    row.grad_trg_l2 = problem.with_grad ? safe_relative_l2(err2_grad_trg, ref2_grad_trg)
                                        : std::numeric_limits<double>::quiet_NaN();
    return row;
}

} // namespace

int main(int argc, char *argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int sizes[] = {1000, 2000, 5000, 10000};
    const int digits_list[] = {3, 6, 9, 12};
    const double eps_list[] = {1e-3, 1e-6, 1e-9, 1e-12};

    std::vector<AccuracyRow> rows;
    rows.reserve(2 * 4 * (sizeof(sizes) / sizeof(sizes[0])));

    for (int n : sizes) {
        for (bool with_grad : {false, true}) {
            const ProblemData problem = make_problem(n, with_grad);
            const ReferenceData reference = build_reference(problem);
            for (int i = 0; i < 4; ++i)
                rows.push_back(run_case(problem, reference, digits_list[i], eps_list[i]));
        }
    }

    if (rank == 0) {
        std::printf("N,digits,eps,mode,pot_src_l2,pot_trg_l2,grad_src_l2,grad_trg_l2\n");
        for (const auto &row : rows) {
            std::printf("%d,%d,%.0e,%s,%.12e,%.12e,%.12e,%.12e\n", row.n_particles, row.digits, row.eps,
                        row.mode.c_str(), row.pot_src_l2, row.pot_trg_l2, row.grad_src_l2, row.grad_trg_l2);
        }
    }

    MPI_Finalize();
    return 0;
}
