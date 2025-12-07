#include "potential.hpp"
#include <cmath>

Potential::Potential(const Params& p_) : p(p_) {}

// ---------------- Gaussian envelope ----------------
Eigen::VectorXd Potential::gaussian_potential(double t) const {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(p.N);

    double dt = t - p.t_shift;
    double sg = p.sigma_gaus;
    double env = std::exp(-(dt*dt) / (sg*sg));

    result.setConstant(env);
    return result;
}

// ---------------- Time impulse ----------------
Eigen::VectorXd Potential::time_impulse(double t) const {
    Eigen::VectorXd result(p.N);

    double dt = t - p.t_shift;
    double env = std::exp(-(dt*dt) / ((p.sigma_gaus)*(p.sigma_gaus)));
    double osc = std::sin(p.au_omega * dt);

    for (int i = 0; i < p.N; ++i) {
        double x = p.two_dim ? p.xl_2D[i][0] : p.xl_1D[i];
        result(i) = p.e * p.E0 * osc * x * env;
    }
    return result;
}

// ---------------- Sinus wave ----------------
Eigen::VectorXd Potential::sinus_wave(double t) const {
    Eigen::VectorXd result(p.N);

    double osc = std::sin(p.au_omega * t);

    for (int i = 0; i < p.N; ++i) {
        double x = p.two_dim ? p.xl_2D[i][0] : p.xl_1D[i];
        result(i) = 2.0 * p.e * p.E0 * osc * x;
    }
    return result;
}

// ---------------- DDF ----------------
Eigen::VectorXd Potential::ddf(double t) const {
    Eigen::VectorXd result(p.N);

    double cutoff = p.sigma_ddf;

    double val = 0.0;
    if (t < cutoff)
        val = p.E0 * std::cos((0.1/27.2113834) * t);

    for (int i = 0; i < p.N; ++i) {
        double x = p.two_dim ? p.xl_2D[i][0] : p.xl_1D[i];
        result(i) = val * x;
    }
    return result;
}

// ---------------- Dispatcher ----------------
Eigen::VectorXd Potential::get_potential(double t, const std::string& mode) const {
    if (mode == "time_impulse")
        return time_impulse(t);
    else if (mode == "sinus")
        return sinus_wave(t);
    else if (mode == "ddf")
        return ddf(t);
    else
        throw std::runtime_error("Unknown potential mode: " + mode);
}
