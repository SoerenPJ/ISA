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
    double osc = std::sin(p.au_omega * dt + p.field_phase);

    for (int i = 0; i < p.N; ++i) {
        double x = p.two_dim ? p.xl_2D[i][0] : p.xl_1D[i];
        result(i) = p.e * p.E0 * osc * x * env;
    }
    return result;
}

// ---------------- Sinus wave ----------------
Eigen::VectorXd Potential::sinus_wave(double t) const {
    Eigen::VectorXd result(p.N);

    double osc = std::sin(p.au_omega * t + p.field_phase);

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
        val = p.E0 * std::cos(p.au_omega_ddf * t + p.field_phase);

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

double Potential::vvR(double R) const{
    if (R>19.75)
        return 1.0/R;
    
    double R2 = R * R; 
    double R3 = R2 * R; 
    double R4 = R2 * R2;
    double R6 = R3 * R3; 
    
    return 1.01663 / (6.37713 + R)
         + 5.01882 / (71.7417 + R2)
         + 0.0195388 / (0.464504 + R3)
         + 117.821  / (56.9684  + R4)
         - 420.292  / (1429.49  + R3 * R2)
         - 435.634  / (157.826  + R6)
         + 127.506  / (170.114  + R6 * R)
         + 771.863  / (1056.55  + R6 * R2)
         - 8166.2   / (55034.0  + R6 * R3)
         - 48706.2  / (1.23854e6 + R6 * R4); 
    
}

Eigen::MatrixXd Potential::build_coulomb_matrix() const {
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(p.N, p.N);


    for (int i = 0; i < p.N; i++) {
        for (int j = 0; j < p.N; j++) {

            if (i == j) {
                V(i, j) = 1.0;   // DEFINE U HERE, not vvR(0)
                continue;
            }

            double R;
            if (p.two_dim) {
                double dx = p.xl_2D[i][0] - p.xl_2D[j][0];
                double dy = p.xl_2D[i][1] - p.xl_2D[j][1];
                R = std::sqrt(dx*dx + dy*dy);
            } else {
                R = std::abs(p.xl_1D[i] - p.xl_1D[j]);
            }

            V(i, j) = vvR(R);
        }
    }

    return V;
}
