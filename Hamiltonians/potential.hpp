#pragma once
#include <Eigen/Dense>
#include <string>
#include "params/params.hpp"

class Potential {
public:
    Potential(const Params& p);

    Eigen::VectorXd gaussian_potential(double t) const;
    Eigen::VectorXd time_impulse(double t) const;
    Eigen::VectorXd sinus_wave(double t) const;
    Eigen::VectorXd ddf(double t) const;

    Eigen::VectorXd get_potential(double t, const std::string& mode) const;

    double vvR(double R) const;
    Eigen::MatrixXd build_coulomb_matrix() const;

private:
    const Params& p;
};
