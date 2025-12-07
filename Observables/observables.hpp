#pragma once
#include <Eigen/Dense>

double compute_dipole_moment(
    const Eigen::MatrixXcd &rho_t,
    const Eigen::MatrixXcd &rho0,
    const Eigen::VectorXd  &xl, int e
);
