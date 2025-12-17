#pragma once
#include <eigen3/Eigen/Dense>
# include <string>
#include "params/params.hpp"

using namespace std;

class Potential {
public:
    Potential(const Params& p);

    Eigen::VectorXd gaussian_potential(double t) const;
    Eigen::VectorXd time_impulse(double t) const;
    Eigen::VectorXd sinus_wave(double t) const;
    Eigen::VectorXd ddf(double t) const;

    // generic dispatcher
    Eigen::VectorXd get_potential(double t, const string& mode) const;
    
    double vvR(double R) const;
    Eigen::MatrixXd build_coulomb_matrix() const;
private:
    const Params& p;
    
};