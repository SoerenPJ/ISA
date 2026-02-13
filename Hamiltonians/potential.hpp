#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <complex>
#include <filesystem>
#include "params/params.hpp"

// Match the global MatrixC type used elsewhere (RowMajor complex matrix)
using MatrixC = Eigen::Matrix<
    std::complex<double>,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor
>;

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

    // ===========================
    // Magnetic field helpers
    // ===========================

    // Compute static Bz corresponding to the driving field amplitude E0
    // using the same convention as in the bachelor code:
    //   Bz = E0 / (e / (2 * m_e))
    double compute_Bz() const;

    // Peierls phase for a bond between two 2D points (xa,ya) -> (xb,yb)
    //   phi = (e/hbar) * ∫ A·dl  with A corresponding to Bz.
    double calculate_phi(double xa, double xb,
                         double ya, double yb,
                         double Bz) const;

    struct PeierlsPhase {
        int i;      // site index a
        int j;      // site index b
        double phi; // phase for bond (i,j)
    };

    // Build Peierls phases for all nearest-neighbour bonds in a 2D lattice
    // (graphene etc.). Uses p.xl_2D and p.a as in the bachelor code.
    std::vector<PeierlsPhase> build_peierls_phases(double Bz) const;

    // Bond list (i,j) for nearest neighbours; used for self-consistent induced phase.
    using Bond = std::pair<int, int>;
    std::vector<Bond> get_bonds() const;

    // Apply external magnetic field via Peierls phases to a spinful Hamiltonian H_spin.
    // H_spin must already be 2*N_sites x 2*N_sites (after spin_tonian),
    // and only the graphene/2D case is modified.
    void apply_peierls_to_spinful_hamiltonian(MatrixC& H_spin) const;

    // Export Peierls phases to a text file for Python plotting.
    // Each line: i j  xa ya  xb yb  phi
    void export_peierls_phases(const std::filesystem::path& out_path) const;

private:
    const Params& p;
};
