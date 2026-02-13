#include "potential.hpp"
#include <cmath>
#include <fstream>
#include <complex>
using namespace std::complex_literals;

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
        // For spinless (implicit spin degeneracy) we used a factor 2.
        // When spin is treated explicitly (spin_on == true), drop that extra factor.
        const double spin_factor = p.spin_on ? 1.0 : 2.0;
        result(i) = spin_factor * p.e * p.E0 * osc * x;
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
                V(i, j) = p.coulomb_onsite_eV / p.au_eV;  // Hubbard U (a.u.), e.g. 5-10 eV for graphene
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

// ===========================
// Magnetic field helpers
// ===========================

double Potential::compute_Bz() const
{
    // In atomic units, e = 1 and m_e = 1, so
    //   Bz = E0 / (e / (2 m_e)) = 2 m_e E0 / e  →  2 * E0.
    const double e_au    = static_cast<double>(p.e);      // = 1
    const double m_e_au  = 1.0;                           // electron mass in a.u.
    return (p.E0) / (e_au/(2*m_e_au));                  // effectively 2 * E0
}

double Potential::calculate_phi(double xa, double xb,
                                double ya, double yb,
                                double Bz) const
{
    // Atomic-units version of the bachelor formula:
    //   phi = (e/hbar) * ∫ A·dl,  with A corresponding to Bz.
    // For our straight-line gauge, this reduces to
    //   phi = (e/hbar) * Bz * ( (ya + yb) * (xa - xb) / 2 ).
    const double e_au    = static_cast<double>(p.e);
    const double hbar_au = static_cast<double>(p.au_hbar); // = 1 in a.u.

    const double geom = ( (ya + yb) * (xa - xb) ) * 0.5;
    return (e_au / hbar_au) * Bz * geom;
}

std::vector<Potential::PeierlsPhase>
Potential::build_peierls_phases(double Bz) const
{
    std::vector<PeierlsPhase> phases;

    if (!p.two_dim)
        return phases; // only defined for 2D lattices

    const int N_sites = static_cast<int>(p.xl_2D.size());
    if (N_sites == 0)
        return phases;

    const double cutoff = p.a + 1e-3;       
    const double cutoff2 = cutoff * cutoff;

    for (int i = 0; i < N_sites; ++i) {
        const double xa = p.xl_2D[i][0];
        const double ya = p.xl_2D[i][1];
        for (int j = i + 1; j < N_sites; ++j) {
            const double xb = p.xl_2D[j][0];
            const double yb = p.xl_2D[j][1];

            const double dx = xa - xb;
            const double dy = ya - yb;
            const double r2 = dx*dx + dy*dy;

            // nearest-neighbour bond criterion
            if (r2 <= cutoff2) {
                PeierlsPhase pp;
                pp.i   = i;
                pp.j   = j;
                pp.phi = calculate_phi(xa, xb, ya, yb, Bz);
                phases.push_back(pp);
            }
        }
    }

    return phases;
}

void Potential::apply_peierls_to_spinful_hamiltonian(MatrixC& H_spin) const
{
    // Only meaningful for 2D graphene with explicit spin
    if (!(p.lattice == "graphene" || p.lattice == "Graphene"))
        return;
    if (!p.two_dim)
        return;

    const int N_mat = static_cast<int>(H_spin.rows());
    if (N_mat % 2 != 0)
        return; // not a 2N x 2N spinful Hamiltonian

    const int N_sites = N_mat / 2;

    const double Bz = compute_Bz();
    auto phases = build_peierls_phases(Bz);

    for (const auto& pp : phases) {
        int i = pp.i;
        int j = pp.j;
        if (i < 0 || j < 0 || i >= N_sites || j >= N_sites)
            continue;

        double phi = pp.phi;

        

        std::complex<double> phase_up = std::exp( 1i * phi );   // exp(+i phi)
        std::complex<double> phase_dn = std::exp(-1i * phi );   // exp(-i phi)


        // Up block
        std::complex<double> t_ij_up = H_spin(i, j);
        H_spin(i, j) = t_ij_up * phase_up;
        H_spin(j, i) = std::conj(H_spin(i, j));

        // Down block
        int iu = i + N_sites;
        int ju = j + N_sites;

        std::complex<double> t_ij_dn = H_spin(iu, ju);
        H_spin(iu, ju) = t_ij_dn * phase_dn;
        H_spin(ju, iu) = std::conj(H_spin(iu, ju));
    }
}

void Potential::export_peierls_phases(const std::filesystem::path& out_path) const
{
    // Only meaningful for 2D graphene
    if (!(p.lattice == "graphene" || p.lattice == "Graphene"))
        return;
    if (!p.two_dim)
        return;

    const double Bz = compute_Bz();
    auto phases = build_peierls_phases(Bz);

    std::ofstream fout(out_path);
    if (!fout)
        return;

    // Header with basic info
    fout << "# Peierls phases for graphene bonds\n";
    fout << "# Bz = " << Bz << " [a.u.]\n";
    fout << "# columns: i j  xa ya  xb yb  phi\n";

    for (const auto& pp : phases) {
        int i = pp.i;
        int j = pp.j;
        if (i < 0 || j < 0 || i >= static_cast<int>(p.xl_2D.size()) || j >= static_cast<int>(p.xl_2D.size()))
            continue;

        double xa = p.xl_2D[i][0];
        double ya = p.xl_2D[i][1];
        double xb = p.xl_2D[j][0];
        double yb = p.xl_2D[j][1];
        double phi = pp.phi;

        fout << i << " " << j << " "
             << xa << " " << ya << " "
             << xb << " " << yb << " "
             << phi << "\n";
    }
}
