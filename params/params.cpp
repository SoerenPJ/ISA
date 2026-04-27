#include "params.hpp"
#include <toml++/toml.h>
#include <cmath>
#include <unordered_set>
#include <set>
#include <utility>
#include <cstdint>

static std::string lower_ascii(std::string s)
{
    for (char& c : s)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    return s;
}
#include <array>
#include <vector>
#include <string>
#include <cmath>
#include <utility>

namespace {

    static std::array<std::array<double,2>, 6> generate_hexagon_start_at_zero(double bond_length)
    {
        double angles_deg[6] = {180.0, 240.0, 300.0, 0.0, 60.0, 120.0};
        std::array<std::array<double,2>, 6> hex{};
        for (int k = 0; k < 6; ++k){
            double angles_rad = angles_deg[k] * (M_PI / 180.0);
            hex[k] = {
                bond_length * std::cos(angles_rad),
                bond_length * std::sin(angles_rad)
            };
        }
        const std::array<double,2> shift = hex[0];
        for (auto& p : hex)
            p = { p[0] - shift[0], p[1] - shift[1] };
        return hex;
    }

    static std::vector<std::pair<int,int>> generate_positions(int size_x, int size_y,
        const std::string& formation_shape, const std::string& formation)
    {
        std::vector<std::pair<int,int>> pos;
        if (formation_shape == "rectangle") {
            for (int j = 0; j < size_y; ++j)
                for (int i = 0; i < size_x; ++i)
                    pos.emplace_back(i, j);
        }
        else if (formation_shape == "triangle") {
            if (formation == "zigzag") {
                for (int j = 0; j < size_y; ++j)
                    for (int i = 0; i < size_x; ++i)
                        if (i + j <= size_x - 1)
                            pos.emplace_back(i, j);
            }
            else if (formation == "armchair") {
                int cx = size_x / 2;
                for (int j = 0; j < size_y; ++j)
                    for (int i = 0; i < size_x; ++i)
                        if (std::abs(i - cx) <= j)
                            pos.emplace_back(i, j);
            }
        }
        else if (formation_shape == "hexagon") {
            const int s = size_x;
            for (int j = -(s - 1); j <= (s - 1); ++j) {
                for (int i = -(s - 1); i <= (s - 1); ++i) {
                    if (std::abs(i)     > s - 1) continue;
                    if (std::abs(j)     > s - 1) continue;
                    if (std::abs(i + j) > s - 1) continue;
                    if (formation == "armchair") {
                        const int n_at_max =
                            (std::abs(i)     == s - 1 ? 1 : 0) +
                            (std::abs(j)     == s - 1 ? 1 : 0) +
                            (std::abs(i + j) == s - 1 ? 1 : 0);
                        const int v3 = ((2 * i + j) % 3 + 3) % 3;
                        if (s % 2 == 1) {
                            if (n_at_max == 2) continue;
                            if (n_at_max == 1 && s >= 5 && v3 == 0) continue;
                        } else {
                            if (n_at_max == 1 && v3 == 1) continue;
                        }
                    }
                    pos.emplace_back(i, j);
                }
            }
        }
        return pos;
    }

    static std::array<double,2> add(const std::array<double,2>& a, const std::array<double,2>& b)
    { return {a[0]+b[0], a[1]+b[1]}; }

    static std::array<double,2> sub(const std::array<double,2>& a, const std::array<double,2>& b)
    { return {a[0]-b[0], a[1]-b[1]}; }

    static std::array<double,2> mul(const std::array<double,2>& a, double s)
    { return {a[0]*s, a[1]*s}; }

    static std::vector<std::array<double,2>> dedup_points_round_6(
        const std::vector<std::array<double,2>>& pts)
    {
        constexpr double scale = 1e6;
        struct Key {
            std::int64_t x, y;
            bool operator==(const Key& o) const noexcept { return x==o.x && y==o.y; }
            bool operator<(const Key& o)  const noexcept { return x!=o.x ? x<o.x : y<o.y; }
        };
        std::set<Key> seen;
        std::vector<std::array<double,2>> unique;
        unique.reserve(pts.size());
        for (const auto& p : pts) {
            const std::int64_t xq = static_cast<std::int64_t>(std::llround(p[0]*scale));
            const std::int64_t yq = static_cast<std::int64_t>(std::llround(p[1]*scale));
            if (seen.emplace(Key{xq,yq}).second)
                unique.push_back({static_cast<double>(xq)/scale,
                                  static_cast<double>(yq)/scale});
        }
        return unique;
    }

    static std::vector<std::array<double,2>> build_molecule_points(
        double bond_length, int size_x, int size_y,
        const std::string& formation,
        const std::string& formation_shape)
    {
        if (formation_shape == "bowtie") {
            auto pts = build_molecule_points(bond_length, size_x, size_y, formation, "triangle");
            if (formation == "zigzag") {
                double x_tip = -1e18;
                for (const auto& p : pts) x_tip = std::max(x_tip, p[0]);
                x_tip -= bond_length;
                const std::size_t n = pts.size();
                pts.reserve(2 * n);
                for (std::size_t k = 0; k < n; ++k)
                    pts.push_back({2.0 * x_tip - pts[k][0], pts[k][1]});
                return dedup_points_round_6(pts);
            }
            if (formation == "armchair") {
                double y_tip = -1e18;
                for (const auto& p : pts) y_tip = std::max(y_tip, p[1]);
                y_tip -= bond_length * std::sqrt(3.0);
                const std::size_t n = pts.size();
                pts.reserve(2 * n);
                for (std::size_t k = 0; k < n; ++k)
                    pts.push_back({pts[k][0], 2.0 * y_tip - pts[k][1]});
                return dedup_points_round_6(pts);
            }
            // fallback
            double x_tip = -1e18;
            for (const auto& p : pts) x_tip = std::max(x_tip, p[0]);
            const std::size_t n = pts.size();
            pts.reserve(2 * n);
            for (std::size_t k = 0; k < n; ++k)
                pts.push_back({2.0 * x_tip - pts[k][0], pts[k][1]});
            return dedup_points_round_6(pts);
        }

        const auto hex = generate_hexagon_start_at_zero(bond_length);

        if (formation == "zigzag" ||
            formation_shape == "rectangle" ||
            formation_shape == "hexagon")
        {
            const auto positions =
                generate_positions(size_x, size_y, formation_shape, formation);
            const auto hx = hex;
            const auto hx2_minus_hx0 = sub(hx[2], hx[0]);
            std::array<std::array<double,2>,6> temp_x;
            for (int k=0;k<6;++k) temp_x[k] = add(hx[k], hx2_minus_hx0);
            const auto move_x = add(hx2_minus_hx0, sub(hx[3], temp_x[5]));
            const auto hx1_minus_hx5 = sub(hx[1], hx[5]);
            std::array<std::array<double,2>,6> temp_y;
            for (int k=0;k<6;++k) temp_y[k] = add(hx[k], hx1_minus_hx5);
            const auto move_y = add(hx1_minus_hx5, sub(hx[2], temp_y[4]));
            std::vector<std::array<double,2>> all;
            all.reserve(positions.size() * 6ULL);
            for (const auto& [i, j] : positions) {
                const auto shift = add(mul(move_x, static_cast<double>(i)),
                                       mul(move_y, static_cast<double>(j)));
                for (const auto& v : hx) all.push_back(add(v, shift));
            }
            return dedup_points_round_6(all);
        }

        if (formation == "armchair") {
            const auto hx = hex;
            const auto hx1_minus_hx5 = sub(hx[1], hx[5]);
            std::array<std::array<double,2>,6> temp;
            for (int k=0;k<6;++k) temp[k] = add(hx[k], hx1_minus_hx5);
            const auto move_down  = add(hx1_minus_hx5, sub(hx[2], temp[4]));
            const auto move_left  = sub(hx[0], hx[4]);
            const auto move_right = sub(hx[2], hx[0]);
            struct Node { std::array<double,2> pos; int level; };
            std::vector<Node> queue;
            queue.push_back({{0.0, 0.0}, 0});
            std::unordered_set<std::int64_t> seen;
            auto hash_center = [](const std::array<double,2>& p) {
                constexpr double s = 1e6;
                const std::int64_t x = static_cast<std::int64_t>(std::llround(p[0]*s));
                const std::int64_t y = static_cast<std::int64_t>(std::llround(p[1]*s));
                return (x << 32) ^ y;
            };
            std::vector<std::array<double,2>> all;
            while (!queue.empty()) {
                const auto node = queue.front(); queue.erase(queue.begin());
                if (node.level >= size_y) continue;
                std::array<double,2> center{0.0, 0.0};
                for (const auto& v : hx) {
                    center[0] += v[0]+node.pos[0]; center[1] += v[1]+node.pos[1];
                }
                center[0] /= 6.0; center[1] /= 6.0;
                const auto key = hash_center(center);
                if (seen.count(key)) continue;
                seen.insert(key);
                for (const auto& v : hx) all.push_back(add(v, node.pos));
                const int next = node.level + 1;
                if (next < size_y) {
                    queue.push_back({ add(node.pos, move_down), next });
                    queue.push_back({ add(add(node.pos, move_down), move_left), next });
                    queue.push_back({ add(add(node.pos, move_down), move_right), next });
                }
            }
            return dedup_points_round_6(all);
        }
        return {};
    }

} // namespace


Params::Params()
{
    au_eV = 27.2113834;
    au_nm = 0.05291772083;
    au_s  = 2.418884326502e-17;
    au_fs = au_s * 1e15;
    au_c  = 137.03599971;
    au_kg = 9.10938291e-31;
    au_J  = 4.3597447222060e-18;
    au_m  = 5.29177210544e-11;
    au_kB = 8.617e-5 / au_eV;
    au_I  = 3.50944506e16;
    alpha = 1.0 / au_c;
    au_hbar = 1;
    e = 1;
    au_w  = 4.1341373336493e16;
    au_me = 9.10938291e-31;
}

void Params::load_from_toml(const std::string& filename)
{
    auto tbl = toml::parse_file(filename);
    N = tbl["system"]["N"].value_or(50);
    a = tbl["system"]["lattice_const"].value_or(0.1421) / au_nm;
    lattice = lower_ascii(tbl["system"]["lattice"].value_or(std::string("ssh")));
    two_dim = tbl["system"]["two_dim"].value_or(false);
    formation = lower_ascii(tbl["system"]["formation"].value_or(std::string("zigzag")));
    if (formation != "zigzag" && formation != "armchair") formation = "zigzag";
    formation_shape = lower_ascii(tbl["system"]["formation_shape"].value_or("triangle"));
    size_x = tbl["system"]["size_x"].value_or(2);
    size_y = tbl["system"]["size_y"].value_or(2);
    rotation_angle_deg = tbl["system"]["rotation_angle_deg"].value_or(0.0);
    t1    = tbl["hamiltonian"]["t1"].value_or(-2.8) / au_eV;
    t2    = tbl["hamiltonian"]["t2"].value_or(-2.8) / au_eV;
    mu    = tbl["hamiltonian"]["mu"].value_or(0.0)  / au_eV;
    gamma = tbl["hamiltonian"]["gamma"].value_or(0.01) / au_eV;
    spin_on = tbl["hamiltonian"]["spin_on"].value_or(false);
    dt               = tbl["simulation"]["dt"].value_or(0.2);
    max_internal_dt  = tbl["simulation"]["max_internal_dt"].value_or(0.1) / au_fs;
    t_end            = tbl["simulation"]["t_max"].value_or(500.0) / au_fs;
    t0               = tbl["simulation"]["t0"].value_or(0.0) / au_fs;
    a_tol            = tbl["simulation"]["a_tol"].value_or(1e-10);
    r_tol            = tbl["simulation"]["r_tol"].value_or(1e-12);
    use_strict_solver        = tbl["simulation"]["use_strict_solver"].value_or(false);
    Intensity   = tbl["field"]["intensity"].value_or(1e13);
    field_mode  = lower_ascii(tbl["field"]["mode"].value_or(std::string("time_impulse")));
    field_phase = tbl["field"]["phase"].value_or(0.0);
    au_omega     = tbl["field"]["omega"].value_or(0.2) / au_eV;
    au_omega_ddf = tbl["field"]["ddf_omega"].value_or(0.1) / au_eV;
    t_shift     = tbl["field"]["t_shift"].value_or(200.0) / au_fs;
    sigma_gaus  = tbl["field"]["sigma_gaus"].value_or(60.0) / au_fs;
    sigma_ddf   = tbl["field"]["sigma_ddf"].value_or(0.01) / au_fs;
    B_ext       = tbl["field"]["B_ext"].value_or(false);
    omega_cut_off  = tbl["analysis"]["omega_cut_off"].value_or(6.0) / au_eV;
    fourier_dt_fs  = tbl["analysis"]["fourier_dt_fs"].value_or(0.005);
    run_sigma_ext  = tbl["analysis"]["run_sigma_ext"].value_or(false);
    run_dipole_acc = tbl["analysis"]["run_dipole_acc"].value_or(false);
    T                 = tbl["thermo"]["T"].value_or(0.001);
    use_charge_doping = tbl["thermo"]["use_charge_doping"].value_or(false);
    Q_doping          = tbl["thermo"]["Q_doping"].value_or(0.0);
    coulomb_on            = tbl["features"]["coulomb"].value_or(true);
    coulomb_onsite_eV     = tbl["features"]["coulomb_onsite_eV"].value_or(5.0);
    self_consistent_phase = tbl["features"]["self_consistent_phase"].value_or(true);
    zeeman_external       = tbl["features"]["zeeman_external"].value_or(true);
    zeeman_induced        = tbl["features"]["zeeman_induced"].value_or(true);
}

void Params::finalize()
{
    au_mu_0 = 4.0 * M_PI / (au_c * au_c);
    const double intensity_au = (Intensity / au_kg) * (au_s * au_s * au_s);
    E0 = std::sqrt((2.0 * M_PI * intensity_au) / au_c);
    build_lattice();
    if (two_dim && !xl_2D.empty()) {
        if (lattice == "pentalene") {
            double xmin=xl_2D[0][0], xmax=xl_2D[0][0], ymin=xl_2D[0][1], ymax=xl_2D[0][1];
            for (const auto& r : xl_2D) {
                if (r[0]<xmin) xmin=r[0]; if (r[0]>xmax) xmax=r[0];
                if (r[1]<ymin) ymin=r[1]; if (r[1]>ymax) ymax=r[1];
            }
            area_2d = std::max((xmax-xmin+2.0*a)*(ymax-ymin+2.0*a), 1.0);
        } else {
            const double hex_area = (3.0*std::sqrt(3.0)*a*a)*0.5;
            area_2d = hex_area * static_cast<double>(std::max(1, size_x*size_y));
        }
    } else { area_2d = 1.0; }
}

void Params::build_lattice()
{
    xl_1D.clear(); xl_2D.clear();
    if (lattice == "ssh" || lattice == "chain") {
        xl_1D.reserve(N);
        for (int i=0;i<N;++i) xl_1D.push_back(a*i);
        return;
    }
    if (lattice == "graphene") {
        const auto pts = build_molecule_points(a, size_x, size_y, formation, formation_shape);
        xl_2D.assign(pts.begin(), pts.end());
        if (std::abs(rotation_angle_deg) > 0.0) {
            const double theta = rotation_angle_deg*(M_PI/180.0);
            const double c=std::cos(theta), s=std::sin(theta);
            for (auto& r : xl_2D) {
                const double x=r[0], y=r[1];
                r[0]=c*x-s*y; r[1]=s*x+c*y;
            }
        }
        if (!xl_2D.empty()) {
            double cx=0.0, cy=0.0;
            for (const auto& r : xl_2D) { cx+=r[0]; cy+=r[1]; }
            cx /= static_cast<double>(xl_2D.size());
            cy /= static_cast<double>(xl_2D.size());
            for (auto& r : xl_2D) { r[0]-=cx; r[1]-=cy; }
        }
        N = static_cast<int>(xl_2D.size());
        two_dim = true;
        xl_1D.reserve(static_cast<size_t>(N));
        for (const auto& r : xl_2D) xl_1D.push_back(r[0]);
        return;
    }
    lattice = "ssh";
    xl_1D.reserve(N);
    for (int i=0;i<N;++i) xl_1D.push_back(a*i);
}