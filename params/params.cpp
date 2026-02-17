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

        // make the angles to rad
        for (int k = 0; k < 6; ++k){
            double angles_rad = angles_deg[k] * (M_PI / 180.0);
            hex[k] = {
                bond_length * std::cos(angles_rad),
                bond_length * std::sin(angles_rad)
            };
        }

        // shifting the hexagon so the first point is at (0,0)
        const std::array<double,2> shift = hex[0];
        for (auto& p : hex)
            p = { p[0] - shift[0], p[1] - shift[1] };
        return hex;
    }/*
    elif formation_shape == "triangle":
        if formation == "zigzag":
            # right-angled triangle (aligned left-top)
            positions = [(i, j) for j in range(size_y) for i in range(size_x) if i + j <= size_x - 1]
*/
static std::vector<std::pair<int,int>>generate_positions(int size_x, int size_y, const std::string& formation_shape,
const std::string& formation)
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

    return pos;
}


    static std::array<double,2> add(const std::array<double,2>& a, const std::array<double,2>& b)
    {
        return {a[0] + b[0], a[1] + b[1]};
    }

    static std::array<double,2> sub(const std::array<double,2>& a, const std::array<double,2>& b)
    {
        return {a[0] - b[0], a[1] - b[1]};
    }

    static std::array<double,2> mul(const std::array<double,2>& a, double s)
    {
        return {a[0] * s, a[1] * s};
    }
   
    
    static std::vector<std::array<double,2>> dedup_points_round_6(const std::vector<std::array<double,2>>& pts)
    {
        constexpr double scale = 1e6;
        struct Key {
            std::int64_t x;
            std::int64_t y;
            bool operator==(const Key& o) const noexcept { return x == o.x && y == o.y; }
            bool operator<(const Key& o) const noexcept {
                if (x != o.x) return x < o.x;
                return y < o.y;
            }
        };

        std::set<Key> seen;
        std::vector<std::array<double,2>> unique;
        unique.reserve(pts.size());

        for (const auto& p : pts) {
            const std::int64_t xq = static_cast<std::int64_t>(std::llround(p[0] * scale));
            const std::int64_t yq = static_cast<std::int64_t>(std::llround(p[1] * scale));
            if (seen.emplace(Key{xq, yq}).second)
                unique.push_back({static_cast<double>(xq) / scale, static_cast<double>(yq) / scale});
        }
        return unique;
    }

    static std::vector<std::array<double,2>> build_molecule_points(double bond_length,
                      int size_x,
                      int size_y,
                      const std::string& formation,
                      const std::string& formation_shape)

{
    const auto hex = generate_hexagon_start_at_zero(bond_length);
    std::vector<std::array<double,2>> all;

    // ============================================================
    // ZIGZAG  (grid-based, affine placement)
    // ============================================================
    if (formation == "zigzag") {

        const auto positions =
            generate_positions(size_x, size_y, formation_shape, "zigzag");

        // temp_x = hx + (hx[2] - hx[0])
        // move_x = (hx[2] - hx[0]) + (hx[3] - temp_x[5])
        const auto hx = hex;

        const auto hx2_minus_hx0 = sub(hx[2], hx[0]);
        std::array<std::array<double,2>,6> temp_x;
        for (int k = 0; k < 6; ++k)
            temp_x[k] = add(hx[k], hx2_minus_hx0);

        const auto move_x =
            add(hx2_minus_hx0, sub(hx[3], temp_x[5]));

        // temp_y = hx + (hx[1] - hx[5])
        // move_y = (hx[1] - hx[5]) + (hx[2] - temp_y[4])
        const auto hx1_minus_hx5 = sub(hx[1], hx[5]);
        std::array<std::array<double,2>,6> temp_y;
        for (int k = 0; k < 6; ++k)
            temp_y[k] = add(hx[k], hx1_minus_hx5);

        const auto move_y =
            add(hx1_minus_hx5, sub(hx[2], temp_y[4]));

        // ---- Place hexagons ----
        all.reserve(positions.size() * 6ULL);

        for (const auto& [i, j] : positions) {
            const auto shift =
                add(mul(move_x, static_cast<double>(i)),
                    mul(move_y, static_cast<double>(j)));

            for (const auto& v : hx)
                all.push_back(add(v, shift));
        }

        return dedup_points_round_6(all);
    }

    // ============================================================
    if (formation == "armchair") {

        const auto hx = hex;

        // move_down = hx[1] - hx[5] + hx[2] - (hx + (hx[1] - hx[5]))[4]
        const auto hx1_minus_hx5 = sub(hx[1], hx[5]);
        std::array<std::array<double,2>,6> temp;
        for (int k = 0; k < 6; ++k)
            temp[k] = add(hx[k], hx1_minus_hx5);

        const auto move_down =
            add(hx1_minus_hx5, sub(hx[2], temp[4]));

        const auto move_left  = sub(hx[0], hx[4]);
        const auto move_right = sub(hx[2], hx[0]);

        // ---- BFS structures ----
        struct Node {
            std::array<double,2> pos;
            int level;
        };

        std::vector<Node> queue;
        queue.push_back({{0.0, 0.0}, 0});

        std::unordered_set<std::int64_t> seen;

        auto hash_center = [](const std::array<double,2>& p) {
            constexpr double s = 1e6;
            const std::int64_t x =
                static_cast<std::int64_t>(std::llround(p[0] * s));
            const std::int64_t y =
                static_cast<std::int64_t>(std::llround(p[1] * s));
            return (x << 32) ^ y;
        };

        while (!queue.empty()) {
            const auto node = queue.front();
            queue.erase(queue.begin());

            if (node.level >= size_y)
                continue;

            // Compute center of hexagon
            std::array<double,2> center{0.0, 0.0};
            for (const auto& v : hx) {
                center[0] += v[0] + node.pos[0];
                center[1] += v[1] + node.pos[1];
            }
            center[0] /= 6.0;
            center[1] /= 6.0;

            const auto key = hash_center(center);
            if (seen.count(key))
                continue;

            seen.insert(key);

            // Add hexagon
            for (const auto& v : hx)
                all.push_back(add(v, node.pos));

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
//name space end 
}


// ======================
// Constructor: constants only
// ======================
Params::Params()
{
    // Keep these consistent with the project's reference Python constants.
    au_eV = 27.2113834;
    au_nm = 0.05291772083;
    au_s  = 2.418884326502e-17;

    // NOTE: In this codebase, `au_fs` means "femtoseconds per atomic unit of time".
    // (If you prefer "atomic units per femtosecond", that's 1/au_fs.)
    au_fs = au_s * 1e15; // ~0.0241888 fs

    au_c  = 137.03599971;

    au_kg = 9.10938291e-31;
    au_J  = 4.3597447222060e-18;      // 1 Hartree in Joule
    au_m  = 5.29177210544e-11;        // Bohr radius in meter

    // kB in atomic units (Hartree/K). Same as (8.617e-5 eV/K) / au_eV.
    au_kB = 8.617e-5 / au_eV;

    // Keep as a reference constant (not used for the legacy intensity convention)
    au_I  = 3.50944506e16;

    // Derived / convenience
    alpha = 1.0 / au_c;
    au_hbar = 1;
    e = 1;

    // Keep for compatibility if used elsewhere
    au_w  = 4.1341373336493e16; // Eh / ħ in s^-1 (approx)
    au_me = 9.10938291e-31;
}

// ======================
// Load TOML
// ======================
void Params::load_from_toml(const std::string& filename)
{
    auto tbl = toml::parse_file(filename);

    // ---- system ----
    N = tbl["system"]["N"].value_or(50);
    a = tbl["system"]["lattice_const"].value_or(0.142) / au_nm;
    
    // default to SSH/1D if not provided
    lattice = lower_ascii(tbl["system"]["lattice"].value_or(std::string("ssh")));
    two_dim = tbl["system"]["two_dim"].value_or(false);
    formation = lower_ascii(tbl["system"]["formation"].value_or(std::string("zigzag")));
    if (formation != "zigzag" && formation != "armchair")
        formation = "zigzag";

    formation_shape = lower_ascii(tbl["system"]["formation_shape"].value_or("triangle"));

    size_x = tbl["system"]["size_x"].value_or(2);
    size_y = tbl["system"]["size_y"].value_or(2);

    // ---- hamiltonian ----
    t1 = tbl["hamiltonian"]["t1"].value_or(-2.8) / au_eV;
    t2 = tbl["hamiltonian"]["t2"].value_or(-2.8) / au_eV;
    mu = tbl["hamiltonian"]["mu"].value_or(0.0) / au_eV;
    gamma = tbl["hamiltonian"]["gamma"].value_or(0.01) / au_eV;
    // read optional spin flag from [hamiltonian] table
    spin_on = tbl["hamiltonian"]["spin_on"].value_or(false);

    // ---- simulation ----
    
    dt    = tbl["simulation"]["dt"].value_or(0.2); // do not change, this is the way 
    t_end = tbl["simulation"]["t_max"].value_or(500.0) / au_fs;
    t0    = tbl["simulation"]["t0"].value_or(0.0) / au_fs;
    a_tol = tbl["simulation"]["a_tol"].value_or(1e-10);
    r_tol = tbl["simulation"]["r_tol"].value_or(1e-12);
    // ---- solver ----
    use_strict_solver = tbl["simulation"]["use_strict_solver"].value_or(false);

    
    Intensity   = tbl["field"]["intensity"].value_or(1e13);

    field_mode  = lower_ascii(tbl["field"]["mode"].value_or(std::string("time_impulse")));
    field_phase = tbl["field"]["phase"].value_or(0.0);

    au_omega    = tbl["field"]["omega"].value_or(0.2) / au_eV;
    au_omega_ddf = tbl["field"]["ddf_omega"].value_or(0.1) / au_eV;
    t_shift     = tbl["field"]["t_shift"].value_or(200.0) / au_fs;
    sigma_gaus  = tbl["field"]["sigma_gaus"].value_or(60.0) /au_fs;
    sigma_ddf   = tbl["field"]["sigma_ddf"].value_or(0.01)/ au_fs;
    B_ext       = tbl["field"]["B_ext"].value_or(false); 
    omega_cut_off = tbl["analysis"]["omega_cut_off"].value_or(6.0) /au_eV;
    fourier_dt_fs = tbl["analysis"]["fourier_dt_fs"].value_or(0.005);

    // ---- thermo ----
    T = tbl["thermo"]["T"].value_or(300);

    // ---- features ----
    coulomb_on = tbl["features"]["coulomb"].value_or(true);
    coulomb_onsite_eV = tbl["features"]["coulomb_onsite_eV"].value_or(10.0);
    self_consistent_phase = tbl["features"]["self_consistent_phase"].value_or(true);
    zeeman_external = tbl["features"]["zeeman_external"].value_or(true);
    zeeman_induced = tbl["features"]["zeeman_induced"].value_or(true);
}

// ======================
// Derived quantities
// ======================
void Params::finalize()
{
    au_mu_0 = 4.0 * M_PI / (au_c * au_c);  // vacuum permeability in a.u.
    const double intensity_au = (Intensity / au_kg) * (au_s * au_s * au_s);
    E0 = std::sqrt((2.0 * M_PI * intensity_au) / au_c);
    build_lattice();

    // Effective 2D area for induced vector potential (graphene: unit-cell area * hexagon count)
    if (two_dim && !xl_2D.empty()) {
        const double hex_area = (3.0 * std::sqrt(3.0) * a * a) * 0.5;
        int n_hex = std::max(1, size_x * size_y);
        area_2d = hex_area * static_cast<double>(n_hex);
    } else {
        area_2d = 1.0;
    }
}

// ======================
// Geometry
// ======================
void Params::build_lattice()
{
    xl_1D.clear();
    xl_2D.clear();

    // Treat "chain" as SSH/1D for backwards compatibility
    if (lattice == "ssh" || lattice == "chain") {
        xl_1D.reserve(N);
        for (int i = 0; i < N; ++i)
            xl_1D.push_back(a * i);
        return;
    }

    if (lattice == "graphene") {
        const auto pts = build_molecule_points(a, size_x, size_y, formation, formation_shape);
        xl_2D.assign(pts.begin(), pts.end());

        N = static_cast<int>(xl_2D.size());
        two_dim = true;

        xl_1D.reserve(static_cast<size_t>(N));
        for (const auto& r : xl_2D)
            xl_1D.push_back(r[0]);

        return;
    }



    // Fallback: always build a 1D chain so we never segfault in potential/coulomb.
    lattice = "ssh";
    xl_1D.reserve(N);
    for (int i = 0; i < N; ++i)
        xl_1D.push_back(a * i);
}
