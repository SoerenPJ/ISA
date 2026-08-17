#!/bin/bash
# ohno_U_sweep.sh — scan the Ohno ONSITE U_0 in multiples of the hopping t.
#
# The Ohno kernel is
#
#     V(r_ij) = e^2 / sqrt( (e^2/U_0)^2 + r_ij^2 )
#
# so U_0 = V(0) is the onsite value AND, through the softening length e^2/U_0,
# the whole short-range shape of the Coulomb matrix. The long-range tail is
# e^2/r for EVERY U_0 — raising U_0 shrinks the softening length, it does not
# rescale the potential. Sweeping U_0 therefore probes how the interaction
# strength at and near a site drives the spectrum and the extinction.
#
# This is NOT hubbard_U_sweep.sh. That one scans [features] hubbard_U_eV, which
# overrides the Hubbard U and retunes only the V_ee DIAGONAL, leaving the
# kernel's off-diagonal tail untouched. This sweep changes the kernel itself.
#
# CRITICAL — why this script strips hubbard_U_eV:
#   main.cpp:161-174 does
#       U_override = (p.hubbard_U_eV >= 0.0);
#       U_au       = U_override ? p.hubbard_U_eV/au_eV : pot.v_kernel(0.0);
#       if (U_override) p.V_ee.diagonal().setConstant(U_au);
#   so as long as hubbard_U_eV is present the onsite is PINNED at its value no
#   matter what ohno_U_eV says, and this sweep would vary only the tail. With it
#   removed, U = pot.v_kernel(0.0) = ohno_U and the swept value drives both the
#   Coulomb matrix and the Hubbard U consistently. The summary's V0_eV column is
#   the check: it MUST track the swept U, never sit at a constant 15.72.
#
# CAVEAT when reading the summary: UHF is nonlinear with multiple stationary
# points, and a U scan is exactly where that bites. On this system a change of
# 5.7e-10 Ha (1.5e-8 eV) in the onsite U was enough to flip the converged
# solution to a different branch (gap_eV 0.370685 -> 0.895354, BOTH reporting
# converged = 1). A non-monotonic gap_eV or S_total across the grid is more
# likely a branch change than physics. converged = 1 is not evidence of the
# ground state — cross-check suspicious points with hubbard_seed_sweep.sh /
# hubbard_mix_sweep.sh. See HUBBARD_FEATURE.md section 6.
#
# Per U it keeps: eigenvalues.txt, sigma_ext.txt, V_ee.txt (the Ohno matrix),
# magnetization.txt, hubbard_convergence.txt, input.toml, and the two time series
# the moment-vs-dipole figure needs: dipole_time_evolution.txt and
# spin_diag_time_evolution.txt (the induced per-site moment; see save_spin_diag
# below). The spin_diag file is the bulky one — N_t x (2*N_sites + 1) numbers.
#
# Usage:
#   ./ohno_U_sweep.sh configs/graphene_zigzag_triangle.toml
#
# Env overrides:
#   N_T=6            number of t-multiples: U = 1t, 2t, ... N_T*t  (t = |t1| from config)
#   U_LIST="..."     explicit U grid in eV; wins over N_T
#   OMEGA_CUT=25     sigma_ext frequency ceiling (eV). Must clear the U-split gap:
#                    with the Hubbard on the resonance FOLLOWS that gap (13.57 eV at
#                    U = 15.72 eV), so a 6 eV ceiling truncates the spectrum and the
#                    peak tracker latches onto the FFT edge pile-up instead.
#   MAX_JOBS=N       parallel jobs (default: cores - 1)
#   SIM=./sim_blas   simulator binary
#
# Output:
#   data/ohnoUsweep_<tag>/
#       lattice_points.txt, bond_indices.txt     (once — geometry is U-independent)
#       summary.txt        (U_eV  U_over_t  V0_eV  gap_eV  S_total  converged  iters)
#       U_<val>/ eigenvalues.txt sigma_ext.txt V_ee.txt magnetization.txt
#                hubbard_convergence.txt input.toml
#
# Plot with:
#   python3 ploting/ohno_U_sweep_plot.py data/ohnoUsweep_<tag>

set -u

SIM=${SIM:-./sim_blas}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OMEGA_CUT=${OMEGA_CUT:-6}
N_T=${N_T:-4}

BASE_CONFIG=${1:-}
if [ -z "$BASE_CONFIG" ] || [ ! -f "$BASE_CONFIG" ]; then
    echo "Usage: ./ohno_U_sweep.sh configs/your_config.toml"; exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found/executable. Build it first (./build_BLAS.sh)."; exit 1
fi
# MKL build (if chosen) needs the oneAPI runtime; the BLAS build needs nothing.
if echo "$SIM" | grep -q mkl && [ -f /opt/intel/oneapi/setvars.sh ] \
   && ! echo "${LD_LIBRARY_PATH:-}" | grep -q mkl; then
    set +u; source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true; set -u
fi

# Parallelism. Use the INSTALLED core count, not `nproc` alone: under a cgroup or
# affinity mask `nproc` can report 1 even on a 16-core box.
NCPU=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null)
[ -z "$NCPU" ] || [ "$NCPU" -lt 1 ] && NCPU=$(nproc --all 2>/dev/null || nproc 2>/dev/null || echo 1)
NCPU=$((NCPU - 1))
[ "$NCPU" -lt 1 ] && NCPU=1
MAX_JOBS=${MAX_JOBS:-$NCPU}
case "$MAX_JOBS" in ''|*[!0-9]*) MAX_JOBS=$NCPU ;; esac
[ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1

# ---- tag + hopping from the config ------------------------------------------
val() { grep -E "^\s*$1\b" "$BASE_CONFIG" | head -1; }
formation=$(      val 'formation'          | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(val 'formation_shape'    | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(         val 'size_x'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(         val 'size_y'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$(       val 'rotation_angle_deg' | awk -F'=' '{print $2}' | tr -d ' \t\r')
TAG="${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

# t is read from the config, not hardcoded, so the grid follows the model.
t1=$(val 't1' | awk -F'=' '{print $2}' | awk '{print $1}' | tr -d ' \t\r')
t_abs=${t1#-}
if [ -z "$t_abs" ]; then echo "ERROR: could not read t1 from $BASE_CONFIG"; exit 1; fi

# U grid: multiples of t. U = 0 is deliberately excluded — the softening length
# e^2/U_0 diverges there, so V vanishes identically and the kernel degenerates.
U_LIST=${U_LIST:-}
if [ -z "$U_LIST" ]; then
    U_LIST=$(awk -v t="$t_abs" -v n="$N_T" 'BEGIN{for(i=2;i<=n;i++) printf "%.6g ", i*t}')
fi

intensity=$(val 'intensity' | awk -F'=' '{print $2}' | awk '{print $1}')
if [ "$intensity" = "0" ] || [ "$intensity" = "0.0" ]; then
    echo "WARNING: [field] intensity = $intensity — sigma_ext needs a nonzero impulse"
    echo "         drive (e.g. intensity = 1e15, mode = \"ddf\"). Fix the config first."
fi

OUT_DIR="data/ohnoUsweep_${TAG}"
mkdir -p "$OUT_DIR"
rm -rf "${OUT_DIR:?}"/U_* "$OUT_DIR/summary.txt"

FROZEN=$(mktemp --suffix=.toml); cp "$BASE_CONFIG" "$FROZEN"
trap 'rm -f "$FROZEN"' EXIT

echo "Ohno onsite-U sweep"
echo "  config : $BASE_CONFIG   (tag=$TAG)"
echo "  t      : $t_abs eV  (from t1 = $t1)"
echo "  U grid : $U_LIST  eV"
echo "  omega  : ceiling $OMEGA_CUT eV"
echo "  sim    : $SIM   ($MAX_JOBS parallel jobs, ${NCPU} cores detected)"
echo "  output : $OUT_DIR"
echo

# set_key <file> <section> <key> <value>: replace inside [section], or insert there
# if missing. Section-aware on purpose — a flat key match would hit the wrong table.
set_key() {
    local f=$1 sec=$2 key=$3 v=$4 tmp
    tmp=$(mktemp)
    awk -v sec="[$sec]" -v key="$key" -v val="$v" '
        BEGIN { cur=""; done=0; seen=0 }
        /^[[:space:]]*\[[^]]*\]/ {
            if (cur == sec && !done) { print key " = " val; done=1 }
            cur=$0; sub(/#.*/,"",cur); gsub(/[[:space:]]/,"",cur)
            if (cur == sec) seen=1
            print; next
        }
        {
            if (cur == sec && $0 ~ "^[[:space:]]*" key "[[:space:]]*=") {
                if (!done) { print key " = " val; done=1 }
                next
            }
            print
        }
        END {
            if (!done) {
                if (!seen) { print ""; print sec }
                print key " = " val
            }
        }
    ' "$f" > "$tmp" && mv "$tmp" "$f"
}

run_U() {
    local U=$1
    local dest="$OUT_DIR/U_${U}"
    local cfg dir out

    cfg=$(mktemp --suffix=.toml)
    cp "$FROZEN" "$cfg"

    # THE critical line — see the header. Without it the onsite stays pinned.
    grep -v 'hubbard_U_eV' "$cfg" > "$cfg.t" && mv "$cfg.t" "$cfg"

    # The only knobs this sweep touches.
    set_key "$cfg" features coulomb_kernel '"ohno"'
    set_key "$cfg" features ohno_U_eV      "$U"
    set_key "$cfg" analysis run_sigma_ext  true
    set_key "$cfg" analysis omega_cut_off  "$OMEGA_CUT"
    set_key "$cfg" analysis save_rho_full  false
    # spin_diag_time_evolution.txt is the induced diagonal rho_ii(t) - rho0_ii for
    # the up block then the down block, i.e. exactly the per-site induced moment
    # m_l(t) = drho_ll,up - drho_ll,dn that the moment-vs-dipole figure needs
    # (Density.cpp:535-539). It is the only source for it, so this sweep must
    # switch it ON — it used to be forced false purely to save disk.
    set_key "$cfg" analysis save_spin_diag true

    echo "[U=$U] running..."
    out=$("$SIM" "$cfg" 2>&1)
    # C++ prints the path quoted (std::filesystem::path operator<<) -> strip quotes
    dir=$(echo "$out" | grep "All outputs saved under" | awk '{print $NF}' | tr -d '"')

    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo "[U=$U] WARNING: no output directory produced" >&2
        echo "$out" | tail -20 >&2
        rm -f "$cfg"; return 1
    fi

    mkdir -p "$dest"
    for f in eigenvalues.txt sigma_ext.txt V_ee.txt magnetization.txt \
             hubbard_convergence.txt alpha_ext.txt \
             dipole_time_evolution.txt spin_diag_time_evolution.txt; do
        [ -f "$dir/$f" ] && cp "$dir/$f" "$dest/"
    done
    cp "$cfg" "$dest/input.toml"

    # geometry is U-independent: keep one copy at the top level
    for f in lattice_points.txt bond_indices.txt; do
        [ -f "$dir/$f" ] && [ ! -f "$OUT_DIR/$f" ] && cp "$dir/$f" "$OUT_DIR/"
    done

    rm -rf "$dir"; rm -f "$cfg"
    echo "[U=$U] done -> $dest"
}
export -f run_U set_key
export SIM FROZEN OUT_DIR OMEGA_CUT

# First U runs alone so the shared top-level geometry files are copied exactly
# once (no race between parallel workers over the same paths).
first=1
for U in $U_LIST; do
    if [ "$first" = "1" ]; then run_U "$U"; first=0; continue; fi
    run_U "$U" &
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do sleep 1; done
done
wait

# ---------- summary ----------
# V0_eV is the anti-regression column: it is V_ee(0,0) as actually built, so it
# MUST equal the swept U. If it sits at a constant ~15.72 the hubbard_U_eV strip
# failed and every number in this sweep is meaningless.
{
    echo "#U_eV  U_over_t  V0_eV  gap_eV  S_total  converged  iters"
    for d in $(ls -d "$OUT_DIR"/U_* 2>/dev/null); do
        U=$(basename "$d" | sed 's/^U_//')
        mag="$d/magnetization.txt"; vee="$d/V_ee.txt"
        S="nan"; gap="nan"; conv="nan"; it="nan"; V0="nan"
        if [ -f "$mag" ]; then
            S=$(   grep -o 'S_total=[^ ]*'   "$mag" | head -1 | cut -d= -f2)
            gap=$( grep -o 'gap_eV=[^ ]*'    "$mag" | head -1 | cut -d= -f2)
            conv=$(grep -o 'converged=[^ ]*' "$mag" | head -1 | cut -d= -f2)
            it=$(  grep -o 'iters=[^ ]*'     "$mag" | head -1 | cut -d= -f2)
        fi
        # V_ee.txt is in HARTREE -> eV
        [ -f "$vee" ] && V0=$(awk 'NR==1{printf "%.6g", $1*27.2113834; exit}' "$vee")
        UT=$(awk -v u="$U" -v t="$t_abs" 'BEGIN{printf "%.4g", u/t}')
        echo "$U  $UT  ${V0:-nan}  ${gap:-nan}  ${S:-nan}  ${conv:-nan}  ${it:-nan}"
    done | sort -g -k1
} > "$OUT_DIR/summary.txt"

echo
echo "Sweep finished. Data in $OUT_DIR"
column -t "$OUT_DIR/summary.txt" 2>/dev/null || cat "$OUT_DIR/summary.txt"
echo
echo "  V0_eV must track U_eV — if it is constant, the hubbard_U_eV strip failed."
echo "  plot : python3 ploting/ohno_U_sweep_plot.py $OUT_DIR"
