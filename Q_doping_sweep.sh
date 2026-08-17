#!/bin/bash
# Q_doping_sweep.sh — MINIMAL sweep: the ONLY thing that changes between runs is
# the charge doping Q (extra electrons relative to neutral).
#
# Everything else — structure, U, Hartree/Hubbard flags, field, spin, gauge level —
# is taken verbatim from the config you pass in. The script only forces
#   [thermo] use_charge_doping = true      (so Q_doping is what fills rho0, not mu)
#   [thermo] Q_doping          = <Q>
#   [analysis] run_sigma_ext   = true      (unless WITH_SIGMA=0)
# so that a single sweep answers: how do the levels/occupations, the dipole
# response and the extinction spectrum move as electrons are added or removed.
#
# This is the FULL-simulation twin of hubbard_Qdoping_sweep.sh: that one solves the
# static UHF ground state and keeps the spatial charge/spin maps, this one runs the
# whole time evolution and keeps the spectrum + dipole + sigma_ext per Q.
#
# Usage:
#   ./Q_doping_sweep.sh configs/graphene_zigzag_triangle.toml
#   Q_LIST="-2 -1 0 1 2" ./Q_doping_sweep.sh configs/graphene_zigzag_triangle.toml
#   SIM=./sim_mkl JOBS=4 ./Q_doping_sweep.sh <config>
#
# Env overrides:
#   Q_LIST      doping grid, extra electrons (default "-2 -1 0 1 2")
#   SIM         simulator binary            (default ./sim_blas)
#   JOBS        parallel simulations        (default: nproc, capped at 8)
#   WITH_SIGMA  1 = force run_sigma_ext=true (default 1), 0 = leave config alone
#   KEEP_RHO    1 = also keep rho0_j_space.txt (needed only when hubbard is OFF)
#
# Output:
#   data/Qsweep_<formation>_<shape>_<Nx>x<Ny>_rot<angle>/
#       lattice_points.txt, bond_indices.txt      (once — geometry is Q-independent)
#       summary.txt                               (Q  N_e  S_total  gap_eV  converged  iters)
#       Q_<value>/
#           eigenvalues.txt              (Hartree, real imag)
#           hubbard_spectrum.txt         (index energy_eV spin occupation) — if hubbard on
#           rho0_j_space.txt             (eigenbasis rho0; occupations = diagonal)
#           magnetization.txt            (site x y sublattice n_up n_dn m_i + header stats)
#           dipole_time_evolution.txt    (time_au  dipole_au)
#           sigma_ext.txt                (omega_au  sigma_au)
#           alpha_ext.txt                (Re/Im polarizability)
#           input.toml                   (exact config used — provenance)
#
# Plot with:
#   python3 ploting/Q_doping_plots.py data/Qsweep_<tag>

set -u

SIM=${SIM:-./sim_blas}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# MKL build needs the oneAPI runtime on LD_LIBRARY_PATH
if echo "$SIM" | grep -q mkl && [ -f /opt/intel/oneapi/setvars.sh ] \
   && ! echo "${LD_LIBRARY_PATH:-}" | grep -q mkl; then
    set +u; source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true; set -u
fi

BASE_CONFIG=${1:-}
if [ -z "$BASE_CONFIG" ] || [ ! -f "$BASE_CONFIG" ]; then
    echo "Usage: ./Q_doping_sweep.sh configs/your_config.toml"; exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found/executable. Build it first (./build_BLAS.sh)."; exit 1
fi

Q_LIST=${Q_LIST:-"0 1 2 3 4"}
WITH_SIGMA=${WITH_SIGMA:-1}
KEEP_RHO=${KEEP_RHO:-1}

NCPU=$(nproc)
JOBS=${JOBS:-$(( NCPU < 8 ? NCPU : 8 ))}
[ "$JOBS" -lt 1 ] && JOBS=1

# ---------- tag from the structure fields ----------
val() { grep -E "^\s*$1\b" "$BASE_CONFIG" | head -1; }
formation=$(      val 'formation'          | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(val 'formation_shape'    | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(         val 'size_x'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(         val 'size_y'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$(       val 'rotation_angle_deg' | awk -F'=' '{print $2}' | tr -d ' \t\r')

if [ -z "$formation" ] || [ -z "$size_x" ] || [ -z "$size_y" ] || [ -z "$rotation" ]; then
    echo "ERROR: could not extract the structure fields from $BASE_CONFIG"
    echo "  formation='$formation' shape='$formation_shape' size=${size_x}x${size_y} rot='$rotation'"
    exit 1
fi
TAG="${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

# Freeze the config: a concurrent edit of the original must not leak into the sweep.
FROZEN_CONFIG=$(mktemp --suffix=.toml)
cp "$BASE_CONFIG" "$FROZEN_CONFIG"
trap 'rm -f "$FROZEN_CONFIG"' EXIT

OUT_DIR="data/Qsweep_${TAG}"
mkdir -p "$OUT_DIR"
rm -rf "${OUT_DIR:?}"/Q_* "$OUT_DIR/summary.txt"

# ---------- TOML key setter: replace inside [section], or insert if missing ----------
set_key() {   # set_key <file> <section> <key> <value>
    local f=$1 sec=$2 key=$3 val=$4 tmp
    tmp=$(mktemp)
    awk -v sec="[$sec]" -v key="$key" -v val="$val" '
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

# ---------- run one Q ----------
run_Q() {
    local Q=$1
    local dest="$OUT_DIR/Q_${Q}"
    local cfg dir out

    cfg=$(mktemp --suffix=.toml)
    cp "$FROZEN_CONFIG" "$cfg"

    # The ONLY knobs this sweep touches.
    set_key "$cfg" thermo use_charge_doping true
    set_key "$cfg" thermo Q_doping "$Q"
    [ "$WITH_SIGMA" = "1" ] && set_key "$cfg" analysis run_sigma_ext true

    echo "[Q=$Q] running..."
    out=$("$SIM" "$cfg" 2>&1)
    # C++ prints the path quoted (std::filesystem::path operator<<) -> strip quotes
    dir=$(echo "$out" | grep "All outputs saved under" | awk '{print $NF}' | tr -d '"')

    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo "[Q=$Q] WARNING: no output directory produced" >&2
        echo "$out" | tail -20 >&2
        rm -f "$cfg"; return 1
    fi

    mkdir -p "$dest"
    for f in eigenvalues.txt hubbard_spectrum.txt magnetization.txt \
             dipole_time_evolution.txt sigma_ext.txt alpha_ext.txt \
             hubbard_convergence.txt; do
        [ -f "$dir/$f" ] && cp "$dir/$f" "$dest/"
    done
    [ "$KEEP_RHO" = "1" ] && [ -f "$dir/rho0_j_space.txt" ] && cp "$dir/rho0_j_space.txt" "$dest/"
    cp "$cfg" "$dest/input.toml"

    # geometry is Q-independent: keep one copy at the top level
    for f in lattice_points.txt bond_indices.txt; do
        [ -f "$dir/$f" ] && [ ! -f "$OUT_DIR/$f" ] && cp "$dir/$f" "$OUT_DIR/"
    done

    rm -rf "$dir"
    rm -f "$cfg"
    echo "[Q=$Q] done -> $dest"
}

echo "Q doping sweep"
echo "  config : $BASE_CONFIG   (tag: $TAG)"
echo "  sim    : $SIM   (${JOBS} parallel jobs, OMP=${OMP_NUM_THREADS})"
echo "  Q_LIST : $Q_LIST"
echo "  output : $OUT_DIR"
echo ""

# First Q runs alone so lattice_points.txt/bond_indices.txt are copied exactly once
# (no race between parallel workers over the shared top-level files).
first=1
for Q in $Q_LIST; do
    if [ "$first" = "1" ]; then
        run_Q "$Q"
        first=0
        continue
    fi
    run_Q "$Q" &
    while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do sleep 1; done
done
wait

# ---------- summary table ----------
{
    echo "# Q  N_e  S_total  gap_eV  converged  iters"
    for d in $(ls -d "$OUT_DIR"/Q_* 2>/dev/null); do
        Q=$(basename "$d" | sed 's/^Q_//')
        mag="$d/magnetization.txt"
        spec="$d/hubbard_spectrum.txt"
        S="nan"; gap="nan"; conv="nan"; it="nan"; Ne="nan"
        if [ -f "$mag" ]; then
            S=$(  grep -o 'S_total=[^ ]*'   "$mag" | head -1 | cut -d= -f2)
            gap=$(grep -o 'gap_eV=[^ ]*'    "$mag" | head -1 | cut -d= -f2)
            conv=$(grep -o 'converged=[^ ]*' "$mag" | head -1 | cut -d= -f2)
            it=$( grep -o 'iters=[^ ]*'     "$mag" | head -1 | cut -d= -f2)
            Ne=$(awk '!/^#/ {s += $5 + $6} END {printf "%.6f", s}' "$mag")
        elif [ -f "$spec" ]; then
            Ne=$(awk '!/^#/ {s += $4} END {printf "%.6f", s}' "$spec")
        fi
        echo "$Q  ${Ne:-nan}  ${S:-nan}  ${gap:-nan}  ${conv:-nan}  ${it:-nan}"
    done | sort -g -k1
} > "$OUT_DIR/summary.txt"

echo ""
echo "Sweep finished."
echo "  data : $OUT_DIR/Q_<value>/"
column -t "$OUT_DIR/summary.txt" 2>/dev/null || cat "$OUT_DIR/summary.txt"
echo ""
echo "  plot : python3 ploting/Q_doping_plots.py $OUT_DIR"
