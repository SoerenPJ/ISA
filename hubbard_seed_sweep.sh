#!/bin/bash
# hubbard_seed_sweep.sh — scan the initial-guess amplitude of the UHF self-
# consistency loop and confirm the outcome is INDEPENDENT of it.
#
# The initial guess is fully deterministic and set by a single number you choose,
#   [features] hubbard_seed = m_seed :   n_up(i) = 0.5 + 0.5*m_seed*s_i
#                                        n_dn(i) = 0.5 - 0.5*m_seed*s_i
# (s_i = +/-1 sublattice). There is NO random engine: every run is exactly
# reproducible, and re-running the same seed gives the same result. This sweep
# just sets m_seed to many different values and checks the loop converges to the
# SAME state for EVERY value (not only a lucky one).
#
# The self-consistent magnetic state is unique, so a healthy solver must, for
# every seed:
#   * converge (the whole point of the check),
#   * reach the SAME S_total / sum|m| / gap,
#   * reach the SAME site populations (n_up, n_dn) — "population throughout",
#   * keep N_total conserved along the whole trace.
# iters may differ (a closer start converges faster); the physics must not.
#
# Usage:
#   ./hubbard_seed_sweep.sh configs/graphene_zigzag_triangle.toml
#   SEED_LIST="0.05 0.1 ... 1.0" U_EV=3.0 MIX=0.3 ./hubbard_seed_sweep.sh <config>
#
# Output:
#   data/hubbard_seed_sweep_<tag>.txt   columns:
#       seed  S_total  sum_abs_m  gap_eV  converged  iters  final_error
#   data/hubbard_seed_sweep_<tag>_traces/seed<v>.txt   (residual trace per seed)
#   data/hubbard_seed_sweep_<tag>_mag/seed<v>.txt      (population texture per seed)
#
# Plot with: python3 ploting/hubbard_seed_sweep_plot.py data/hubbard_seed_sweep_<tag>.txt

set -u

SIM=${SIM:-./sim_blas}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

if echo "$SIM" | grep -q mkl && [ -f /opt/intel/oneapi/setvars.sh ] \
   && ! echo "${LD_LIBRARY_PATH:-}" | grep -q mkl; then
    set +u
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
    set -u
fi

BASE_CONFIG=${1:-}
if [ -z "$BASE_CONFIG" ] || [ ! -f "$BASE_CONFIG" ]; then
    echo "Usage: ./hubbard_seed_sweep.sh configs/your_config.toml"
    exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found/executable. Build it first (e.g. ./build_BLAS.sh)."
    exit 1
fi

# initial-guess amplitude grid (0<seed<=1 is the physical range; a larger value
# just over-polarizes the guess). m_seed = 0 is a symmetric fixed point (no
# symmetry breaking) so it is excluded. Override SEED_LIST=...
SEED_LIST=${SEED_LIST:-"0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"}
# fixed physics + mixing for the whole sweep. Override U_EV=... MIX=...
U_EV=${U_EV:-3.0}
MIX=${MIX:-0.3}

val() { grep -E "^\s*$1\b" "$BASE_CONFIG" | head -1; }
formation=$(      val 'formation'          | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(val 'formation_shape'    | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(         val 'size_x'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(         val 'size_y'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$(       val 'rotation_angle_deg' | awk -F'=' '{print $2}' | tr -d ' \t\r')
TAG="${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

mkdir -p data
OUT="data/hubbard_seed_sweep_${TAG}.txt"
TRACE_DIR="data/hubbard_seed_sweep_${TAG}_traces"
MAG_DIR="data/hubbard_seed_sweep_${TAG}_mag"
mkdir -p "$TRACE_DIR" "$MAG_DIR"

t1=$(val 't1' | awk -F'=' '{print $2}' | tr -d ' \t\r')
{
    echo "# Hubbard initial-guess sweep — outcome independence of the seed amplitude"
    echo "# config=${BASE_CONFIG}  tag=${TAG}  t1=${t1} eV  U_eV=${U_EV}  mix=${MIX}"
    echo "# seed  S_total  sum_abs_m  gap_eV  converged  iters  final_error"
} > "$OUT"

echo "Config : $BASE_CONFIG"
echo "Tag    : $TAG   (t1=${t1} eV,  U=${U_EV} eV,  mix=${MIX})"
echo "Seeds  : $SEED_LIST   (hubbard_seed amplitude, deterministic)"
echo "Output : $OUT"
echo

set_key() {
    local f=$1 key=$2 v=$3
    awk -v k="$key" -v val="$v" '
        $0 ~ "^[[:space:]]*"k"[[:space:]]*=" { print k" = "val; next }
        { print }
    ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

for SEED in $SEED_LIST; do
    STEM="hubSeed_${TAG}_s${SEED}"
    CFG=$(mktemp --suffix=.toml "/tmp/${STEM}.XXXX")
    CFG_NAMED="$(dirname "$CFG")/${STEM}_$(basename "$CFG")"
    mv "$CFG" "$CFG_NAMED"; CFG="$CFG_NAMED"
    cp "$BASE_CONFIG" "$CFG"

    set_key "$CFG" intensity 0.0
    set_key "$CFG" t_max 0.05
    set_key "$CFG" spin_on true
    set_key "$CFG" hubbard true
    set_key "$CFG" run_sigma_ext false
    set_key "$CFG" run_dipole_acc false
    set_key "$CFG" save_rho_full false
    set_key "$CFG" hubbard_seed "$SEED"
    set_key "$CFG" hubbard_mix "$MIX"
    # fix U + mix + seed: strip any existing lines then insert active ones right
    # after [features].
    grep -v 'hubbard_U_eV\|hubbard_mix\|hubbard_seed' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"
    awk -v u="$U_EV" -v m="$MIX" -v s="$SEED" '
        { print }
        /^\[features\]/ {
            print "hubbard_U_eV = " u
            print "hubbard_mix = " m
            print "hubbard_seed = " s
        }
    ' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"

    rm -rf Simulations/${STEM}_* 2>/dev/null

    printf "seed = %-5s ... " "$SEED"
    "$SIM" "$CFG" > /dev/null 2>&1

    DIR=$(ls -d Simulations/${STEM}_* 2>/dev/null | head -1)
    MAG="$DIR/magnetization.txt"
    if [ -z "$DIR" ] || [ ! -f "$MAG" ]; then
        echo "FAILED (no magnetization.txt)"
        rm -f "$CFG"
        continue
    fi

    hdr=$(grep -m1 'S_total=' "$MAG")
    get() { echo "$hdr" | sed -n "s/.*$1=\([^ ]*\).*/\1/p"; }
    S_total=$(get S_total); sum_abs_m=$(get sum_abs_m); gap=$(get gap_eV)
    conv=$(get converged); iters=$(get iters)

    cp "$MAG" "$MAG_DIR/seed${SEED}.txt"   # keep the population texture per seed

    CONV="$DIR/hubbard_convergence.txt"
    final_err="nan"
    if [ -f "$CONV" ]; then
        final_err=$(grep -v '^#' "$CONV" | tail -1 | awk '{print $2}')
        cp "$CONV" "$TRACE_DIR/seed${SEED}.txt"
    fi

    echo "$SEED $S_total $sum_abs_m $gap $conv $iters $final_err" >> "$OUT"
    printf "iters=%-4s conv=%s  S=%-7s sum|m|=%-7s gap=%-6s eV\n" \
        "$iters" "$conv" "$S_total" "$sum_abs_m" "$gap"

    rm -f "$CFG"
done

echo
echo "Done. Wrote $OUT"
echo "Plot: python3 ploting/hubbard_seed_sweep_plot.py $OUT"
