#!/bin/bash
# hubbard_hartree_sweep.sh — compare the self-consistent UHF ground state WITH
# vs WITHOUT the nonlocal charge Hartree ([features] hubbard_hartree = true/false),
# across the same mixing grid, at fixed U.
#
# Hartree OFF is the pure onsite UHF (U splits spin, no nonlocal charge
# feedback). Hartree ON adds, on top, the self-consistent nonlocal charge field
#   phi_i = sum_{j!=i} V_ee(i,j) (n_up_j + n_dn_j - 1)
# which lets doped charge (or any charge imbalance) redistribute across the
# flake (e.g. to the edges) instead of staying frozen at the bare TB density.
# Turning it on is expected to CHANGE the converged S_total / sum|m| / gap
# whenever there is anything for the charge to redistribute (doping, edges,
# broken sublattice symmetry) — unlike the pure mixing sweep (hubbard_mix_sweep.sh),
# where the converged state must NOT depend on the knob being scanned.
#
# It can also change HOW HARD the loop is to converge: the nonlocal Hartree
# adds a second stiff feedback channel (see main.cpp's CDW-instability note on
# the charge kernel), so the same mix that is comfortable Hartree-OFF may need
# more iterations, or fail to converge, Hartree-ON.
#
# For each (mix, hartree) pair this runs one cheap static UHF solve (no
# dynamics, intensity=0) and harvests what solve_hubbard_mft() writes BEFORE
# the time evolution, same as hubbard_mix_sweep.sh, PLUS whether main.cpp's
# own coulomb/hubbard_hartree mismatch warning fired (it must fire iff
# hartree=off and the base config has coulomb=true — see main.cpp ~line 296).
#
# Usage:
#   ./hubbard_hartree_sweep.sh configs/graphene_zigzag_triangle.toml
#   MIX_LIST="0.005 0.01 0.02 0.03" U_EV=15.7217 ./hubbard_hartree_sweep.sh <config>
#
# Output:
#   data/hubbard_hartree_sweep_<tag>.txt   columns:
#       mix  hartree  S_total  sum_abs_m  gap_eV  converged  iters  final_error  mismatch_warned
#   data/hubbard_hartree_sweep_<tag>_traces/mix<val>_h<0|1>.txt  (residual trace per run)
#
# Plot with: python3 ploting/hubbard_hartree_sweep_plot.py data/hubbard_hartree_sweep_<tag>.txt

set -u

SIM=${SIM:-./sim_blas}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# Only the MKL build needs the oneAPI runtime on the library path.
if echo "$SIM" | grep -q mkl && [ -f /opt/intel/oneapi/setvars.sh ] \
   && ! echo "${LD_LIBRARY_PATH:-}" | grep -q mkl; then
    set +u
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
    set -u
fi

BASE_CONFIG=${1:-}
if [ -z "$BASE_CONFIG" ] || [ ! -f "$BASE_CONFIG" ]; then
    echo "Usage: ./hubbard_hartree_sweep.sh configs/your_config.toml"
    exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found/executable. Build it first (e.g. ./build_BLAS.sh)."
    exit 1
fi

# mixing grid (0<mix<=1), same default as hubbard_mix_sweep.sh. Override MIX_LIST=...
MIX_LIST=${MIX_LIST:-"0.005 0.01 0.02 0.03 0.04 0.05"}
# fixed Hubbard U (eV) for the whole sweep. Override U_EV=...
U_EV=${U_EV:-15.7217}
# hartree flags to compare: 0 = off, 1 = on. Override HARTREE_LIST="0 1" to reorder/subset.
HARTREE_LIST=${HARTREE_LIST:-"0 1"}

# ---- tag from the config (formation / shape / size / rotation) -------------
val() { grep -E "^\s*$1\b" "$BASE_CONFIG" | head -1; }
formation=$(      val 'formation'          | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(val 'formation_shape'    | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(         val 'size_x'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(         val 'size_y'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$(       val 'rotation_angle_deg' | awk -F'=' '{print $2}' | tr -d ' \t\r')
coulomb_on=$(     val 'coulomb'            | grep -v 'coulomb_onsite' | awk -F'=' '{print $2}' | tr -d ' \t\r')
TAG="${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

mkdir -p data
OUT="data/hubbard_hartree_sweep_${TAG}.txt"
TRACE_DIR="data/hubbard_hartree_sweep_${TAG}_traces"
mkdir -p "$TRACE_DIR"

t1=$(val 't1' | awk -F'=' '{print $2}' | tr -d ' \t\r')
{
    echo "# Hubbard Hartree ON/OFF comparison sweep"
    echo "# config=${BASE_CONFIG}  tag=${TAG}  t1=${t1} eV  U_eV=${U_EV}  coulomb=${coulomb_on}"
    echo "# mix  hartree  S_total  sum_abs_m  gap_eV  converged  iters  final_error  mismatch_warned"
} > "$OUT"

echo "Config  : $BASE_CONFIG"
echo "Tag     : $TAG   (t1=${t1} eV,  U=${U_EV} eV,  coulomb=${coulomb_on})"
echo "Mix     : $MIX_LIST"
echo "Hartree : $HARTREE_LIST   (0=off 1=on)"
echo "Output  : $OUT"
echo

# set_key <file> <key> <value>: replace an existing (uncommented) key = ... line.
set_key() {
    local f=$1 key=$2 v=$3
    awk -v k="$key" -v val="$v" '
        $0 ~ "^[[:space:]]*"k"[[:space:]]*=" { print k" = "val; next }
        { print }
    ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

for MIX in $MIX_LIST; do
for H in $HARTREE_LIST; do
    HFLAG=$([ "$H" = "1" ] && echo true || echo false)
    STEM="hubHart_${TAG}_m${MIX}_h${H}"
    CFG=$(mktemp --suffix=.toml "/tmp/${STEM}.XXXX")
    CFG_NAMED="$(dirname "$CFG")/${STEM}_$(basename "$CFG")"
    mv "$CFG" "$CFG_NAMED"; CFG="$CFG_NAMED"
    cp "$BASE_CONFIG" "$CFG"

    # cheap static-solve configuration (only the UHF ground state is needed)
    set_key "$CFG" intensity 0.0
    set_key "$CFG" t_max 0.05
    set_key "$CFG" spin_on true
    set_key "$CFG" hubbard true
    set_key "$CFG" hubbard_mix "$MIX"
    set_key "$CFG" hubbard_hartree "$HFLAG"
    set_key "$CFG" run_sigma_ext false
    set_key "$CFG" run_dipole_acc false
    set_key "$CFG" save_rho_full false
    # fix U for the whole sweep: drop any hubbard_U_eV/hubbard_mix/hubbard_hartree
    # line then insert active ones right after [features].
    grep -v 'hubbard_U_eV\|hubbard_mix\|hubbard_hartree' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"
    awk -v u="$U_EV" -v m="$MIX" -v h="$HFLAG" '
        { print }
        /^\[features\]/ { print "hubbard_U_eV = " u; print "hubbard_mix = " m; print "hubbard_hartree = " h }
    ' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"

    rm -rf Simulations/${STEM}_* 2>/dev/null

    printf "mix = %-5s hartree = %s ... " "$MIX" "$H"
    LOG=$(mktemp)
    "$SIM" "$CFG" > "$LOG" 2>&1

    DIR=$(ls -d Simulations/${STEM}_* 2>/dev/null | head -1)
    MAG="$DIR/magnetization.txt"
    if [ -z "$DIR" ] || [ ! -f "$MAG" ]; then
        echo "FAILED (no magnetization.txt)"
        rm -f "$CFG" "$LOG"
        continue
    fi

    hdr=$(grep -m1 'S_total=' "$MAG")
    get() { echo "$hdr" | sed -n "s/.*$1=\([^ ]*\).*/\1/p"; }
    S_total=$(get S_total); sum_abs_m=$(get sum_abs_m); gap=$(get gap_eV)
    conv=$(get converged); iters=$(get iters)

    # main.cpp warns on stderr iff coulomb=true and hubbard_hartree=false: the
    # static ground state then has no nonlocal Coulomb while the dynamics does.
    # Should read 1 for H=0 (with coulomb=true) and 0 for H=1.
    warned=$(grep -qi 'hubbard_hartree = false' "$LOG" && echo 1 || echo 0)

    CONV="$DIR/hubbard_convergence.txt"
    final_err="nan"
    if [ -f "$CONV" ]; then
        final_err=$(grep -v '^#' "$CONV" | tail -1 | awk '{print $2}')
        cp "$CONV" "$TRACE_DIR/mix${MIX}_h${H}.txt"
    fi

    echo "$MIX $H $S_total $sum_abs_m $gap $conv $iters $final_err $warned" >> "$OUT"
    printf "iters=%-4s conv=%s  S=%-7s sum|m|=%-7s gap=%-6s eV  warned=%s\n" \
        "$iters" "$conv" "$S_total" "$sum_abs_m" "$gap" "$warned"

    rm -f "$CFG" "$LOG"
done
done

echo
echo "Done. Wrote $OUT"
echo "Plot: python3 ploting/hubbard_hartree_sweep_plot.py $OUT"
