#!/bin/bash
# hubbard_mix_sweep.sh — scan the linear-mixing parameter of the UHF self-
# consistency loop and record how it affects CONVERGENCE.
#
# Physics is fixed (same U, same lattice); only the damping factor
#   n <- (1 - mix)*n_old + mix*n_new
# is varied. The self-consistent state is unique, so every mix that converges
# must reach the SAME S_total / sum|m| / gap — mixing only changes HOW MANY
# iterations it takes (and, at large mix, whether it converges at all). This is
# exactly the "does it always converge, even if some take longer" question.
#
# For each mix it runs one cheap static UHF solve (no dynamics) and harvests
# what solve_hubbard_mft() writes BEFORE the time evolution:
#   magnetization.txt       header -> S_total sum_abs_m gap_eV converged iters
#   hubbard_convergence.txt -> full residual trace (copied per mix)
#
# Usage:
#   ./hubbard_mix_sweep.sh configs/graphene_zigzag_triangle.toml
#   MIX_LIST="0.05 0.1 0.2 ... 0.95" U_EV=3.0 ./hubbard_mix_sweep.sh <config>
#
# Output:
#   data/hubbard_mix_sweep_<tag>.txt   columns:
#       mix  S_total  sum_abs_m  gap_eV  converged  iters  final_error
#   data/hubbard_mix_sweep_<tag>_traces/mix<val>.txt  (residual trace per mix)
#
# Plot with: python3 ploting/hubbard_mix_sweep_plot.py data/hubbard_mix_sweep_<tag>.txt

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
    echo "Usage: ./hubbard_mix_sweep.sh configs/your_config.toml"
    exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found/executable. Build it first (e.g. ./build_BLAS.sh)."
    exit 1
fi

# mixing grid (0<mix<=1). Small mix = heavy damping (slow but safe); large mix =
# little damping (fast if it works, may oscillate/diverge). Override MIX_LIST=...
MIX_LIST=${MIX_LIST:-"0.05 0.1 0.2 0.3 0.4 0.5"}
# fixed Hubbard U (eV) for the whole sweep. Override U_EV=...
U_EV=${U_EV:-15.7217}

# ---- tag from the config (formation / shape / size / rotation) -------------
val() { grep -E "^\s*$1\b" "$BASE_CONFIG" | head -1; }
formation=$(      val 'formation'          | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(val 'formation_shape'    | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(         val 'size_x'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(         val 'size_y'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$(       val 'rotation_angle_deg' | awk -F'=' '{print $2}' | tr -d ' \t\r')
TAG="${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

mkdir -p data
OUT="data/hubbard_mix_sweep_${TAG}.txt"
TRACE_DIR="data/hubbard_mix_sweep_${TAG}_traces"
mkdir -p "$TRACE_DIR"

t1=$(val 't1' | awk -F'=' '{print $2}' | tr -d ' \t\r')
{
    echo "# Hubbard mixing sweep — convergence vs linear-mixing factor"
    echo "# config=${BASE_CONFIG}  tag=${TAG}  t1=${t1} eV  U_eV=${U_EV}"
    echo "# mix  S_total  sum_abs_m  gap_eV  converged  iters  final_error"
} > "$OUT"

echo "Config : $BASE_CONFIG"
echo "Tag    : $TAG   (t1=${t1} eV,  U=${U_EV} eV)"
echo "Mix    : $MIX_LIST"
echo "Output : $OUT"
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
    STEM="hubMix_${TAG}_m${MIX}"
    CFG=$(mktemp --suffix=.toml "/tmp/${STEM}.XXXX")
    CFG_NAMED="$(dirname "$CFG")/${STEM}_$(basename "$CFG")"
    mv "$CFG" "$CFG_NAMED"; CFG="$CFG_NAMED"
    cp "$BASE_CONFIG" "$CFG"

    # cheap static-solve configuration (only the UHF ground state is needed)
    set_key "$CFG" intensity 0.0
    set_key "$CFG" t_max 0.05
    set_key "$CFG" spin_on true
    set_key "$CFG" hubbard true
    #set_key "$CFG" hartree_scf true #uncomment and comment out the others to make a pure hartree sweep 
    set_key "$CFG" hubbard_mix "$MIX"
    set_key "$CFG" run_sigma_ext false
    set_key "$CFG" run_dipole_acc false
    set_key "$CFG" save_rho_full false
    # fix U for the whole sweep: drop any hubbard_U_eV line then insert an active
    # one right after [features].
    grep -v 'hubbard_U_eV\|hubbard_mix' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"
    awk -v u="$U_EV" -v m="$MIX" '
        { print }
        /^\[features\]/ { print "hubbard_U_eV = " u; print "hubbard_mix = " m }
    ' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"

    rm -rf Simulations/${STEM}_* 2>/dev/null

    printf "mix = %-5s ... " "$MIX"
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

    CONV="$DIR/hubbard_convergence.txt"
    final_err="nan"
    if [ -f "$CONV" ]; then
        final_err=$(grep -v '^#' "$CONV" | tail -1 | awk '{print $2}')
        cp "$CONV" "$TRACE_DIR/mix${MIX}.txt"
    fi

    echo "$MIX $S_total $sum_abs_m $gap $conv $iters $final_err" >> "$OUT"
    printf "iters=%-4s conv=%s  S=%-7s sum|m|=%-7s gap=%-6s eV\n" \
        "$iters" "$conv" "$S_total" "$sum_abs_m" "$gap"

    rm -f "$CFG"
done

echo
echo "Done. Wrote $OUT"
echo "Plot: python3 ploting/hubbard_mix_sweep_plot.py $OUT"
