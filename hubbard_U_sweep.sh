#!/bin/bash
# hubbard_U_sweep.sh — scan the Hubbard U and collect the static UHF ground-state
# diagnostics, for a sensitivity analysis of the mean-field magnetism.
#
# For each U (in eV) it runs one simulation with [features] hubbard_U_eV = U and
# harvests the quantities the UHF solver already writes BEFORE the time evolution:
#
#   magnetization.txt   header -> U_eV  S_total  sum_abs_m  gap_eV  converged  iters
#   hubbard_convergence.txt     -> full residual trace (copied per U)
#
# The dynamics is not needed for these ground-state diagnostics, so each run is
# made cheap: intensity = 0, t_max tiny. Everything the
# sweep reports is written by solve_hubbard_mft() before evolve_rho_over_time().
#
# Usage:
#   ./hubbard_U_sweep.sh configs/graphene_zigzag_triangle.toml
#   U_LIST="0.1 0.5 1 2 3 5 8 10" ./hubbard_U_sweep.sh configs/graphene_zigzag_triangle.toml
#
# Output:
#   data/hubbard_U_sweep_<tag>.txt            columns:
#       U_eV  S_total  sum_abs_m  gap_eV  converged  iters  final_error
#   data/hubbard_U_sweep_<tag>_traces/U<val>.txt   (one convergence trace per U)
#
# Plot with:  python3 ploting/hubbard_U_sweep_plot.py data/hubbard_U_sweep_<tag>.txt

set -u

SIM=${SIM:-./sim_blas}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# Only the MKL build needs the oneAPI runtime on the library path; the BLAS
# build (sim_blas, the default) links the system OpenBLAS and needs nothing.
if echo "$SIM" | grep -q mkl && [ -f /opt/intel/oneapi/setvars.sh ] \
   && ! echo "${LD_LIBRARY_PATH:-}" | grep -q mkl; then
    set +u
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
    set -u
fi

BASE_CONFIG=$1
if [ -z "${BASE_CONFIG:-}" ] || [ ! -f "$BASE_CONFIG" ]; then
    echo "Usage: ./hubbard_U_sweep.sh configs/your_config.toml"
    exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found/executable. Build it first (e.g. ./build_BLAS.sh)."
    exit 1
fi

# Default U grid (eV). Includes the literature anchors 2t=5.6, Uc=2.23t=6.24,
# bare cRPA 3.63t=10.16 for t=2.8 eV. Override with U_LIST=...
U_LIST=${U_LIST:-"0.1 0.5 1.0 1.5 2.0 2.8 3.5 4.0 4.5 5.0 5.6 6.24 7.0 8.0 9.0 10.16"}

# ---- tag from the config (formation / shape / size / rotation) -------------
val() { grep -E "^\s*$1\b" "$BASE_CONFIG" | head -1; }
formation=$(      val 'formation'          | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(val 'formation_shape'    | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(         val 'size_x'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(         val 'size_y'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$(       val 'rotation_angle_deg' | awk -F'=' '{print $2}' | tr -d ' \t\r')
TAG="${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

mkdir -p data
OUT="data/hubbard_U_sweep_${TAG}.txt"
TRACE_DIR="data/hubbard_U_sweep_${TAG}_traces"
mkdir -p "$TRACE_DIR"

t1=$(val 't1' | awk -F'=' '{print $2}' | tr -d ' \t\r')
{
    echo "# Hubbard U sweep — static UHF ground-state diagnostics"
    echo "# config=${BASE_CONFIG}  tag=${TAG}  t1=${t1} eV"
    echo "# U_eV  S_total  sum_abs_m  gap_eV  converged  iters  final_error"
} > "$OUT"

echo "Config : $BASE_CONFIG"
echo "Tag    : $TAG   (t1=${t1} eV)"
echo "U grid : $U_LIST"
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

for U in $U_LIST; do
    STEM="hubU_${TAG}_U${U}"
    CFG=$(mktemp --suffix=.toml "/tmp/${STEM}.XXXX")
    # give the sim a recognisable config stem so we can find its output folder
    CFG_NAMED="$(dirname "$CFG")/${STEM}_$(basename "$CFG")"
    mv "$CFG" "$CFG_NAMED"; CFG="$CFG_NAMED"
    cp "$BASE_CONFIG" "$CFG"

    # force the cheap static-solve configuration
    set_key "$CFG" intensity 0.0
    set_key "$CFG" t_max 0.05
    set_key "$CFG" spin_on true
    set_key "$CFG" hubbard true
    set_key "$CFG" run_sigma_ext false
    set_key "$CFG" run_dipole_acc false
    set_key "$CFG" save_rho_full false
    # set the charge U: drop any existing (commented or not) hubbard_U_eV line,
    # then insert an active one right after the [features] header.
    grep -v 'hubbard_U_eV' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"
    awk -v u="$U" '
        { print }
        /^\[features\]/ { print "hubbard_U_eV = " u }
    ' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"

    # a fresh run must not collide with a stale folder of the same stem
    rm -rf Simulations/${STEM}_* 2>/dev/null

    printf "U = %-6s eV ... " "$U"
    "$SIM" "$CFG" > /dev/null 2>&1

    DIR=$(ls -d Simulations/${STEM}_* 2>/dev/null | head -1)
    MAG="$DIR/magnetization.txt"
    if [ -z "$DIR" ] || [ ! -f "$MAG" ]; then
        echo "FAILED (no magnetization.txt)"
        rm -f "$CFG"
        continue
    fi

    # the header line: "# U_eV=.. S_total=.. sum_abs_m=.. gap_eV=.. converged=.. iters=.."
    hdr=$(grep -m1 'U_eV=' "$MAG")
    get() { echo "$hdr" | sed -n "s/.*$1=\([^ ]*\).*/\1/p"; }
    S_total=$(get S_total); sum_abs_m=$(get sum_abs_m); gap=$(get gap_eV)
    conv=$(get converged); iters=$(get iters)

    # record the Lieb prediction once, from the sublattice column of the texture:
    # S_Lieb = |N_A - N_B| / 2  (col 4 = sublattice label +1/-1)
    if ! grep -q 'lieb_S=' "$OUT"; then
        NA=$(grep -v '^#' "$MAG" | awk '$4==1'  | wc -l)
        NB=$(grep -v '^#' "$MAG" | awk '$4==-1' | wc -l)
        LIEB=$(awk -v a="$NA" -v b="$NB" 'BEGIN{d=a-b; if(d<0)d=-d; print d/2}')
        echo "# lieb_S=${LIEB} N_A=${NA} N_B=${NB}" >> "$OUT"
    fi

    CONV="$DIR/hubbard_convergence.txt"
    final_err="nan"
    if [ -f "$CONV" ]; then
        final_err=$(grep -v '^#' "$CONV" | tail -1 | awk '{print $2}')
        cp "$CONV" "$TRACE_DIR/U${U}.txt"
    fi

    echo "$U $S_total $sum_abs_m $gap $conv $iters $final_err" >> "$OUT"
    printf "S=%-8s  sum|m|=%-8s  gap=%-8s eV  iters=%-4s conv=%s\n" \
        "$S_total" "$sum_abs_m" "$gap" "$iters" "$conv"

    rm -f "$CFG"
done

echo
echo "Done. Wrote $OUT"
echo "Plot: python3 ploting/hubbard_U_sweep_plot.py $OUT"
