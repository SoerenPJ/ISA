#!/bin/bash
# hubbard_doping_sweep.sh — solve the UHF ground state at several doping levels and
# collect the spatial charge/spin textures, to see WHERE the added electrons go and
# how the magnetism changes with doping (paper Fig. 2 / Fig. S1).
#
# Doping is set by the CHEMICAL POTENTIAL mu (grand-canonical Fermi-Dirac fill),
# NOT a fixed integer electron count. This keeps the SCF fill smooth so open-shell
# doped cases converge (the canonical hard-T=0 fill cycles on near-degenerate
# Fermi-level states). N floats with mu (fractional); the script reports it per run.
#
# For each mu it runs the cheap static solve THREE times:
#   * OFF: U>0,  hubbard_hartree = false -> charge just fills bare levels (no self-consistency)
#   * ON : U>0,  hubbard_hartree = true  -> self-consistent nonlocal charge Hartree
#   * U0 : U=0,  hubbard_hartree = true  -> "clean" Hartree, NO Hubbard (reference)
# OFF vs ON reproduces the paper's "w/ vs w/o self-consistency" comparison for the
# maps. U0 vs ON is the with/without-Hubbard pair for the LEVEL-FLOW figure: how the
# energy levels move during the self-consistency iteration, with a pure charge Hartree
# versus Hartree + Hubbard (where the up/dn levels split and the magnetic gap opens).
# Set WITH_U0=0 to skip the extra U=0 run.
# Each run writes magnetization.txt (site x y sublattice n_up n_dn m_i); we harvest
# both the charge n_i = n_up+n_dn and the spin m_i from it, plus hubbard_levels.txt
# (frontier eigenvalues per SCF iteration).
#
# Usage:
#   ./hubbard_doping_sweep.sh configs/graphene_zigzag_triangle.toml
#   MU_LIST="0 4 8 12" U_EV=3.64 MIX=0.01 ./hubbard_doping_sweep.sh <config>
#
# Output (one magnetization file per mu x self-consistency):
#   data/hubbard_doping_<tag>_maps/mu<val>_hart{ON,OFF}.txt
#   data/hubbard_doping_<tag>_maps/levels_mu<val>_{ON,OFF,U0}.txt  (SCF level flow)
#   data/hubbard_doping_<tag>_maps/bond_indices.txt   (lattice skeleton, once)
#
# Plot with: python3 ploting/hubbard_doping_maps.py data/hubbard_doping_<tag>_maps

set -u

SIM=${SIM:-./sim_blas}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

if echo "$SIM" | grep -q mkl && [ -f /opt/intel/oneapi/setvars.sh ] \
   && ! echo "${LD_LIBRARY_PATH:-}" | grep -q mkl; then
    set +u; source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true; set -u
fi

BASE_CONFIG=${1:-}
if [ -z "$BASE_CONFIG" ] || [ ! -f "$BASE_CONFIG" ]; then
    echo "Usage: ./hubbard_doping_sweep.sh configs/your_config.toml"; exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found/executable. Build it first (./build_BLAS.sh)."; exit 1
fi

# doping grid = CHEMICAL POTENTIAL (eV) via grand-canonical Fermi-Dirac filling.
# mu-filling makes the SCF fill smooth (kT ~ 26 meV at T=300K), so open-shell doped
# cases converge instead of cycling on the hard-T=0 canonical fill. N is then set by
# mu (fractional, not an exact integer) and rises slowly with mu (charging energy).
# Override MU_LIST=... . Neutral is mu=0; the dot is stiff (~0.16-0.3 e/eV).
MU_LIST=${MU_LIST:-"0 1.0 2.0 3.0"}
U_EV=${U_EV:-3.64}
MIX=${MIX:-0.01}       # small: open-shell doped fillings cycle at mix>=0.05; 0.01
                       # converges everywhere tested (~3000 iters). Raise for speed
                       # only if your mu list avoids the near-half-filling region.
MAX_ITER=${MAX_ITER:-50000}   # raise together with a smaller MIX (beta=0.01 needs thousands)
WITH_U0=${WITH_U0:-1}  # also solve U=0 (clean Hartree, no Hubbard) for the level-flow figure

val() { grep -E "^\s*$1\b" "$BASE_CONFIG" | head -1; }
formation=$(      val 'formation'          | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(val 'formation_shape'    | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(         val 'size_x'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(         val 'size_y'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$(       val 'rotation_angle_deg' | awk -F'=' '{print $2}' | tr -d ' \t\r')
TAG="${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

OUT_DIR="data/hubbard_doping_${TAG}_maps"
mkdir -p "$OUT_DIR"
# Drop stale magnetization maps + convergence traces from a previous (e.g. longer)
# MU_LIST so the folder reflects ONLY the current grid — otherwise the plotter globs
# them and re-plots leftover mu values. bond_indices.txt is kept (regenerated if missing).
rm -f "$OUT_DIR"/mu*_hart*.txt "$OUT_DIR"/conv_mu*.txt "$OUT_DIR"/population.txt \
      "$OUT_DIR"/levels_mu*.txt

# population tracker: one row per run recording the total electron count N and how
# much it drifts across the SCF loop. mu-filling is grand-canonical, so N is EXPECTED
# to float with mu across the grid; what should NOT change is N within a single solve
# (N_drift = N_last_iter - N_first_iter). This file lets the plotter show both.
POP_FILE="$OUT_DIR/population.txt"
{
    echo "# population tracker — total electron count N per run"
    echo "# N_final  = sum_i (n_up+n_dn) from magnetization.txt (converged texture)"
    echo "# N_first/N_last = N_total at the first/last SCF iteration (hubbard_convergence.txt)"
    echo "# N_drift  = N_last - N_first  (should be ~0 if population is conserved in the loop)"
    echo "# mu  hartree  N_final  N_first_iter  N_last_iter  N_drift  S_z"
} > "$POP_FILE"

echo "Config : $BASE_CONFIG"
echo "Tag    : $TAG   (U=${U_EV} eV, mix=${MIX})"
echo "mu grid: $MU_LIST eV   (chemical potential; each run 3x: Hartree OFF/ON + U=0)"
echo "Output : $OUT_DIR"
echo

set_key() {
    local f=$1 key=$2 v=$3
    awk -v k="$key" -v val="$v" '
        $0 ~ "^[[:space:]]*"k"[[:space:]]*=" { sub(/#.*/, "", $0); print k" = "val; next }
        { print }' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}
ins_feature() {   # ensure key exists under [features]
    local f=$1 key=$2 v=$3
    grep -q "^[[:space:]]*$key[[:space:]]*=" "$f" && set_key "$f" "$key" "$v" \
        || sed -i "/^\[features\]/a $key = $v" "$f"
}

# one entry per solve: "<tag>:<hubbard_hartree>:<U in eV>".
#   OFF/ON = the doping-map pair (Hubbard U on, self-consistent charge off/on)
#   U0     = clean Hartree without Hubbard, only used for the level-flow comparison
RUNS="OFF:false:$U_EV ON:true:$U_EV"
[ "$WITH_U0" = 1 ] && RUNS="$RUNS U0:true:0"

copied_bonds=0
for MU in $MU_LIST; do
    for RUN in $RUNS; do
        HTAG=${RUN%%:*}; rest=${RUN#*:}; HART=${rest%%:*}; U_RUN=${rest##*:}
        STEM="hubDope_${TAG}_mu${MU}_${HTAG}"
        CFG=$(mktemp --suffix=.toml "/tmp/${STEM}.XXXX")
        CFG_NAMED="$(dirname "$CFG")/${STEM}_$(basename "$CFG")"
        mv "$CFG" "$CFG_NAMED"; CFG="$CFG_NAMED"
        cp "$BASE_CONFIG" "$CFG"

        set_key "$CFG" intensity 0.0
        set_key "$CFG" t_max 0.05
        set_key "$CFG" spin_on true
        set_key "$CFG" hubbard true
        set_key "$CFG" use_charge_doping false
        set_key "$CFG" mu "$MU"
        set_key "$CFG" hubbard_mix "$MIX"
        set_key "$CFG" run_sigma_ext false
        set_key "$CFG" run_dipole_acc false
        set_key "$CFG" save_rho_full false
        ins_feature "$CFG" hubbard_U_eV "$U_RUN"
        ins_feature "$CFG" hubbard_hartree "$HART"
        ins_feature "$CFG" hubbard_max_iter "$MAX_ITER"
        ins_feature "$CFG" hubbard_mu_filling true

        rm -rf Simulations/${STEM}_* 2>/dev/null
        printf "mu=%-5s run=%-3s (U=%-5s Hartree=%-5s) ... " "$MU" "$HTAG" "$U_RUN" "$HART"
        "$SIM" "$CFG" > /dev/null 2>&1

        DIR=$(ls -d Simulations/${STEM}_* 2>/dev/null | head -1)
        MAG="$DIR/magnetization.txt"
        if [ -z "$DIR" ] || [ ! -f "$MAG" ]; then
            echo "FAILED (no magnetization.txt)"; rm -f "$CFG"; continue
        fi
        # frontier energy levels per SCF iteration (level-flow figure), all three runs
        [ -f "$DIR/hubbard_levels.txt" ] \
            && cp "$DIR/hubbard_levels.txt" "$OUT_DIR/levels_mu${MU}_${HTAG}.txt"
        # The U=0 run is a level-flow reference only: it is not part of the
        # doping-map / population comparison, so stop here for it.
        if [ "$HTAG" = "U0" ]; then
            hdr=$(grep -m1 'S_total=' "$MAG")
            printf "levels only  S_z=%s iters=%s\n" \
                "$(echo "$hdr" | sed -n 's/.*S_total=\([^ ]*\).*/\1/p')" \
                "$(echo "$hdr" | sed -n 's/.*iters=\([^ ]*\).*/\1/p')"
            rm -f "$CFG"; continue
        fi
        cp "$MAG" "$OUT_DIR/mu${MU}_hart${HTAG}.txt"
        # convergence trace (iter, residual, N, S_z) for the residuals figure
        [ -f "$DIR/hubbard_convergence.txt" ] \
            && cp "$DIR/hubbard_convergence.txt" "$OUT_DIR/conv_mu${MU}_${HTAG}.txt"
        if [ "$copied_bonds" -eq 0 ] && [ -f "$DIR/bond_indices.txt" ]; then
            cp "$DIR/bond_indices.txt" "$OUT_DIR/bond_indices.txt"; copied_bonds=1
        fi

        hdr=$(grep -m1 'S_total=' "$MAG")
        get() { echo "$hdr" | sed -n "s/.*$1=\([^ ]*\).*/\1/p"; }
        # N is not in the header (mu-filling floats it); sum n_up+n_dn from the data.
        # Skip all comment lines (!/^#/): magnetization.txt has three header lines.
        Ntot=$(awk '!/^#/{n+=$5+$6} END{printf "%.2f", n}' "$MAG")
        # population trace: N_total at the first/last SCF iter (col 3 of the conv file),
        # so we can see whether the electron count is conserved through the loop.
        CONV="$DIR/hubbard_convergence.txt"
        if [ -f "$CONV" ]; then
            Nfirst=$(awk '!/^#/{print $3; exit}' "$CONV")
            Nlast=$( awk '!/^#/{v=$3} END{print v}'   "$CONV")
        else
            Nfirst=nan; Nlast=nan
        fi
        Ndrift=$(awk -v a="$Nfirst" -v b="$Nlast" \
            'BEGIN{ if(a=="nan"||b=="nan"){print "nan"} else {printf "%.3e", b-a} }')
        Sz=$(get S_total)
        printf "%s %s %s %s %s %s %s\n" \
            "$MU" "$HTAG" "$Ntot" "$Nfirst" "$Nlast" "$Ndrift" "${Sz:-nan}" >> "$POP_FILE"
        printf "N=%-7s (drift %-9s) S_z=%-8s conv=%s iters=%s\n" \
            "$Ntot" "$Ndrift" "$Sz" "$(get converged)" "$(get iters)"
        rm -f "$CFG"
    done
done

echo
echo "Done. Wrote $OUT_DIR  (population trace: $POP_FILE)"
echo "Plot: python3 ploting/hubbard_doping_maps.py $OUT_DIR"
