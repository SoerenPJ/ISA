#!/bin/bash
# hubbard_plasmon_sweep.sh — Q-doping sweep comparing TWO models:
#
#   L2    : induced Zeeman + self-consistent Peierls phase only
#           (hubbard = false, hubbard_hartree = false)   <- the OLD model
#   L2HH  : + self-consistent UHF magnetism AND the nonlocal Hartree
#           (hubbard = true,  hubbard_hartree = true)    <- the NEW model
#
# Everything else is held fixed — spin_on, self_consistent_phase and
# zeeman_induced are forced TRUE in BOTH arms — so the difference between the
# two spectra at a given Q is ONLY the Hubbard/Hartree block. Purpose: see
# whether the new model shifts the plasmon resonance in sigma_ext as electrons
# are added.
#
# WHY ONLY TWO ARMS: this script used to run a third, "dynamic Hubbard" arm
# switched by [features] hubbard_dynamic. That flag was REMOVED from params.cpp
# (HUBBARD_FEATURE.md:405) — the onsite mean field is now the single unified
# spin-resolved term V_{i,sigma} = U (n_{i,-sigma} - 1/2), which is live in the
# dynamics by construction. Writing hubbard_dynamic set a key nothing reads, so
# the old "static" and "static+dynamic" arms were bit-identical runs.
#
# CAVEAT when reading the difference at odd Q: the L2 arm never enters the SCF
# loop, so its ground state comes from the HARD T=0 canonical fill Rho_0_charge
# (main.cpp:548). hubbard_fd_fill smears only INSIDE the SCF, so it does not
# apply here. When the doped electron lands in a degenerate zero-mode shell the
# hard fill has to pick arbitrarily among degenerate states, and the L2 baseline
# then carries a spurious symmetry-broken occupation (main.cpp:305-340 documents
# exactly this for Q = 1). That is a real property of the old model, not a bug in
# this sweep — but it is part of what the difference map shows at odd Q.
#
# For each Q it keeps sigma_ext.txt for both models. The extinction spectrum
# spans 0..omega_cut_off (forced to OMEGA_CUT, default 25 eV) = the plot y-range.
# The ceiling must clear the new model's U-split gap — see the OMEGA_CUT note below.
#
# Usage:
#   ./hubbard_plasmon_sweep.sh configs/graphene_zigzag_triangle.toml
#
# Env overrides:
#   Q_LIST="0 1 2 3 4"         charge-doping grid (extra electrons vs neutral)
#   HUB_U=<eV>                 override the Hubbard U; UNSET => use the config's
#                              hubbard_U_eV verbatim (the point of the comparison)
#   OMEGA_CUT=25               sigma_ext frequency ceiling (eV) = plot y-range;
#                              must clear the new model's U-split gap (see below)
#   MAX_JOBS=N                 parallel jobs (default: cores - 1)
#   SIM=./sim_blas             simulator binary
#
# Output:
#   data_LLM/plasmon_Q_<tag>/
#       L2_Q_<Q>/sigma_ext.txt
#       L2HH_Q_<Q>/sigma_ext.txt  + magnetization.txt + hubbard_convergence.txt
#       lattice_points.txt
#       summary.txt                (Q  converged  iters  gap_eV  S_total)
#   Plot:  python3 ploting/hubbard_plasmon_plot.py data_LLM/plasmon_Q_<tag>

set -u

SIM=${SIM:-./sim_blas}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

Q_LIST=${Q_LIST:-"0 1 2 3 4"}
# 25 eV, NOT the 6 eV the old mu-sweep used. Turning the Hubbard on opens a
# U-split gap and the main resonance follows it: on the 5x5 zigzag triangle at
# U = 15.72 eV the L2HH peak sits at 13.57 eV against a converged gap of
# 13.41 eV, while the bare L2 arm peaks at 5.90 eV. A 6 eV ceiling therefore
# truncates the new model's spectrum entirely and the peak tracker locks onto
# the FFT edge pile-up instead. Raise further if you raise U.
OMEGA_CUT=${OMEGA_CUT:-6}
# HUB_U is deliberately NOT defaulted: an unset HUB_U means "use the config's
# hubbard_U_eV as written", so the new arm really is the model the config defines.
HUB_U=${HUB_U:-}

BASE_CONFIG=${1:-}
if [ -z "$BASE_CONFIG" ] || [ ! -f "$BASE_CONFIG" ]; then
    echo "Usage: ./hubbard_plasmon_sweep.sh configs/your_config.toml"
    exit 1
fi
if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found/executable. Build it first (e.g. ./build_BLAS.sh)."
    exit 1
fi
# MKL build (if chosen) needs the oneAPI runtime; BLAS build needs nothing.
if echo "$SIM" | grep -q mkl && [ -f /opt/intel/oneapi/setvars.sh ] \
   && ! echo "${LD_LIBRARY_PATH:-}" | grep -q mkl; then
    set +u; source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true; set -u
fi

# Parallelism: one background job per Q value (each runs its 2 models serially).
# Defaults to all cores; override with MAX_JOBS=N.
# Use the INSTALLED core count, not `nproc` alone: under a cgroup/affinity mask
# `nproc` can report 1 even on a 16-core box. /proc/cpuinfo counts real CPUs.
NCPU=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null)
[ -z "$NCPU" ] || [ "$NCPU" -lt 1 ] && NCPU=$(nproc --all 2>/dev/null || nproc 2>/dev/null || echo 1)
NCPU=$((NCPU - 1))
[ "$NCPU" -lt 1 ] && NCPU=1
MAX_JOBS=${MAX_JOBS:-$NCPU}
case "$MAX_JOBS" in ''|*[!0-9]*) MAX_JOBS=$NCPU ;; esac
[ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1
# ---- tag from the config ----------------------------------------------------
val() { grep -E "^\s*$1\b" "$BASE_CONFIG" | head -1; }
formation=$(      val 'formation'          | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(val 'formation_shape'    | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(         val 'size_x'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(         val 'size_y'             | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$(       val 'rotation_angle_deg' | awk -F'=' '{print $2}' | tr -d ' \t\r')
TAG="${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

U_CFG=$(val 'hubbard_U_eV' | awk -F'=' '{print $2}' | awk '{print $1}' | tr -d ' \t\r')

intensity=$(val 'intensity' | awk -F'=' '{print $2}' | awk '{print $1}')
if [ "$intensity" = "0" ] || [ "$intensity" = "0.0" ]; then
    echo "WARNING: [field] intensity = $intensity — sigma_ext needs a nonzero impulse"
    echo "         drive (e.g. intensity = 1e15, mode = \"ddf\"). Fix the config first."
fi

if [ -d "/work/Home/scr/data_LLM" ]; then DATA_DIR="/work/Home/scr/data_LLM"; else DATA_DIR="data_LLM"; fi
SWEEP_DIR="$DATA_DIR/plasmon_Q_${TAG}"
rm -rf "$SWEEP_DIR"; mkdir -p "$SWEEP_DIR"

FROZEN=$(mktemp --suffix=.toml); cp "$BASE_CONFIG" "$FROZEN"
trap 'rm -f "$FROZEN"' EXIT

echo "Config     : $1   (tag=$TAG)"
echo "Models     : L2 (old: hubbard+hartree off) | L2HH (new: both on)"
echo "Hubbard U  : ${HUB_U:-from config (hubbard_U_eV = ${U_CFG:-unset -> vvR(0) ~15.7}) eV}"
echo "Q grid     : $Q_LIST     omega ceiling: $OMEGA_CUT eV"
echo "Simulator  : $SIM"
echo "Parallelism: $MAX_JOBS jobs  (detected ${NCPU} cores; override with MAX_JOBS=N)"
echo "Output     : $SWEEP_DIR"
echo

# upsert <file> <key> <value> <section>: replace an uncommented "key = ..." line
# if present, otherwise insert a fresh one right after the [section] header.
upsert() {
    local f=$1 key=$2 v=$3 sec=$4
    if grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$f"; then
        awk -v k="$key" -v val="$v" \
            '$0 ~ "^[[:space:]]*"k"[[:space:]]*=" {print k" = "val; next} {print}' \
            "$f" > "$f.t" && mv "$f.t" "$f"
    else
        awk -v k="$key" -v val="$v" -v s="$sec" \
            '{print} $0 ~ "^\\["s"\\]" {print k" = "val}' \
            "$f" > "$f.t" && mv "$f.t" "$f"
    fi
}

run_sim() {  # -> output dir
    "$SIM" "$1" 2>&1 | grep "All outputs saved under" | awk '{print $NF}' | tr -d '"'
}

# write a config for one (Q, model) and run it, copying the outputs to dest.
run_case() {
    local Q=$1 model=$2 hub=$3
    local cfg; cfg=$(mktemp --suffix=.toml)
    cp "$FROZEN" "$cfg"

    # doping replaces the old mu axis. mu is not touched at all: with
    # use_charge_doping = true it is ignored by both the SCF fill and the bare
    # path (main.cpp:548 -> Rho_0_charge(evals, N, Q_doping, spin_on)).
    upsert "$cfg" use_charge_doping true thermo
    upsert "$cfg" Q_doping "$Q" thermo

    # shared L2 physics — identical in both arms
    upsert "$cfg" spin_on true hamiltonian
    upsert "$cfg" zeeman_induced true features
    upsert "$cfg" zeeman_external false features
    upsert "$cfg" self_consistent_phase true features
    upsert "$cfg" run_sigma_ext true analysis
    upsert "$cfg" run_dipole_acc false analysis
    upsert "$cfg" save_rho_full false analysis
    upsert "$cfg" save_spin_diag false analysis
    upsert "$cfg" omega_cut_off "$OMEGA_CUT" analysis

    # the ONLY difference between the models: the Hubbard/Hartree block.
    upsert "$cfg" hubbard "$hub" features
    upsert "$cfg" hubbard_hartree "$hub" features
    # hartree_scf must be off: main.cpp:288 has
    #   hartree_only = p.hartree_scf && !p.hubbard
    # so a config leaving it true would silently turn the L2 arm into a THIRD
    # model (spin-blind U = 0 Hartree SCF) instead of the bare baseline.
    upsert "$cfg" hartree_scf false features
    # U comes from the config verbatim unless HUB_U is set explicitly.
    if [ "$hub" = "true" ] && [ -n "$HUB_U" ]; then
        upsert "$cfg" hubbard_U_eV "$HUB_U" features
    fi

    local dir; dir=$(run_sim "$cfg")
    if [ -n "$dir" ] && [ -f "$dir/sigma_ext.txt" ]; then
        local dest="$SWEEP_DIR/${model}_Q_${Q}"; mkdir -p "$dest"
        cp "$dir/sigma_ext.txt" "$dest/"
        # with hubbard on these are the only record of whether the SCF actually
        # converged at this Q — a limit-cycled point must not be read as physics.
        for f in magnetization.txt hubbard_convergence.txt; do
            [ -f "$dir/$f" ] && cp "$dir/$f" "$dest/"
        done
        [ -f "$dir/lattice_points.txt" ] && [ ! -f "$SWEEP_DIR/lattice_points.txt" ] && \
            cp "$dir/lattice_points.txt" "$SWEEP_DIR/"
        rm -rf "$dir"
    else
        echo "WARNING: no sigma_ext for Q=$Q $model" >&2
    fi
    rm -f "$cfg"
}

run_Q() {
    local Q=$1
    echo "  Q = $Q"
    run_case "$Q" L2   false   # old model: no Hubbard, no nonlocal Hartree
    run_case "$Q" L2HH true    # new model: UHF magnetism + nonlocal Hartree
}
export -f run_Q run_case run_sim upsert
export SIM FROZEN SWEEP_DIR OMEGA_CUT HUB_U

for Q in $Q_LIST; do
    run_Q "$Q" &
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do sleep 1; done
done
wait

# ---- SCF health of the new arm ----------------------------------------------
# Read this BEFORE the figure: converged = 0 anywhere means that Q's spectrum is
# a limit-cycle artifact, not the model's answer.
{
    echo "#Q  converged  iters  gap_eV  S_total"
    for d in $(ls -d "$SWEEP_DIR"/L2HH_Q_* 2>/dev/null); do
        Q=$(basename "$d" | sed 's/^L2HH_Q_//')
        mag="$d/magnetization.txt"
        conv="nan"; it="nan"; gap="nan"; S="nan"
        if [ -f "$mag" ]; then
            conv=$(grep -o 'converged=[^ ]*' "$mag" | head -1 | cut -d= -f2)
            it=$(  grep -o 'iters=[^ ]*'     "$mag" | head -1 | cut -d= -f2)
            gap=$( grep -o 'gap_eV=[^ ]*'    "$mag" | head -1 | cut -d= -f2)
            S=$(   grep -o 'S_total=[^ ]*'   "$mag" | head -1 | cut -d= -f2)
        fi
        echo "$Q  ${conv:-nan}  ${it:-nan}  ${gap:-nan}  ${S:-nan}"
    done | sort -g -k1
} > "$SWEEP_DIR/summary.txt"

echo
echo "Done. Data in $SWEEP_DIR"
echo "L2HH SCF health:"
column -t "$SWEEP_DIR/summary.txt" 2>/dev/null || cat "$SWEEP_DIR/summary.txt"
echo
echo "Plot: python3 ploting/hubbard_plasmon_plot.py $SWEEP_DIR"
