#!/bin/bash
# run_all_hubbard.sh — run the Hubbard / self-consistent-Hartree ground-state
# analyses across many configs in one go. For each config it runs (and plots):
#
#   1. doping charge/spin maps   (hubbard_doping_sweep.sh + hubbard_doping_maps.py)
#   2. mixing convergence sweep  (hubbard_mix_sweep.sh    + hubbard_mix_sweep_plot.py)
#   3. initial-guess seed sweep  (hubbard_seed_sweep.sh   + hubbard_seed_sweep_plot.py)
#
# Usage:
#   ./run_all_hubbard.sh                       # default curated list of graphene flakes
#   ./run_all_hubbard.sh configs/a.toml configs/b.toml
#   DO_MIX=0 DO_SEED=0 ./run_all_hubbard.sh    # only the doping maps
#   Q_LIST="0 4 8" U_EV=3.64 ./run_all_hubbard.sh configs/graphene_zigzag_triangle.toml
#
# Knobs (env): DO_DOPING/DO_MIX/DO_SEED (1/0), Q_LIST, MIX_LIST, SEED_LIST, U_EV, MIX.
# Each analysis writes into data/... ; this script prints where every figure landed.

set -u
export MPLBACKEND=${MPLBACKEND:-Agg}       # headless: plotters just save PNG/PDF
SIM=${SIM:-./sim_blas}
export SIM

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(
        configs/graphene_zigzag_triangle.toml
        configs/graphene_armchair_triangle.toml
        configs/graphene_zigzag_bowtie.toml
        configs/graphene_armchair_bowtie.toml
        configs/graphene_zigzag.toml
        configs/zigzag_hubdyn_control.toml
        configs/pentalene.toml
        configs/graphene_armchair.toml          # 20x20: the slow one, kept last
    )
fi

DO_DOPING=${DO_DOPING:-1}
DO_MIX=${DO_MIX:-1}
DO_SEED=${DO_SEED:-1}
# modest default grids so the batch stays quick; override for finer scans
export Q_LIST=${Q_LIST:-"0 2 4"}
export MIX_LIST=${MIX_LIST:-"0.1 0.3 0.5 0.7 0.9"}
export SEED_LIST=${SEED_LIST:-"0.1 0.3 0.5 0.7 0.9"}
export U_EV=${U_EV:-3.64}
export MIX=${MIX:-0.1}

if [ ! -x "$SIM" ]; then
    echo "Simulator '$SIM' not found. Build it first: ./build_BLAS.sh"; exit 1
fi

# run a sweep, then run the "Plot: ..." command it printed (robust to the tag name)
run_and_plot() {
    local out; out=$("$@" 2>&1)
    echo "$out" | grep -E '^(Q=|mix =|seed =|Config|Tag|FAILED|NOT)' | sed 's/^/    /'
    local plot; plot=$(echo "$out" | grep -oE 'python3 ploting/[^ ]+ .+' | tail -1)
    if [ -n "$plot" ]; then
        eval "$plot" 2>&1 | grep -E '^wrote' | sed 's/^/    /'
    else
        echo "    (no plot command produced — sweep may have failed)"
    fi
}

for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then echo "== skip (missing): $cfg"; continue; fi
    echo
    echo "======================================================================"
    echo "== $cfg"
    echo "======================================================================"
    if [ "$DO_DOPING" = 1 ]; then
        echo "-- doping charge/spin maps (Q_LIST=$Q_LIST)"
        run_and_plot ./hubbard_doping_sweep.sh "$cfg"
    fi
    if [ "$DO_MIX" = 1 ]; then
        echo "-- mixing convergence sweep (MIX_LIST=$MIX_LIST)"
        run_and_plot ./hubbard_mix_sweep.sh "$cfg"
    fi
    if [ "$DO_SEED" = 1 ]; then
        echo "-- initial-guess seed sweep (SEED_LIST=$SEED_LIST)"
        run_and_plot ./hubbard_seed_sweep.sh "$cfg"
    fi
done

echo
echo "All done. Figures are under data/ (doping maps in data/hubbard_doping_*_maps/)."
echo "List them with:  ls data/hubbard_*"
