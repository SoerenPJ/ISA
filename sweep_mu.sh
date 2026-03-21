#!/bin/bash

BASE_CONFIG=$1

if [ -z "$BASE_CONFIG" ]; then
    echo "Usage: ./sweep_mu.sh configs/your_config.toml"
    exit 1
fi

OUTFILE="sigma_mu_sweep.txt"
rm -f "$OUTFILE"

echo "# mu freq sigma_base sigma_full diff" > "$OUTFILE"

echo "Starting μ sweep using $BASE_CONFIG"
echo ""

for mu in $(seq 0 1 1)
do
    echo "Running mu = $mu"

    # ---------- BASELINE ----------
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/self_consistent_phase = true/self_consistent_phase = false/" \
        -e "s/zeeman_external = true/zeeman_external = false/" \
        -e "s/zeeman_induced = true/zeeman_induced = false/" \
        "$BASE_CONFIG" > tmp.toml

    ./sim_mkl tmp.toml

    dir=$(ls -td Simulations/* | head -1)
    cp "$dir/sigma_ext.txt" sigma_base.txt
    rm -rf "$dir"


    # ---------- FULL PHYSICS ----------
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/self_consistent_phase = false/self_consistent_phase = true/" \
        -e "s/zeeman_external = false/zeeman_external = true/" \
        -e "s/zeeman_induced = false/zeeman_induced = true/" \
        "$BASE_CONFIG" > tmp.toml

    ./sim_mkl tmp.toml

    dir=$(ls -td Simulations/* | head -1)
    cp "$dir/sigma_ext.txt" sigma_full.txt
    rm -rf "$dir"


    # ---------- SAVE BASE, FULL, AND DIFFERENCE ----------
    paste sigma_base.txt sigma_full.txt | \
    awk -v mu=$mu '{print mu, $1, $2, $4, $2-$4}' >> "$OUTFILE"

done

rm -f tmp.toml sigma_base.txt sigma_full.txt

echo ""
echo "Sweep finished"
echo "Results saved in $OUTFILE"
