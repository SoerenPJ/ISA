#!/bin/bash

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

BASE_CONFIG=$1

if [ -z "$BASE_CONFIG" ]; then
    echo "Usage: ./large_sweep.sh configs/your_config.toml"
    exit 1
fi

formation=$(grep '^formation =' "$BASE_CONFIG" | awk -F\" '{print $2}')
size_x=$(grep '^size_x' "$BASE_CONFIG" | awk '{print $3}')
size_y=$(grep '^size_y' "$BASE_CONFIG" | awk '{print $3}')
rotation=$(grep '^rotation_angle_deg' "$BASE_CONFIG" | awk '{print $3}')

OUTFILE="sigma_mu_${formation}_${size_x}x${size_y}_rot${rotation}.txt"

rm -f "$OUTFILE"
echo "# mu freq sigma_base sigma_full diff" > "$OUTFILE"

echo "Starting μ sweep using $BASE_CONFIG"
echo "Output file: $OUTFILE"
echo ""

run_mu () {

    mu=$1
    echo "Running mu = $mu"

    tmp=$(mktemp)
    sigma_base=$(mktemp)
    sigma_full=$(mktemp)

    # ---------- BASELINE ----------
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/self_consistent_phase = true/self_consistent_phase = false/" \
        -e "s/zeeman_external = true/zeeman_external = false/" \
        -e "s/zeeman_induced = true/zeeman_induced = false/" \
        -e "s/spin_on = true/spin_on = false/" \
        "$BASE_CONFIG" > "$tmp"

    out=$(./sim_mkl "$tmp")

    dir=$(echo "$out" | grep "All outputs saved under" | awk -F\" '{print $2}')

    cp "$dir/sigma_ext.txt" "$sigma_base"
    rm -rf "$dir"


    # ---------- FULL PHYSICS ----------
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/self_consistent_phase = false/self_consistent_phase = true/" \
        -e "s/zeeman_external = false/zeeman_external = true/" \
        -e "s/zeeman_induced = false/zeeman_induced = true/" \
        -e "s/spin_on = false/spin_on = true/" \
        "$BASE_CONFIG" > "$tmp"

    out=$(./sim_mkl "$tmp")

    dir=$(echo "$out" | grep "All outputs saved under" | awk -F\" '{print $2}')

    cp "$dir/sigma_ext.txt" "$sigma_full"
    rm -rf "$dir"


    lines_base=$(wc -l < "$sigma_base")
    lines_full=$(wc -l < "$sigma_full")

    if [ "$lines_base" -eq "$lines_full" ]; then
        paste "$sigma_base" "$sigma_full" | \
        awk -v mu=$mu '{print mu, $1, $2, $4, $2-$4}' >> "$OUTFILE"
    else
        echo "WARNING: skipping mu=$mu (incomplete output)"
    fi

    rm -f "$tmp" "$sigma_base" "$sigma_full"
}

export -f run_mu
export BASE_CONFIG OUTFILE


for mu in $(seq 2.3 0.01 3.5)
do
    run_mu $mu &

    while [ $(jobs -r | wc -l) -ge 4 ]
    do
        sleep 1
    done
done

wait

echo ""
echo "Sweep finished"
echo "Results saved in $OUTFILE"
