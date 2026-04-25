#!/bin/bash
# gauge_sweep.sh — sweep mu and collect gauge consistency metrics at each step.
#
# Runs full-physics simulations (spin_on, zeeman_induced, zeeman_external,
# self_consistent_phase all enabled), calls gauge_batch.py to extract scalar
# metrics, then deletes the simulation folder to save disk.
#
# Usage:
#   ./gauge_sweep.sh configs/graphene_armchair.toml
#
# Output:
#   gauge_metrics_mu_<formation>_<Nx>x<Ny>_rot<angle>.txt
#
# Columns:
#   mu  corr_flux  alpha_flux  mean_rel_peak  max_rel_peak  mean_ratio_peak
#   rms_flux_peak  dynamic_range  mean_rel_curl_peak

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

BASE_CONFIG=$1

if [ -z "$BASE_CONFIG" ]; then
    echo "Usage: ./gauge_sweep.sh configs/your_config.toml"
    exit 1
fi

formation=$(grep '^formation =' "$BASE_CONFIG" | awk -F\" '{print $2}')
size_x=$(grep '^size_x' "$BASE_CONFIG" | awk '{print $3}')
size_y=$(grep '^size_y' "$BASE_CONFIG" | awk '{print $3}')
rotation=$(grep '^rotation_angle_deg' "$BASE_CONFIG" | awk '{print $3}')

OUTFILE="gauge_metrics_mu_${formation}_${size_x}x${size_y}_rot${rotation}.txt"
TS_DIR="gauge_ts_mu_${formation}_${size_x}x${size_y}_rot${rotation}"

rm -f "$OUTFILE"
rm -rf "$TS_DIR"
mkdir -p "$TS_DIR"
echo "# mu corr_flux alpha_flux mean_rel_peak max_rel_peak mean_ratio_peak rms_flux_peak dynamic_range mean_rel_curl_peak" > "$OUTFILE"

echo "Starting gauge mu sweep using $BASE_CONFIG"
echo "Output file: $OUTFILE"
echo "Timeseries dir: $TS_DIR"
echo ""

run_gauge_mu() {
    mu=$1
    echo "Running mu = $mu"

    tmp=$(mktemp --suffix=.toml)

    # Force all full-physics flags on regardless of base config state
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/self_consistent_phase = false/self_consistent_phase = true/" \
        -e "s/zeeman_external = false/zeeman_external = true/" \
        -e "s/zeeman_induced = false/zeeman_induced = true/" \
        -e "s/spin_on = false/spin_on = true/" \
        "$BASE_CONFIG" > "$tmp"

    out=$(./sim_mkl "$tmp" 2>&1)

    dir=$(echo "$out" | grep "All outputs saved under" | awk '{print $NF}')

    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo "WARNING: no sim dir found for mu=$mu — writing nan row" >&2
        echo "$mu nan nan nan nan nan nan nan nan" > "${OUTFILE}.mu_${mu}"
        rm -f "$tmp"
        return
    fi

    # Extract scalar gauge metrics (one line, 8 values)
    metrics=$(python3 ploting/gauge_batch.py "$dir" 2>/dev/null)
    if [ -z "$metrics" ]; then
        metrics="nan nan nan nan nan nan nan nan"
    fi

    echo "$mu $metrics" > "${OUTFILE}.mu_${mu}"

    # Save per-mu timeseries for violin plot
    MPLBACKEND=Agg python3 ploting/gauge_comparison.py "$dir" &>/dev/null
    ts="$dir/gauge_plots/flux_comparison_timeseries.txt"
    [ -f "$ts" ] && cp "$ts" "${TS_DIR}/mu_${mu}.txt"

    rm -rf "$dir"
    rm -f "$tmp"
}

export -f run_gauge_mu
export BASE_CONFIG OUTFILE TS_DIR

for mu in $(seq 2.3 0.01 3.5)
do
    run_gauge_mu $mu &

    while [ $(jobs -r | wc -l) -ge 4 ]
    do
        sleep 1
    done
done

wait

for f in $(ls "${OUTFILE}".mu_* 2>/dev/null | sort -t_ -k2 -n); do
    cat "$f" >> "$OUTFILE"
    rm -f "$f"
done

echo ""
echo "Gauge sweep finished."
echo "Results saved in $OUTFILE"
