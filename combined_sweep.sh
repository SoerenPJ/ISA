#!/bin/bash
# combined_sweep.sh — mu sweep collecting sigma_ext AND gauge metrics for L0, L1, L2.
#
# For each mu value three simulations are run:
#   L0  Electrostatic baseline  (spin_on=false, zeeman_induced=false, self_consistent_phase=false)
#   L1  + Induced Zeeman term   (spin_on=true,  zeeman_induced=true,  self_consistent_phase=false)
#   L2  + Induced Peierls phase (spin_on=true,  zeeman_induced=true,  self_consistent_phase=true)
#
# Each parallel job writes to its own per-job temp file to avoid race conditions.
# After all jobs finish the temp files are sorted by mu and merged.
#
# Usage:
#   ./combined_sweep.sh configs/graphene_armchair.toml
#
# Output files:
#   sigma_mu_<formation>_<Nx>x<Ny>_rot<angle>.txt
#       columns: mu  freq  sigma_L0  sigma_L1  sigma_L2  diff_L1  diff_L2
#
#   gauge_metrics_mu_<formation>_<Nx>x<Ny>_rot<angle>.txt
#       columns: mu  level  corr_flux  alpha_flux  mean_rel_peak  max_rel_peak
#                mean_ratio_peak  rms_flux_peak  dynamic_range  mean_rel_curl_peak

NCPU=$(nproc)
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS MKL_NUM_THREADS

MAX_JOBS=$((NCPU / OMP_NUM_THREADS))
[ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1

echo "Detected $NCPU CPU threads"
echo "Using $OMP_NUM_THREADS threads per simulation"
echo "Running up to $MAX_JOBS simulations in parallel"

BASE_CONFIG=$1

if [ -z "$BASE_CONFIG" ]; then
    echo "Usage: ./combined_sweep.sh configs/your_config.toml"
    exit 1
fi

formation=$(grep '^\s*formation\b' "$BASE_CONFIG" | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(   grep '^\s*size_x'             "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(   grep '^\s*size_y'             "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$( grep '^\s*rotation_angle_deg' "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')

# Abort early if any field came out empty so the filenames are not silently broken
if [ -z "$formation" ] || [ -z "$size_x" ] || [ -z "$size_y" ] || [ -z "$rotation" ]; then
    echo "ERROR: could not extract one or more config fields from $BASE_CONFIG"
    echo "  formation='$formation'  size_x='$size_x'  size_y='$size_y'  rotation='$rotation'"
    echo "  Run:  cat -A \"$BASE_CONFIG\" | grep -E 'formation|size_x|size_y|rotation_angle_deg'"
    echo "  to inspect the actual characters in those lines."
    exit 1
fi

SIGMA_FILE="sigma_mu_${formation}_${size_x}x${size_y}_rot${rotation}.txt"
GAUGE_FILE="gauge_metrics_mu_${formation}_${size_x}x${size_y}_rot${rotation}.txt"
TS_DIR="gauge_ts_mu_${formation}_${size_x}x${size_y}_rot${rotation}"

# Remove old output files and any leftover per-job temp files
rm -f "$SIGMA_FILE" "$GAUGE_FILE"
rm -f "${SIGMA_FILE}".mu_* "${GAUGE_FILE}".mu_*
rm -rf "$TS_DIR"
mkdir -p "$TS_DIR"

echo "# mu freq sigma_L0 sigma_L1 sigma_L2 diff_L1 diff_L2" > "$SIGMA_FILE"
echo "# mu level corr_flux alpha_flux mean_rel_peak max_rel_peak mean_ratio_peak rms_flux_peak dynamic_range mean_rel_curl_peak" > "$GAUGE_FILE"

echo "Starting combined mu sweep using $BASE_CONFIG"
echo "Sigma output : $SIGMA_FILE"
echo "Gauge output : $GAUGE_FILE"
echo "Timeseries dir: $TS_DIR"
echo ""

# ---------- Helper: run one sim level and return its output dir ----------
# Usage: run_level <tmp_config> <output_var_name>
# Prints the output directory path to stdout, or empty string on failure.
run_sim() {
    local cfg=$1
    local out
    out=$(./sim_mkl "$cfg" 2>&1)
    echo "$out" | grep "All outputs saved under" | awk '{print $NF}'
}

# ---------- Per-mu worker ----------
run_mu() {
    local mu=$1
    local do_gauge=$2
    echo "Running mu = $mu"

    local tmp
    tmp=$(mktemp --suffix=.toml)

    local sigma_L0 sigma_L1 sigma_L2
    sigma_L0=$(mktemp)
    sigma_L1=$(mktemp)
    sigma_L2=$(mktemp)

    local sigma_job="${SIGMA_FILE}.mu_${mu}"
    local gauge_job="${GAUGE_FILE}.mu_${mu}"

    # ------------------------------------------------------------------ L0
    # Electrostatic baseline: spin off, no zeeman, no self-consistent phase
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/spin_on *= *true/spin_on = false/" \
        -e "s/zeeman_induced *= *true/zeeman_induced = false/" \
        -e "s/zeeman_external *= *true/zeeman_external = false/" \
        -e "s/self_consistent_phase *= *true/self_consistent_phase = false/" \
        "$BASE_CONFIG" > "$tmp"

    local dir
    dir=$(run_sim "$tmp")
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        cp "$dir/sigma_ext.txt" "$sigma_L0"
        if [ "$do_gauge" = "1" ]; then
            local m
            m=$(python3 ploting/gauge_batch.py "$dir" 2>/dev/null)
            [ -z "$m" ] && m="nan nan nan nan nan nan nan nan"
            echo "$mu L0 $m" >> "$gauge_job"
            MPLBACKEND=Agg python3 ploting/gauge_comparison.py "$dir" &>/dev/null
            local ts="$dir/gauge_plots/flux_comparison_timeseries.txt"
            [ -f "$ts" ] && cp "$ts" "${TS_DIR}/L0_mu_${mu}.txt"
        fi
        rm -rf "$dir"
    else
        echo "WARNING: no L0 dir for mu=$mu" >&2
        touch "$sigma_L0"
        [ "$do_gauge" = "1" ] && echo "$mu L0 nan nan nan nan nan nan nan nan" >> "$gauge_job"
    fi

    # ------------------------------------------------------------------ L1
    # Add induced Zeeman correction; Peierls phase still off
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/spin_on *= *false/spin_on = true/" \
        -e "s/zeeman_induced *= *false/zeeman_induced = true/" \
        -e "s/zeeman_external *= *true/zeeman_external = false/" \
        -e "s/self_consistent_phase *= *true/self_consistent_phase = false/" \
        "$BASE_CONFIG" > "$tmp"

    dir=$(run_sim "$tmp")
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        cp "$dir/sigma_ext.txt" "$sigma_L1"
        if [ "$do_gauge" = "1" ]; then
            local m
            m=$(python3 ploting/gauge_batch.py "$dir" 2>/dev/null)
            [ -z "$m" ] && m="nan nan nan nan nan nan nan nan"
            echo "$mu L1 $m" >> "$gauge_job"
            MPLBACKEND=Agg python3 ploting/gauge_comparison.py "$dir" &>/dev/null
            local ts="$dir/gauge_plots/flux_comparison_timeseries.txt"
            [ -f "$ts" ] && cp "$ts" "${TS_DIR}/L1_mu_${mu}.txt"
        fi
        rm -rf "$dir"
    else
        echo "WARNING: no L1 dir for mu=$mu" >&2
        touch "$sigma_L1"
        [ "$do_gauge" = "1" ] && echo "$mu L1 nan nan nan nan nan nan nan nan" >> "$gauge_job"
    fi

    # ------------------------------------------------------------------ L2
    # Full physics: induced Zeeman + induced Peierls phase (self-consistent)
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/spin_on *= *false/spin_on = true/" \
        -e "s/zeeman_induced *= *false/zeeman_induced = true/" \
        -e "s/zeeman_external *= *true/zeeman_external = false/" \
        -e "s/self_consistent_phase *= *false/self_consistent_phase = true/" \
        "$BASE_CONFIG" > "$tmp"

    dir=$(run_sim "$tmp")
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        cp "$dir/sigma_ext.txt" "$sigma_L2"
        if [ "$do_gauge" = "1" ]; then
            local m
            m=$(python3 ploting/gauge_batch.py "$dir" 2>/dev/null)
            [ -z "$m" ] && m="nan nan nan nan nan nan nan nan"
            echo "$mu L2 $m" >> "$gauge_job"
            MPLBACKEND=Agg python3 ploting/gauge_comparison.py "$dir" &>/dev/null
            local ts="$dir/gauge_plots/flux_comparison_timeseries.txt"
            [ -f "$ts" ] && cp "$ts" "${TS_DIR}/L2_mu_${mu}.txt"
        fi
        rm -rf "$dir"
    else
        echo "WARNING: no L2 dir for mu=$mu" >&2
        touch "$sigma_L2"
        [ "$do_gauge" = "1" ] && echo "$mu L2 nan nan nan nan nan nan nan nan" >> "$gauge_job"
    fi

    # ------------------------------------------------------------------ sigma merge
    local nl0 nl1 nl2
    nl0=$(wc -l < "$sigma_L0")
    nl1=$(wc -l < "$sigma_L1")
    nl2=$(wc -l < "$sigma_L2")

    if [ "$nl0" -eq "$nl1" ] && [ "$nl0" -eq "$nl2" ] && [ "$nl0" -gt 0 ]; then
        # columns: freq  s_L0  s_L1  s_L2  diff_L1(=L0-L1)  diff_L2(=L0-L2)
        paste "$sigma_L0" "$sigma_L1" "$sigma_L2" | \
            awk -v mu="$mu" '{print mu, $1, $2, $4, $6, $2-$4, $2-$6}' > "$sigma_job"
    else
        echo "WARNING: skipping sigma for mu=$mu (line counts: L0=$nl0 L1=$nl1 L2=$nl2)" >&2
    fi

    rm -f "$tmp" "$sigma_L0" "$sigma_L1" "$sigma_L2"
}

export -f run_mu run_sim
export BASE_CONFIG SIGMA_FILE GAUGE_FILE TS_DIR

# ---------- mu sweep ----------
mu_values=( $(seq 2.3 0.1 3.5) )
count=${#mu_values[@]}
first_mu=${mu_values[0]}
mid_mu=${mu_values[$(( (count-1)/2 ))]}
last_mu=${mu_values[$(( count-1 ))]}

for mu in "${mu_values[@]}"; do
    if [ "$mu" = "$first_mu" ] || [ "$mu" = "$mid_mu" ] || [ "$mu" = "$last_mu" ]; then
        do_gauge=1
    else
        do_gauge=0
    fi

    run_mu "$mu" "$do_gauge" &

    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
done

wait

# ---------- Merge per-job files into final outputs (sorted by mu) ----------
for f in $(ls "${SIGMA_FILE}".mu_* 2>/dev/null | sort -t_ -k2 -n); do
    cat "$f" >> "$SIGMA_FILE"
    rm -f "$f"
done

for f in $(ls "${GAUGE_FILE}".mu_* 2>/dev/null | sort -t_ -k2 -n); do
    cat "$f" >> "$GAUGE_FILE"
    rm -f "$f"
done

echo ""
echo "Sweep finished."
echo "Sigma results : $SIGMA_FILE"
echo "Gauge results : $GAUGE_FILE"
echo "Timeseries    : $TS_DIR/"
