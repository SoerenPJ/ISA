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
#   data/gauge_metrics_mu_<formation>_<formation_shape>_<Nx>x<Ny>_rot<angle>.txt
#       columns: mu  level  corr_flux  alpha_flux  mean_rel_peak  max_rel_peak
#                mean_ratio_peak  rms_flux_peak  dynamic_range  mean_rel_curl_peak
#
#   data/sweep_data_mu_.../
#       lattice_points.txt          (once, top-level)
#       L{0,1,2}_mu_<mu>/sigma_ext.txt   (and other per-level files)

NCPU=$(nproc)
AVAIL_MEM_GB=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS MKL_NUM_THREADS

: "${MEM_PER_JOB_GB:=1}"
: "${LEVELS:=L0 L1 L2}"
: "${APPEND_GAUGE:=0}"
CPU_JOBS=$(( (NCPU ) / OMP_NUM_THREADS)) # divid by 2 (NCPU / 2) on ucloud
MAX_JOBS_MEM=15 #$(( (AVAIL_MEM_GB * 95 / 100) / MEM_PER_JOB_GB ))
[ "$MAX_JOBS_MEM" -lt 1 ] && MAX_JOBS_MEM=1
MAX_JOBS=$(( CPU_JOBS < MAX_JOBS_MEM ? CPU_JOBS : MAX_JOBS_MEM ))
[ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1

echo "Detected $NCPU CPU threads"
echo "Available RAM: ${AVAIL_MEM_GB} GB  (MEM_PER_JOB_GB=${MEM_PER_JOB_GB})"
echo "Using $OMP_NUM_THREADS threads per simulation"
echo "Running up to $MAX_JOBS simulations in parallel (RAM cap: $MAX_JOBS_MEM, CPU cap: $CPU_JOBS)"
echo "Active levels: $LEVELS"

BASE_CONFIG=$1

if [ -z "$BASE_CONFIG" ]; then
    echo "Usage: ./combined_sweep.sh configs/your_config.toml"
    exit 1
fi

formation=$(grep '^\s*formation\b' "$BASE_CONFIG" | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(grep '^\s*formation_shape' "$BASE_CONFIG" | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(   grep '^\s*size_x'             "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(   grep '^\s*size_y'             "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$( grep '^\s*rotation_angle_deg' "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')

# Freeze the config into a private temp file so that concurrent sweeps sharing
# the same config file cannot corrupt each other's per-mu jobs via a race condition.
FROZEN_CONFIG=$(mktemp --suffix=.toml)
cp "$BASE_CONFIG" "$FROZEN_CONFIG"
BASE_CONFIG="$FROZEN_CONFIG"

if [ -z "$formation" ] || [ -z "$formation_shape" ] || [ -z "$size_x" ] || [ -z "$size_y" ] || [ -z "$rotation" ]; then
    echo "ERROR: could not extract one or more config fields from $BASE_CONFIG"
    echo "  formation='$formation'  formation_shape='$formation_shape'  size_x='$size_x'  size_y='$size_y'  rotation='$rotation'"
    echo "  Run:  cat -A \"$BASE_CONFIG\" | grep -E 'formation|size_x|size_y|rotation_angle_deg'"
    echo "  to inspect the actual characters in those lines."
    exit 1
fi

if [ -d "/work/Home/scr/data_LLM" ]; then
    DATA_DIR="/work/Home/scr/data_LLM"
else
    DATA_DIR="data_LLM"
fi
mkdir -p "$DATA_DIR"

GAUGE_FILE="$DATA_DIR/gauge_metrics_mu_${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}.txt"
TS_DIR="$DATA_DIR/gauge_ts_mu_${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"
SWEEP_DIR="$DATA_DIR/sweep_data_mu_${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

rm -f "${GAUGE_FILE}".mu_*
if [ "$APPEND_GAUGE" != "1" ]; then
    rm -f "$GAUGE_FILE"
    rm -rf "$TS_DIR" "$SWEEP_DIR"
    echo "# mu level corr_flux alpha_flux mean_rel_peak max_rel_peak mean_ratio_peak rms_flux_peak dynamic_range mean_rel_curl_peak" > "$GAUGE_FILE"
fi
mkdir -p "$TS_DIR" "$SWEEP_DIR"

trap 'rm -f "${GAUGE_FILE}".mu_* "$FROZEN_CONFIG"' EXIT

echo "Starting combined mu sweep using $BASE_CONFIG"
echo "Gauge output  : $GAUGE_FILE"
echo "Timeseries dir: $TS_DIR"
echo "Sweep data dir: $SWEEP_DIR"
echo ""

# ---------- Helper: run one sim and return its output dir ----------
run_sim() {
    local cfg=$1
    local out
    out=$(./sim_mkl "$cfg" 2>&1)
    echo "$out" | grep "All outputs saved under" | awk '{print $NF}' | tr -d '"'
}

# ---------- Per-mu worker ----------
run_mu() {
    local mu=$1
    local do_gauge=$2
    echo "Running mu = $mu"

    local tmp
    tmp=$(mktemp --suffix=.toml)

    local gauge_job="${GAUGE_FILE}.mu_${mu}"

    # ------------------------------------------------------------------ L0
    if echo "$LEVELS" | grep -qw "L0"; then
        sed -e "s/^mu *= *.*/mu = $mu/" \
            -e "s/spin_on *= *true/spin_on = false/" \
            -e "s/zeeman_induced *= *true/zeeman_induced = false/" \
            -e "s/zeeman_external *= *true/zeeman_external = false/" \
            -e "s/self_consistent_phase *= *true/self_consistent_phase = false/" \
            "$BASE_CONFIG" > "$tmp"

        local dir
        dir=$(run_sim "$tmp")
        if [ -n "$dir" ] && [ -d "$dir" ]; then
            if [ "$do_gauge" = "1" ]; then
                local m
                m=$(python3 ploting/gauge_batch.py "$dir" 2>>"${GAUGE_FILE%.txt}.log")
                [ -z "$m" ] && m="nan nan nan nan nan nan nan nan"
                echo "$mu L0 $m" >> "$gauge_job"
                MPLBACKEND=Agg python3 ploting/gauge_comparison.py "$dir" &>/dev/null
                local ts="$dir/gauge_plots/flux_comparison_timeseries.txt"
                [ -f "$ts" ] && cp "$ts" "${TS_DIR}/L0_mu_${mu}.txt"
            fi
            local dest_L0="${SWEEP_DIR}/L0_mu_${mu}"
            mkdir -p "$dest_L0"
            # Save lattice_points.txt and bond_indices.txt once at the top level of SWEEP_DIR
            [ -f "$dir/lattice_points.txt" ] && [ ! -f "$SWEEP_DIR/lattice_points.txt" ] && \
                cp "$dir/lattice_points.txt" "$SWEEP_DIR/"
            [ -f "$dir/bond_indices.txt" ] && [ ! -f "$SWEEP_DIR/bond_indices.txt" ] && \
                cp "$dir/bond_indices.txt" "$SWEEP_DIR/"
            for f in sigma_ext.txt J_bond_time_evolution.txt \
                      B_ind_z_time_evolution.txt current_time_evolution.txt \
                      spin_current_time_evolution.txt bond_indices.txt; do
                [ -f "$dir/$f" ] && cp "$dir/$f" "$dest_L0/"
            done
            rm -rf "$dir"
        else
            echo "WARNING: no L0 dir for mu=$mu" >&2
            [ "$do_gauge" = "1" ] && echo "$mu L0 nan nan nan nan nan nan nan nan" >> "$gauge_job"
        fi
    fi

    # ------------------------------------------------------------------ L1
    if echo "$LEVELS" | grep -qw "L1"; then
        sed -e "s/^mu *= *.*/mu = $mu/" \
            -e "s/spin_on *= *false/spin_on = true/" \
            -e "s/zeeman_induced *= *false/zeeman_induced = true/" \
            -e "s/zeeman_external *= *true/zeeman_external = false/" \
            -e "s/self_consistent_phase *= *true/self_consistent_phase = false/" \
            "$BASE_CONFIG" > "$tmp"

        local dir
        dir=$(run_sim "$tmp")
        if [ -n "$dir" ] && [ -d "$dir" ]; then
            if [ "$do_gauge" = "1" ]; then
                local m
                m=$(python3 ploting/gauge_batch.py "$dir" 2>>"${GAUGE_FILE%.txt}.log")
                [ -z "$m" ] && m="nan nan nan nan nan nan nan nan"
                echo "$mu L1 $m" >> "$gauge_job"
                MPLBACKEND=Agg python3 ploting/gauge_comparison.py "$dir" &>/dev/null
                local ts="$dir/gauge_plots/flux_comparison_timeseries.txt"
                [ -f "$ts" ] && cp "$ts" "${TS_DIR}/L1_mu_${mu}.txt"
            fi
            local dest_L1="${SWEEP_DIR}/L1_mu_${mu}"
            mkdir -p "$dest_L1"
            for f in sigma_ext.txt J_bond_time_evolution.txt \
                      B_ind_z_time_evolution.txt current_time_evolution.txt \
                      spin_current_time_evolution.txt bond_indices.txt; do
                [ -f "$dir/$f" ] && cp "$dir/$f" "$dest_L1/"
            done
            rm -rf "$dir"
        else
            echo "WARNING: no L1 dir for mu=$mu" >&2
            [ "$do_gauge" = "1" ] && echo "$mu L1 nan nan nan nan nan nan nan nan" >> "$gauge_job"
        fi
    fi

    # ------------------------------------------------------------------ L2
    if echo "$LEVELS" | grep -qw "L2"; then
        sed -e "s/^mu *= *.*/mu = $mu/" \
            -e "s/spin_on *= *false/spin_on = true/" \
            -e "s/zeeman_induced *= *false/zeeman_induced = true/" \
            -e "s/zeeman_external *= *true/zeeman_external = false/" \
            -e "s/self_consistent_phase *= *false/self_consistent_phase = true/" \
            "$BASE_CONFIG" > "$tmp"

        local dir
        dir=$(run_sim "$tmp")
        if [ -n "$dir" ] && [ -d "$dir" ]; then
            if [ "$do_gauge" = "1" ]; then
                local m
                m=$(python3 ploting/gauge_batch.py "$dir" 2>>"${GAUGE_FILE%.txt}.log")
                [ -z "$m" ] && m="nan nan nan nan nan nan nan nan"
                echo "$mu L2 $m" >> "$gauge_job"
                MPLBACKEND=Agg python3 ploting/gauge_comparison.py "$dir" &>/dev/null
                local ts="$dir/gauge_plots/flux_comparison_timeseries.txt"
                [ -f "$ts" ] && cp "$ts" "${TS_DIR}/L2_mu_${mu}.txt"
            fi
            local dest_L2="${SWEEP_DIR}/L2_mu_${mu}"
            mkdir -p "$dest_L2"
            # For L2 use self-consistent bond currents (Eq. 16) and curl(A_ind) Zeeman B
            [ -f "$dir/J_bond_sc_time_evolution.txt" ] && \
                cp "$dir/J_bond_sc_time_evolution.txt" "$dest_L2/J_bond_time_evolution.txt"
            [ -f "$dir/B_ind_z_curl_time_evolution.txt" ] && \
                cp "$dir/B_ind_z_curl_time_evolution.txt" "$dest_L2/B_ind_z_time_evolution.txt"
            for f in sigma_ext.txt J_bond_sc_time_evolution.txt \
                      B_ind_z_sc_time_evolution.txt B_ind_z_curl_time_evolution.txt \
                      current_time_evolution.txt spin_current_time_evolution.txt \
                      bond_indices.txt; do
                [ -f "$dir/$f" ] && cp "$dir/$f" "$dest_L2/"
            done
            rm -rf "$dir"
        else
            echo "WARNING: no L2 dir for mu=$mu" >&2
            [ "$do_gauge" = "1" ] && echo "$mu L2 nan nan nan nan nan nan nan nan" >> "$gauge_job"
        fi
    fi

    rm -f "$tmp"
}

export -f run_mu run_sim
export BASE_CONFIG GAUGE_FILE TS_DIR SWEEP_DIR LEVELS APPEND_GAUGE

# ---------- mu sweep ----------
mu_values=( $(seq 0.0 0.04 5) )
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

# ---------- Merge per-job gauge files into final output (sorted by mu) ----------
for f in $(ls "${GAUGE_FILE}".mu_* 2>/dev/null | sort -V); do
    cat "$f" >> "$GAUGE_FILE"
    rm -f "$f"
done

echo ""
echo "Sweep finished."
echo "Gauge results : $GAUGE_FILE"
echo "Timeseries    : $TS_DIR/"
echo "Sweep data    : $SWEEP_DIR/  (L{0,1,2}_mu_<mu>/ per mu value)"
