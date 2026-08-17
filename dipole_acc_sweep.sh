#!/bin/bash
# dipole_acc_sweep.sh — mu sweep computing the DIPOLE ACCELERATION (HHG) spectrum
# for L0, L1, L2, driving each (mu, level) at its linear-response RESONANCE frequency.
#
# This mirrors combined_sweep.sh in everything that matters for file handling
# (robust config parsing, frozen config, per-job parallelism, per-level output
# directories, lattice_points saved once), but the payload is different:
#
#   * field mode      -> "time_impulse"  (Gaussian-enveloped pulse that actually
#                        uses `omega`; note: "ddf" uses ddf_omega and would IGNORE
#                        the resonance frequency, so time_impulse is required here)
#   * run_sigma_ext   -> false
#   * run_dipole_acc  -> true
#   * omega           -> the resonance frequency for that (mu, level), taken from
#                        the table produced by ploting/Extract.py
#                        (resonance_vs_mu_<sweep>.txt, columns:
#                         mu  omega_res_L0_eV  omega_res_L1_eV  omega_res_L2_eV)
#
# Levels (identical definitions to combined_sweep.sh):
#   L0  spin_on=false zeeman_induced=false zeeman_external=false self_consistent_phase=false
#   L1  spin_on=true  zeeman_induced=true  zeeman_external=false self_consistent_phase=false
#   L2  spin_on=true  zeeman_induced=true  zeeman_external=false self_consistent_phase=true
#
# Usage:
#   ./dipole_acc_sweep.sh configs/graphene_armchair.toml [resonance_table.txt]
#
#   If the resonance table is omitted it defaults to
#     <DATA_DIR>/resonance_vs_mu_sweep_data_mu_<formation>_<shape>_<Nx>x<Ny>_rot<angle>.txt
#
# Optional pulse-shaping overrides (env vars; the config value is kept if unset).
# The committed configs are tuned for the ddf/sigma_ext run (e.g. t_shift=1000 with
# t_max=180), which is NOT suitable for a time_impulse HHG pulse — set these to drive
# a real pulse inside the simulation window:
#   T_MAX=...  T_SHIFT=...  SIGMA_GAUS=...  INTENSITY=...
#   LEVELS="L0 L1 L2"   (restrict which levels to run)
#
# Output:
#   <DATA_DIR>/dipole_sweep_data_mu_<formation>_<shape>_<Nx>x<Ny>_rot<angle>/
#       lattice_points.txt                 (once, top level — level-independent geometry)
#       bond_indices.txt                   (once, top level — level-independent geometry)
#       resonance_used.txt                 (mu level omega_drive_eV — record of drives)
#       L{0,1,2}_mu_<mu>/
#           dipole_acc.txt                 (omega  Re  Im)
#           dipole_time_evolution.txt
#           current_time_evolution.txt          (current)
#           spin_current_time_evolution.txt     (spin current; L1/L2)
#           B_ind_z_time_evolution.txt          (induced magnetic field; L1/L2)
#           J_bond_time_evolution.txt           (bond currents; L1/L2, L2=self-consistent)
#           A_ind_time_evolution.txt            (induced vector potential; L2 only)
#       L2 additionally keeps the raw self-consistent variants:
#           J_bond_sc_time_evolution.txt B_ind_z_sc_time_evolution.txt B_ind_z_curl_time_evolution.txt
#
#   (saved observables mirror combined_sweep.sh, minus sigma_ext, plus the dipole
#    acceleration and the induced vector potential A_ind.)

NCPU=$(nproc)
AVAIL_MEM_GB=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS MKL_NUM_THREADS

: "${MEM_PER_JOB_GB:=1}"
: "${LEVELS:=L0 L1 L2}"
CPU_JOBS=$(( (NCPU) / OMP_NUM_THREADS)) # divide by 2 (NCPU / 2) on ucloud
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
RES_TABLE_ARG=$2

if [ -z "$BASE_CONFIG" ]; then
    echo "Usage: ./dipole_acc_sweep.sh configs/your_config.toml [resonance_table.txt]"
    exit 1
fi

formation=$(grep '^\s*formation\b' "$BASE_CONFIG" | grep -v 'formation_shape' | awk -F'"' '{print $2}' | tr -d '\r')
formation_shape=$(grep '^\s*formation_shape' "$BASE_CONFIG" | awk -F'"' '{print $2}' | tr -d '\r')
size_x=$(   grep '^\s*size_x'             "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')
size_y=$(   grep '^\s*size_y'             "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')
rotation=$( grep '^\s*rotation_angle_deg' "$BASE_CONFIG" | awk -F'=' '{print $2}' | tr -d ' \t\r')

if [ -z "$formation" ] || [ -z "$formation_shape" ] || [ -z "$size_x" ] || [ -z "$size_y" ] || [ -z "$rotation" ]; then
    echo "ERROR: could not extract one or more config fields from $BASE_CONFIG"
    echo "  formation='$formation'  formation_shape='$formation_shape'  size_x='$size_x'  size_y='$size_y'  rotation='$rotation'"
    exit 1
fi

# Freeze the config so concurrent sweeps sharing the same config cannot race.
FROZEN_CONFIG=$(mktemp --suffix=.toml)
cp "$BASE_CONFIG" "$FROZEN_CONFIG"
BASE_CONFIG="$FROZEN_CONFIG"

if [ -d "/work/Home/scr/data_LLM" ]; then
    DATA_DIR="/work/Home/scr/data_LLM"
else
    DATA_DIR="data_LLM"
fi
mkdir -p "$DATA_DIR"

SWEEP_NAME="sweep_data_mu_${formation}_${formation_shape}_${size_x}x${size_y}_rot${rotation}"

# Resolve the resonance table: explicit arg wins, else the default name in DATA_DIR.
if [ -n "$RES_TABLE_ARG" ]; then
    RES_TABLE="$RES_TABLE_ARG"
else
    RES_TABLE="$DATA_DIR/resonance_vs_mu_${SWEEP_NAME}.txt"
fi

if [ ! -f "$RES_TABLE" ]; then
    echo "ERROR: resonance table not found: $RES_TABLE"
    echo "  Generate it first with:  python3 ploting/Extract.py $DATA_DIR/$SWEEP_NAME"
    echo "  or pass an explicit table path as the 2nd argument."
    rm -f "$FROZEN_CONFIG"
    exit 1
fi

DIP_DIR="$DATA_DIR/dipole_${SWEEP_NAME}"
rm -rf "$DIP_DIR"
mkdir -p "$DIP_DIR"

trap 'rm -f "$FROZEN_CONFIG"' EXIT

echo "Starting dipole-acceleration mu sweep using $BASE_CONFIG"
echo "Resonance table : $RES_TABLE"
echo "Output dir      : $DIP_DIR"
echo ""

# ---------- Helper: run one sim and return its output dir ----------
run_sim() {
    local cfg=$1
    local out
    out=$(./sim_mkl "$cfg" 2>&1)
    echo "$out" | grep "All outputs saved under" | awk '{print $NF}' | tr -d '"'
}

# ---------- Run a single (level, mu) dipole-acceleration simulation ----------
run_level() {
    local level=$1
    local mu=$2
    local w=$3

    # Skip levels with no resonance (NaN / blank in the table).
    case "$w" in
        nan|NaN|NAN|""|None) echo "  skip $level mu=$mu (no resonance)"; return ;;
    esac

    local tmp
    tmp=$(mktemp --suffix=.toml)

    # Common edits: drive at resonance, time_impulse pulse, dipole acc on.
    local sed_args=(
        -e "s/^mu *= *.*/mu = $mu/"
        -e "s/^omega *= *.*/omega = $w/"
        -e 's/^mode *= *.*/mode = "time_impulse"/'
        -e 's/^run_sigma_ext *= *.*/run_sigma_ext = false/'
        -e 's/^run_dipole_acc *= *.*/run_dipole_acc = true/'
    )

    # Optional pulse-shaping overrides (only applied if the env var is set).
    [ -n "$T_MAX" ]      && sed_args+=( -e "s/^t_max *= *.*/t_max = $T_MAX/" )
    [ -n "$T_SHIFT" ]    && sed_args+=( -e "s/^t_shift *= *.*/t_shift = $T_SHIFT/" )
    [ -n "$SIGMA_GAUS" ] && sed_args+=( -e "s/^sigma_gaus *= *.*/sigma_gaus = $SIGMA_GAUS/" )
    [ -n "$INTENSITY" ]  && sed_args+=( -e "s/^intensity *= *.*/intensity = $INTENSITY/" )

    # Level-specific physics flags.
    case "$level" in
        L0)
            sed_args+=(
                -e 's/^spin_on *= *.*/spin_on = false/'
                -e 's/^zeeman_induced *= *.*/zeeman_induced = false/'
                -e 's/^zeeman_external *= *.*/zeeman_external = false/'
                -e 's/^self_consistent_phase *= *.*/self_consistent_phase = false/'
            ) ;;
        L1)
            sed_args+=(
                -e 's/^spin_on *= *.*/spin_on = true/'
                -e 's/^zeeman_induced *= *.*/zeeman_induced = true/'
                -e 's/^zeeman_external *= *.*/zeeman_external = false/'
                -e 's/^self_consistent_phase *= *.*/self_consistent_phase = false/'
            ) ;;
        L2)
            sed_args+=(
                -e 's/^spin_on *= *.*/spin_on = true/'
                -e 's/^zeeman_induced *= *.*/zeeman_induced = true/'
                -e 's/^zeeman_external *= *.*/zeeman_external = false/'
                -e 's/^self_consistent_phase *= *.*/self_consistent_phase = true/'
            ) ;;
        *)
            echo "  unknown level '$level'" >&2
            rm -f "$tmp"; return ;;
    esac

    sed "${sed_args[@]}" "$BASE_CONFIG" > "$tmp"

    local dir
    dir=$(run_sim "$tmp")
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        local dest="${DIP_DIR}/${level}_mu_${mu}"
        mkdir -p "$dest"
        # Save geometry files once at the top level of DIP_DIR (level-independent).
        [ -f "$dir/lattice_points.txt" ] && [ ! -f "$DIP_DIR/lattice_points.txt" ] && \
            cp "$dir/lattice_points.txt" "$DIP_DIR/"
        [ -f "$dir/bond_indices.txt" ] && [ ! -f "$DIP_DIR/bond_indices.txt" ] && \
            cp "$dir/bond_indices.txt" "$DIP_DIR/"

        # Observables common to every run (spin-current is absent for L0 -> guarded).
        for f in dipole_acc.txt dipole_time_evolution.txt \
                  current_time_evolution.txt spin_current_time_evolution.txt; do
            [ -f "$dir/$f" ] && cp "$dir/$f" "$dest/"
        done

        if [ "$level" = "L2" ]; then
            # For L2 use self-consistent bond currents (Eq. 16) and curl(A_ind) Zeeman B
            # under the canonical names, plus keep the raw variants and the vector potential.
            [ -f "$dir/J_bond_sc_time_evolution.txt" ] && \
                cp "$dir/J_bond_sc_time_evolution.txt" "$dest/J_bond_time_evolution.txt"
            [ -f "$dir/B_ind_z_curl_time_evolution.txt" ] && \
                cp "$dir/B_ind_z_curl_time_evolution.txt" "$dest/B_ind_z_time_evolution.txt"
            for f in J_bond_sc_time_evolution.txt B_ind_z_sc_time_evolution.txt \
                      B_ind_z_curl_time_evolution.txt A_ind_time_evolution.txt; do
                [ -f "$dir/$f" ] && cp "$dir/$f" "$dest/"
            done
        else
            # L0 / L1: magnetic field, bond currents, and (if present) vector potential.
            for f in J_bond_time_evolution.txt B_ind_z_time_evolution.txt \
                      A_ind_time_evolution.txt; do
                [ -f "$dir/$f" ] && cp "$dir/$f" "$dest/"
            done
        fi

        [ -f "$dest/dipole_acc.txt" ] || echo "  WARNING: no dipole_acc.txt for $level mu=$mu" >&2
        rm -rf "$dir"
    else
        echo "  WARNING: no $level dir for mu=$mu" >&2
    fi

    rm -f "$tmp"
}

# ---------- Per-mu worker (runs the active levels sequentially) ----------
run_mu() {
    local mu=$1 w0=$2 w1=$3 w2=$4
    echo "Running mu = $mu  (drive eV: L0=$w0 L1=$w1 L2=$w2)"

    echo "$LEVELS" | grep -qw "L0" && run_level L0 "$mu" "$w0"
    echo "$LEVELS" | grep -qw "L1" && run_level L1 "$mu" "$w1"
    echo "$LEVELS" | grep -qw "L2" && run_level L2 "$mu" "$w2"
}

export -f run_mu run_level run_sim
export BASE_CONFIG DIP_DIR LEVELS T_MAX T_SHIFT SIGMA_GAUS INTENSITY

# ---------- Record the drive frequencies actually used ----------
{
    echo "# mu level omega_drive_eV"
    grep -v '^[[:space:]]*#' "$RES_TABLE" | while read -r mu w0 w1 w2; do
        [ -z "$mu" ] && continue
        ml=$(printf '%.2f' "$mu")
        echo "$LEVELS" | grep -qw "L0" && echo "$ml L0 $w0"
        echo "$LEVELS" | grep -qw "L1" && echo "$ml L1 $w1"
        echo "$LEVELS" | grep -qw "L2" && echo "$ml L2 $w2"
    done
} > "$DIP_DIR/resonance_used.txt"

# ---------- mu sweep (driven directly from the resonance table) ----------
while read -r mu_raw w0 w1 w2; do
    [ -z "$mu_raw" ] && continue
    case "$mu_raw" in \#*) continue ;; esac

    # Normalise mu label to 2 decimals to match combined_sweep.sh dir naming.
    mu=$(printf '%.2f' "$mu_raw")

    run_mu "$mu" "$w0" "$w1" "$w2" &

    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
done < <(grep -v '^[[:space:]]*#' "$RES_TABLE")

wait

echo ""
echo "Dipole-acceleration sweep finished."
echo "Output dir   : $DIP_DIR/  (L{0,1,2}_mu_<mu>/dipole_acc.txt per mu value)"
echo "Drives used  : $DIP_DIR/resonance_used.txt"
