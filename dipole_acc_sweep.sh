#!/bin/bash
NCPU=$(nproc)

OMP_NUM_THREADS=1
MKL_NUM_THREADS=1

export OMP_NUM_THREADS
export MKL_NUM_THREADS

MAX_JOBS=$((NCPU / OMP_NUM_THREADS))

echo "Detected $NCPU CPU threads"
echo "Using $OMP_NUM_THREADS threads per simulation"
echo "Running up to $MAX_JOBS simulations in parallel"

BASE_CONFIG=$1
RES_TABLE=$2

if [ -z "$BASE_CONFIG" ] || [ -z "$RES_TABLE" ]; then
    echo "Usage: ./dipole_acc_sweep.sh configs/config.toml resonance_table.txt"
    exit 1
fi

formation=$(grep '^formation =' "$BASE_CONFIG" | awk -F\" '{print $2}')
size_x=$(grep '^size_x' "$BASE_CONFIG" | awk '{print $3}')
size_y=$(grep '^size_y' "$BASE_CONFIG" | awk '{print $3}')
rotation=$(grep '^rotation_angle_deg' "$BASE_CONFIG" | awk '{print $3}')

OUTFILE="dipole_acc_mu_${formation}_${size_x}x${size_y}_rot${rotation}.txt"

rm -f "$OUTFILE"
rm -f "${OUTFILE}".mu_*
echo "# mu freq dipole_base dipole_full diff" > "$OUTFILE"

echo "Starting μ sweep using $BASE_CONFIG"
echo "Output file: $OUTFILE"
echo ""

run_mu () {

    mu=$1
    omega_base=$2
    omega_full=$3

    echo "Running mu = $mu  omega = $omega_base"

    tmp=$(mktemp)
    dipole_base=$(mktemp)
    dipole_full=$(mktemp)
    job_out="${OUTFILE}.mu_${mu}"

    # ---------- BASELINE ----------
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/^omega *= *.*/omega = $omega_base/" \
        -e "s/self_consistent_phase = true/self_consistent_phase = false/" \
        -e "s/zeeman_external = true/zeeman_external = false/" \
        -e "s/zeeman_induced = true/zeeman_induced = false/" \
        -e "s/spin_on = true/spin_on = false/" \
        "$BASE_CONFIG" > "$tmp"

    out=$(./sim_mkl "$tmp")

    dir=$(echo "$out" | grep "All outputs saved under" | awk -F\" '{print $2}')
    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo "ERROR: could not determine output directory for mu=$mu"
        rm -f "$tmp" "$dipole_base" "$dipole_full"
        return
    fi

    if [ ! -f "$dir/dipole_acc.txt" ]; then
        echo "ERROR: dipole_acc.txt not found for mu=$mu"
        rm -rf "$dir"
        rm -f "$tmp" "$dipole_base" "$dipole_full"
        return
    fi

    cp "$dir/dipole_acc.txt" "$dipole_base"
    rm -rf "$dir"


    # ---------- FULL PHYSICS ----------
    sed -e "s/^mu *= *.*/mu = $mu/" \
        -e "s/^omega *= *.*/omega = $omega_full/" \
        -e "s/self_consistent_phase = false/self_consistent_phase = true/" \
        -e "s/zeeman_external = false/zeeman_external = true/" \
        -e "s/zeeman_induced = false/zeeman_induced = true/" \
        -e "s/spin_on = false/spin_on = true/" \
        "$BASE_CONFIG" > "$tmp"

    out=$(./sim_mkl "$tmp")

    dir=$(echo "$out" | grep "All outputs saved under" | awk -F\" '{print $2}')
    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo "ERROR: could not determine output directory for mu=$mu"
        rm -f "$tmp" "$dipole_base" "$dipole_full"
        return
    fi

    if [ ! -f "$dir/dipole_acc.txt" ]; then
        echo "ERROR: dipole_acc.txt not found for mu=$mu"
        rm -rf "$dir"
        rm -f "$tmp" "$dipole_base" "$dipole_full"
        return
    fi

    cp "$dir/dipole_acc.txt" "$dipole_full"
    rm -rf "$dir"


    lines_base=$(wc -l < "$dipole_base")
    lines_full=$(wc -l < "$dipole_full")

    if [ "$lines_base" -eq "$lines_full" ]; then
        paste "$dipole_base" "$dipole_full" | \
        awk -v mu=$mu '{
            freq=$1;
            base = ($2*$2 + $3*$3);
            full = ($5*$5 + $6*$6);
            print mu, freq, base, full, base-full;
        }' > "$job_out"
    else
        echo "WARNING: skipping mu=$mu (incomplete output)"
    fi

    rm -f "$tmp" "$dipole_base" "$dipole_full"
}

export -f run_mu
export BASE_CONFIG OUTFILE


# ---------- μ sweep ----------
while read mu omega_base omega_full
do
    run_mu $mu $omega_base $omega_full &

    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]
    do
        sleep 1
    done

done < <(tail -n +2 "$RES_TABLE" | head -n 30 )

wait

# ---------- Merge all per-job files into OUTFILE (single-threaded, no race) ----------
for f in $(ls "${OUTFILE}".mu_* 2>/dev/null | sort -t_ -k2 -n); do
    cat "$f" >> "$OUTFILE"
    rm -f "$f"
done

echo ""
echo "Sweep finished"
echo "Results saved in $OUTFILE"
