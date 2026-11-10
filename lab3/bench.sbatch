#!/bin/bash
#SBATCH --job-name=jacobi_mpi
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=mem32
#SBATCH --time=00:30:00
#SBATCH --output=logs/jacobi_%j.out
#SBATCH --error=logs/jacobi_%j.err

SIZES="${SIZES:-200 400 800 1200 1600 2000}"
REPEATS="${REPEATS:-3}"
P="$SLURM_NTASKS"
cd "$SLURM_SUBMIT_DIR"
mkdir -p results
RES="results/res_${P}.csv"
echo "n,procs,run,iterations,time_ms,residual" > "$RES"

module load intel/mpi4

for n in $SIZES; do
  for r in $(seq 1 "$REPEATS"); do
    out="results/out_${n}_${P}_${r}.txt"
    mpirun -np "$P" ./jacobi_mpi "data/input_${n}.txt" "$out" > /dev/null
    it=$(awk '/^iterations/{print $2}' "$out")
    t=$(awk '/^time_ms/{print $2}' "$out")
    res=$(awk '/^residual/{print $2}' "$out")
    echo "${n},${P},${r},${it},${t},${res}" >> "$RES"
    rm -f "$out"
  done
done
