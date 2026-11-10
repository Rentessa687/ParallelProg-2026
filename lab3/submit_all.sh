#!/bin/bash
mkdir -p logs results
for P in ${@:-1 2 4 8 16}; do
  sbatch --ntasks="$P" bench.sbatch
done
