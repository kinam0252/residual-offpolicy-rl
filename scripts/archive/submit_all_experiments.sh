#!/bin/bash
# Submit all experiments
set -e
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
SBATCH=/home/jovyan/.slurm-install/bin/sbatch

echo "Submitting v10_off50 (core-own)..."
$SBATCH scripts/slurm_v10_off50.sh

echo "Submitting v11_off80 (core-own)..."
$SBATCH scripts/slurm_v11_off80.sh

echo "Submitting v12_off100 (core-extra)..."
$SBATCH scripts/slurm_v12_off100.sh

echo "Submitting v13_off30 (core-extra)..."
$SBATCH scripts/slurm_v13_off30.sh

echo "Submitting v14_small (core-extra)..."
$SBATCH scripts/slurm_v14_small.sh

echo "Submitting v15_big (core-extra)..."
$SBATCH scripts/slurm_v15_big.sh

echo "Submitting v16_noasymm (core-extra)..."
$SBATCH scripts/slurm_v16_noasymm.sh

echo "Submitting v18_dense (core-extra)..."
$SBATCH scripts/slurm_v18_dense.sh

echo "Submitting v19_sparse (core-extra)..."
$SBATCH scripts/slurm_v19_sparse.sh

echo "Submitting v20_highlr (core-extra)..."
$SBATCH scripts/slurm_v20_highlr.sh

echo "Submitting v21_bigscale (core-extra)..."
$SBATCH scripts/slurm_v21_bigscale.sh

echo "Submitting v22_lowscale (core-extra)..."
$SBATCH scripts/slurm_v22_lowscale.sh

echo "Submitting v23_seed123 (core-extra)..."
$SBATCH scripts/slurm_v23_seed123.sh

echo "Submitting v24_seed456 (core-extra)..."
$SBATCH scripts/slurm_v24_seed456.sh

echo "Submitting v25_100k (core-extra)..."
$SBATCH scripts/slurm_v25_100k.sh

echo "All experiments submitted!"
