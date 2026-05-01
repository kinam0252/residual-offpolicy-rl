#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=groot_eval_sweep
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=72:00:00
#SBATCH --nodelist=worker-6

sleep infinity
