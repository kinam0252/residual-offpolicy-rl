#!/bin/bash
#SBATCH --job-name=dummy_w8
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --nodelist=worker-8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:30:00
sleep 1800
