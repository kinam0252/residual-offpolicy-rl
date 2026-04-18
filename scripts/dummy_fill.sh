#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --job-name=dummy_fill
#SBATCH --output=/dev/null
#SBATCH --time=00:05:00
sleep 300
