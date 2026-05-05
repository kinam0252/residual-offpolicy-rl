#!/bin/bash
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --job-name=dummy_sub
#SBATCH --output=/dev/null
#SBATCH --time=00:05:00
sleep 300
