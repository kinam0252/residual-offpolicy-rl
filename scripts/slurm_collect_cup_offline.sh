#!/bin/bash
#SBATCH --job-name=cup_offline
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/cup_offline_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/cup_offline_%j.err
#SBATCH --array=0-7

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0

cd ~/Repos/Intern/residual-offpolicy-rl

# Split 27 cup positions across 8 array tasks
ALL_IDS=($(seq 0 26))
N=${#ALL_IDS[@]}
NTASKS=8
IDX=$SLURM_ARRAY_TASK_ID

# Compute this task's slice
START=$(( IDX * N / NTASKS ))
END=$(( (IDX + 1) * N / NTASKS ))
SLICE="${ALL_IDS[@]:$START:$((END - START))}"
echo "Task $IDX: episodes $SLICE"

python3 resfit/rl_finetuning/scripts/collect_offline_data_cup.py \
    --groot_checkpoint ~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000 \
    --episode_ids $SLICE \
    --num_episodes_per_env 100 \
    --max_episode_steps 500 \
    --output_dir outputs/offline_cup_batch \
    --resume
