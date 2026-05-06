#!/bin/bash
# Submit base policy rollout video jobs for all 5 tasks (core-on-sub, parallel)
set -e

REPO=/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
DATA=/home/nas_main/kinamkim/DATA/INTERN/training
SLURM_DIR=/home/nas_main/kinamkim/slurms
VIDEO_BASE=${REPO}/debug/rollout_videos

COMMON_ARGS="--num_envs 2 --num_episodes 2 --max_episode_steps 500 --camera back"

submit_task() {
    local TASK=$1
    local CKPT=$2

    sbatch --parsable \
        --partition=core \
        --qos=core-extra \
        --gres=gpu:1 \
        --mem=80G \
        --cpus-per-task=4 \
        --job-name=rollout_${TASK} \
        --output=${SLURM_DIR}/rollout_${TASK}_%j.out \
        --error=${SLURM_DIR}/rollout_${TASK}_%j.err \
        --time=00:15:00 \
        --exclude=worker-6 \
        --wrap="
set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=\$HOME/.local/lib:\$HOME/lib-compat:\$NV/cuda_runtime/lib:\$NV/cublas/lib:\$NV/cudnn/lib:\$NV/cufft/lib:\$NV/cusolver/lib:\$NV/cusparse/lib:\$NV/nvjitlink/lib:\$NV/cuda_nvrtc/lib:\$NV/nccl/lib:\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=~/fake_cuda
export PATH=~/bin:\$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
cd ${REPO}
python3 scripts/rollout_base_policy_video.py \
    --task ${TASK} \
    --groot_checkpoint ${CKPT} \
    --video_dir ${VIDEO_BASE}/${TASK} \
    ${COMMON_ARGS}
echo 'DONE: ${TASK}'
"
    echo "Submitted ${TASK}"
}

mkdir -p ${SLURM_DIR} ${VIDEO_BASE}

submit_task lift   ${DATA}/gr00t_sim_66ep/checkpoint-300000
submit_task cup    ${DATA}/groot_cup_sim_27ep/checkpoint-100000
submit_task pnp    ${DATA}/groot_pnp_sim_100ep/checkpoint-300000
submit_task stack  ${DATA}/groot_stack_sim_66ep/checkpoint-100000
submit_task drawer ${DATA}/groot_drawer_sim_33ep/checkpoint-100000

echo "All 5 jobs submitted!"
