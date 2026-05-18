#!/bin/bash
# Submit multiple drawer checkpoint sweep jobs (core_extra, lock-based work stealing)
# Each job picks up unclaimed (ckpt, drawer) combos automatically.
#
# Usage: bash scripts/slurm/submit_sweep_drawer_ckpt.sh [NUM_JOBS]
set -e

NUM_JOBS=${1:-4}
REPO=/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
SLURM_DIR=/home/nas_main/kinamkim/slurms
OUT_DIR=${REPO}/outputs/sweep_drawer_ckpt

mkdir -p ${SLURM_DIR} ${OUT_DIR}

for i in $(seq 1 ${NUM_JOBS}); do
    JOB_ID=$(sbatch --parsable \
        --qos=extra \
        --gres=gpu:1 \
        --mem=80G \
        --cpus-per-task=4 \
        --job-name=drawer_ckpt_sweep \
        --output=${SLURM_DIR}/drawer_ckpt_sweep_%j.out \
        --error=${SLURM_DIR}/drawer_ckpt_sweep_%j.err \
        --time=02:00:00 \
        --wrap="
set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=\${CUDA_VISIBLE_DEVICES%%,*}
NV=\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=\$HOME/.local/lib:\$HOME/lib-compat:\$NV/cuda_runtime/lib:\$NV/cublas/lib:\$NV/cudnn/lib:\$NV/cufft/lib:\$NV/cusolver/lib:\$NV/cusparse/lib:\$NV/nvjitlink/lib:\$NV/cuda_nvrtc/lib:\$NV/nccl/lib:\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=~/fake_cuda
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd ${REPO}
python3 scripts/workloads/sweep_drawer_ckpt.py \
    --ckpt_dir ~/DATA/INTERN/training/groot_drawer_sim_33ep \
    --output_dir ${OUT_DIR} \
    --drawers 2 3 4 \
    --episodes 3 \
    --physics_drawer
echo 'SWEEP JOB DONE'
")
    echo "Submitted job ${i}/${NUM_JOBS}: ${JOB_ID}"
done

echo "All ${NUM_JOBS} sweep jobs submitted!"
echo "Results will be in: ${OUT_DIR}"
echo "Monitor: squeue -u \$USER -n drawer_ckpt_sweep"
