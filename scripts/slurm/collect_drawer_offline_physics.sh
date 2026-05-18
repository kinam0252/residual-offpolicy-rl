#!/bin/bash
# Submit drawer offline data collection jobs (physics_drawer=True, ckpt-75k)
# Work-stealing: each job picks unclaimed episodes automatically.
#
# Usage: bash scripts/slurm/collect_drawer_offline_physics.sh [NUM_JOBS] [QOS]
#   QOS: share (default), extra, core_extra
set -e

NUM_JOBS=${1:-4}
QOS=${2:-share}
REPO=/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
SLURM_DIR=/home/nas_main/kinamkim/slurms
OUT_DIR=${REPO}/outputs/offline_drawer_physics_75k

mkdir -p ${SLURM_DIR} ${OUT_DIR}

for i in $(seq 1 ${NUM_JOBS}); do
    JOB_ID=$(sbatch --parsable \
        --qos=${QOS} \
        --gres=gpu:1 \
        --mem=80G \
        --cpus-per-task=4 \
        --job-name=drawer_offline_phys \
        --output=${SLURM_DIR}/drawer_offline_phys_%j.out \
        --error=${SLURM_DIR}/drawer_offline_phys_%j.err \
        --time=04:00:00 \
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
python3 scripts/workloads/collect_drawer_offline_physics.py \
    --groot_checkpoint ~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-75000 \
    --active_drawers 2 3 \
    --num_episodes 150 \
    --output_dir ${OUT_DIR} \
    --reward_type dense
")
    echo "Submitted job ${i}/${NUM_JOBS}: ${JOB_ID} (qos=${QOS})"
done

echo ""
echo "All ${NUM_JOBS} jobs submitted."
echo "Output: ${OUT_DIR}"
echo "Monitor: ls ${OUT_DIR}/*.npz | wc -l"
