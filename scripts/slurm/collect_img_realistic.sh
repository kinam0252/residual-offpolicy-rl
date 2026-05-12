#!/bin/bash
#SBATCH --job-name=collect_realistic
#SBATCH --output=/home/nas_main/kinamkim/slurms/collect_realistic_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/collect_realistic_%j.err
#SBATCH --qos=extra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib:$HOME/lib-compat:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

TASK="${TASK:-pnp}"
POSITIONS="${POSITIONS:-configs/pnp_test_3.json}"
echo "[collect_realistic] Task=$TASK positions=$POSITIONS"

python resfit/rl_finetuning/scripts/collect_offline_data_with_images.py \
    --task "$TASK" \
    --episode_positions_file "$POSITIONS" \
    --num_episodes_per_env 1 \
    --max_episode_steps 300 \
    --output_dir "outputs/offline_data/${TASK}_realistic_test" \
    --rl_img_size 84 \
    --realistic \
    "$@"
