#!/bin/bash
#SBATCH --job-name=stk_base_eval
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --nodelist=worker-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/stk_base_eval_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/stk_base_eval_%j.err

. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export CUDA_HOME=~/fake_cuda
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python scripts/eval_base_policy_stack.py \
    --eval_positions_file configs/stack_cube_positions.json \
    --groot_checkpoint ~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000 \
    --num_envs 20 \
    --num_episodes 3 \
    --max_episode_steps 300 \
    --output_dir outputs/stack_base_eval_videos \
    --reward_type dense
