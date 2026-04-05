#!/bin/bash
#SBATCH --job-name=mj_v4_alr
#SBATCH --partition=sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_train_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_train_%j.err

set -e
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export WANDB_API_KEY="$(cat /home/nas_main/kinamkim/Keys/wandb_api_key)"

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/mujoco_td3_v4_highscale/${TIMESTAMP}"

echo "=== MuJoCo TD3 v4: Higher action_scale + actor_lr ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Config: asymmetric, action_scale=0.3, actor_lr=3e-5, 10 envs, 50K steps"

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/mujoco_cube_perturb_table.json \
    --asymmetric_critic \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 1024 \
    --actor_lr 3e-5 \
    --critic_lr 1e-4 \
    --action_scale 0.3 \
    --reward_type dense_clipped \
    --success_threshold 0.03 \
    --max_episode_steps 300 \
    --total_timesteps 50000 \
    --batch_size 256 \
    --buffer_size 200000 \
    --learning_starts 1000 \
    --critic_warmup_steps 500 \
    --num_updates_per_iteration 4 \
    --update_every_n_steps 1 \
    --gamma 0.99 \
    --n_step 3 \
    --stddev_max 0.05 \
    --stddev_min 0.05 \
    --eval_interval 1000 \
    --eval_num_episodes 5 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_mode online \
    --wandb_project mujoco-franka-residual-td3 \
    --wandb_name "v4_highscale_as03_${TIMESTAMP}"
