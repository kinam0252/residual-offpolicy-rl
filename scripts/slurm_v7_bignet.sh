#!/bin/bash
#SBATCH --job-name=mj_v7_bignet
#SBATCH --partition=core
#SBATCH --qos=core-extra
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
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export WANDB_API_KEY="$(cat /home/nas_main/kinamkim/Keys/wandb_api_key)"

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/mujoco_td3_v7_bignet/${TIMESTAMP}"

echo "=== v7_bignet: Big networks (actor=1024, critic=2048) ==="
echo "Job $SLURM_JOB_ID | $(hostname) | CUDA=$CUDA_VISIBLE_DEVICES"
python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/mujoco_cube_perturb_table.json \
    --asymmetric_critic \
    --actor_hidden_dim 1024 \
    --critic_hidden_dim 2048 \
    --actor_lr 1e-05 \
    --critic_lr 0.0001 \
    --action_scale 0.1 \
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
    --wandb_mode online --wandb_project mujoco-franka-residual-td3 \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_name "v7_bignet_${TIMESTAMP}" \
    --checkpoint_interval 5000 \
    --async_eval \
    --ema_alpha 0.9
